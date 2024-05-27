#include "UMAPAnalysisPlugin.h"

#include <util/Icon.h>

#include <PointData/DimensionsPickerAction.h>
#include <PointData/InfoAction.h>

#pragma warning(disable:4477)       // annoy internal: print formatting warnings
#include <annoy/annoylib.h>
#include <hnswlib/space_ip.h>
#include <knncolle/Annoy/Annoy.hpp>
#include <knncolle/utils/Base.hpp>
#include <knncolle/utils/find_nearest_neighbors.hpp>
#include <knncolle/Hnsw/Hnsw.hpp>
#pragma warning(default:4477)

#include <QDebug>
#include <QtCore>

#ifdef USE_OPENMP
#include <omp.h>
#endif

Q_PLUGIN_METADATA(IID "studio.manivault.UMAPAnalysisPlugin")

using namespace mv;
using namespace mv::plugin;

namespace knncolle::hnsw_distances
{
    class InnerProduct : public hnswlib::InnerProductSpace {
    public:
        InnerProduct(size_t ndim) : hnswlib::InnerProductSpace(ndim) {}

        static float normalize(float raw) {
            return raw;
        }
    };
}

using knnAnnoyEuclidean = knncolle::Annoy<Annoy::Euclidean, int, scalar_t, scalar_t, int32_t, scalar_t>;
using knnAnnoyAngular   = knncolle::Annoy<Annoy::Angular, int, scalar_t, scalar_t, int32_t, scalar_t>;
using knnAnnoyDot       = knncolle::Annoy<Annoy::DotProduct, int, scalar_t, scalar_t, int32_t, scalar_t>;

using knnHnswEuclidean  = knncolle::Hnsw<knncolle::hnsw_distances::Euclidean, int, scalar_t, scalar_t>;
using knnHnswDot        = knncolle::Hnsw<knncolle::hnsw_distances::InnerProduct, int, scalar_t, scalar_t>;

static void normalizData(std::vector<scalar_t>& data) {
    float norm = 0.0f;
    for (const auto& val : data)
        norm += val * val;

    norm = 1.0f / (sqrtf(norm) + 1e-30f);

#pragma omp parallel
    for (std::int64_t i = 0; i < data.size(); i++)
        data[i] *= norm;
}


UMAPAnalysisPlugin::UMAPAnalysisPlugin(const PluginFactory* factory) :
    AnalysisPlugin(factory),
    _settingsAction(this),
    _knnSettingsAction(this),
    _outDimensions(2),
    _outputPoints(nullptr),
    _umapWorker(),
    _workerThread(nullptr)
{
}

UMAPAnalysisPlugin::~UMAPAnalysisPlugin()
{
    _workerThread.quit();           // Signal the thread to quit gracefully
    if (!_workerThread.wait(500))   // Wait for the thread to actually finish
        _workerThread.terminate();  // Terminate thread after 0.5 seconds

    deleteWorker();
}

void UMAPAnalysisPlugin::deleteWorker()
{
    if (_umapWorker)
    {
        _umapWorker->changeThread(QThread::currentThread());
        delete _umapWorker;
    }
}

void UMAPAnalysisPlugin::init()
{
    // Create UMAP output dataset (a points dataset which is derived from the input points dataset) and set the output dataset
    setOutputDataset(mv::data().createDerivedDataset("UMAP Embedding", getInputDataset(), getInputDataset()));

    const Dataset<Points> inputPoints = getInputDataset<Points>();
    _outputPoints = getOutputDataset<Points>();

    std::vector<scalar_t> initEmbeddingValues;
    initEmbeddingValues.resize(inputPoints->getNumPoints() * static_cast<size_t>(_outDimensions));
    _outputPoints->setData(initEmbeddingValues.data(), initEmbeddingValues.size() / _outDimensions, _outDimensions);
    events().notifyDatasetDataChanged(_outputPoints);

    // Set the dimension names as visible in the GUI
    _outputPoints->setDimensionNames({ "UMAP x", "UMAP y" });
    events().notifyDatasetDataDimensionsChanged(_outputPoints);

    // Add settings to UI
    _outputPoints->addAction(_settingsAction);
    _outputPoints->addAction(_knnSettingsAction);
    
    // Automatically focus on the UMAP data set
    _outputPoints->getDataHierarchyItem().select();
    _outputPoints->_infoAction->collapse();

    // Initialize current epoch action
    _settingsAction.getCurrentEpochAction().setString(QString::number(0));

    // Compute suggested number of epoch
    _settingsAction.getNumberOfEpochsAction().setValue(umappp::choose_num_epochs(-1, inputPoints->getNumPoints()));

    // Create UMAP worker, which will be executed in another thread
    // Start the analysis when the user clicks the start analysis push button
    connect(&_settingsAction.getStartAction(), &mv::gui::TriggerAction::triggered, this, [this] {

        // Disable actions during analysis
        _settingsAction.setStarted();
        _knnSettingsAction.setReadOnly(true);

        deleteWorker();

        getOutputDataset()->getTask().setRunning();

        Dataset<Points> inputPoints = getInputDataset<Points>();
        _umapWorker = new UMAPWorker(inputPoints, &getOutputDataset()->getTask(), _outDimensions, &_settingsAction, &_knnSettingsAction);

        _umapWorker->changeThread(&_workerThread);

        // To-Worker signals
        connect(this, &UMAPAnalysisPlugin::startWorker, _umapWorker, &UMAPWorker::compute);
        connect(this, &UMAPAnalysisPlugin::stopWorker, _umapWorker, &UMAPWorker::stop, Qt::DirectConnection);

        // From-Worker signals
        connect(_umapWorker, &UMAPWorker::embeddingUpdate, this, [this](const std::vector<scalar_t> embedding, int epoch) {
            getOutputDataset<Points>()->setData(embedding.data(), embedding.size() / _outDimensions, _outDimensions);

            _settingsAction.getCurrentEpochAction().setString(QString::number(epoch));

            events().notifyDatasetDataChanged(getOutputDataset());
            });

        connect(_umapWorker, &UMAPWorker::finished, this, [this]() {
            _settingsAction.setFinished();
            _knnSettingsAction.setReadOnly(false);
            getOutputDataset()->getTask().setFinished();
            });

        _workerThread.start();
        emit startWorker();
        });

    connect(&_settingsAction.getStopAction(), &mv::gui::TriggerAction::triggered, this, [this](bool checked) {
        emit stopWorker();
        }, 
        Qt::DirectConnection);

}


/// ////////// ///
/// UMAPWorker ///
/// ////////// ///

UMAPWorker::UMAPWorker(Dataset<Points>& inputPoints, DatasetTask* parentTask, int outDim, SettingsAction* settings, KnnSettingsAction* knnSettings):
    _umap(),
    _shouldStop(false),
    _inputDataset(inputPoints),
    _computeTask(nullptr),
    _parentTask(parentTask),
    _settingsAction(settings),
    _knnSettingsAction(knnSettings),
    _embedding(),
    _outDimensions(outDim)
{
    _embedding.resize(inputPoints->getNumPoints() * static_cast<size_t>(_outDimensions));
}

void UMAPWorker::changeThread(QThread* targetThread)
{
    this->moveToThread(targetThread);
}

void UMAPWorker::resetThread()
{
    changeThread(QCoreApplication::instance()->thread());
}

void UMAPWorker::compute()
{
    _computeTask = std::make_unique<mv::Task>(this, "UMAP analysis", Task::GuiScopes{ Task::GuiScope::DataHierarchy, Task::GuiScope::Foreground }, Task::Status::Idle);

    _computeTask->setParentTask(_parentTask);
    
    connect(_parentTask, &Task::requestAbort, this, [this]() -> void { _shouldStop = true; }, Qt::DirectConnection);

    _computeTask->setRunning();
    _computeTask->setProgress(0.0f);
    _computeTask->setProgressDescription("Initializing...");
    QCoreApplication::processEvents();

    _shouldStop = false;

    // Get the number of epochs from the settings
    const auto numberOfEpochs = _settingsAction->getNumberOfEpochsAction().getValue();

    // Create list of data from the enabled dimensions
    std::vector<scalar_t> data;
    std::vector<unsigned int> indices;

    // Extract the enabled dimensions from the data
    std::vector<bool> enabledDimensions = _inputDataset->getDimensionsPickerAction().getEnabledDimensions();

    const auto numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    size_t numPoints = _inputDataset->isFull() ? _inputDataset->getNumPoints() : _inputDataset->indices.size();
    data.resize(numPoints * numEnabledDimensions);

    for (int i = 0; i < _inputDataset->getNumDimensions(); i++)
        if (enabledDimensions[i])
            indices.push_back(i);

    _inputDataset->populateDataForDimensions<std::vector<scalar_t>, std::vector<unsigned int>>(data, indices);

    // determine threading
    unsigned int nThreads = 1;

#ifdef USE_OPENMP
    if (_knnSettingsAction->getMultithreadAction().isChecked())
    {
        nThreads = omp_get_max_threads();
        if (nThreads <= 0)
            nThreads = 1;
    }
#endif

    // compute knn
    knncolle::NeighborList<int, scalar_t> nearestNeighbors(numPoints);
    const KnnParameters knnParams = _knnSettingsAction->getKnnParameters();
    const int numNeighbors = knnParams.getK();
    {
        int nDim = numEnabledDimensions;

        qDebug() << "UMAP: compute knn: " << numNeighbors << " neighbors";

        std::unique_ptr<knncolle::Base<int, scalar_t, scalar_t>> searcher;

        if (knnParams.getKnnAlgorithm() == KnnLibrary::ANNOY) {

            if (knnParams.getKnnDistanceMetric() == KnnMetric::COSINE)
                searcher = std::make_unique<knnAnnoyAngular>(nDim, numPoints, data.data(), knnParams.getAnnoyNumTrees(), knnParams.getAnnoyNumChecks());
            else if (knnParams.getKnnDistanceMetric() == KnnMetric::DOT)
                searcher = std::make_unique<knnAnnoyDot>(nDim, numPoints, data.data(), knnParams.getAnnoyNumTrees(), knnParams.getAnnoyNumChecks());
            else
                searcher = std::make_unique<knnAnnoyEuclidean>(nDim, numPoints, data.data(), knnParams.getAnnoyNumTrees(), knnParams.getAnnoyNumChecks());
        }
        else // knnParams.getKnnAlgorithm() == KnnLibrary::HNSW
        {
            if (knnParams.getKnnDistanceMetric() == KnnMetric::COSINE)
            {
                normalizData(data);
                searcher = std::make_unique<knnHnswDot>(nDim, numPoints, data.data(), knnParams.getHNSWm(), knnParams.getHNSWef(), knnParams.getHNSWef());
            }
            else if (knnParams.getKnnDistanceMetric() == KnnMetric::DOT)
                searcher = std::make_unique<knnHnswDot>(nDim, numPoints, data.data(), knnParams.getHNSWm(), knnParams.getHNSWef(), knnParams.getHNSWef());
            else
                searcher = std::make_unique<knnHnswEuclidean>(nDim, numPoints, data.data(), knnParams.getHNSWm(), knnParams.getHNSWef(), knnParams.getHNSWef());
        }

        nearestNeighbors = knncolle::find_nearest_neighbors<int, scalar_t>(searcher.get(), numNeighbors, nThreads);

    }

    qDebug() << "UMAP: initializing...";

    _umap = UMAP();
    _umap.set_num_neighbors(numNeighbors);
    _umap.set_num_epochs(numberOfEpochs);

    // default is spectral
    if (_settingsAction->getInitializeAction().getCurrentText() == "Random")
        _umap.set_initialize(umappp::InitMethod::RANDOM);

    auto status = std::make_unique<UMAP::Status>(_umap.initialize(nearestNeighbors, _outDimensions, _embedding.data()));

    const auto updateEmbedding = [this, numPoints](int ep) -> void {
        _outEmbedding.assign(_embedding.begin(), _embedding.end());
        emit embeddingUpdate(_outEmbedding, ep);
        };

    updateEmbedding(0);

    if (numberOfEpochs == 0 || _shouldStop)
    {
        _shouldStop = false;
        emit finished();
        return;
    }

    qDebug() << "UMAP: start gradient descent: " << numberOfEpochs << " epochs";

    int iter = 1;
    // Iteratively update UMAP embedding
    for (; iter < numberOfEpochs; iter++)
    {
        if (_shouldStop)
            break;

        status->run(iter);

        if (iter % 10 == 0)
            updateEmbedding(iter);

        _computeTask->setProgress(iter / static_cast<float>(numberOfEpochs));
        _computeTask->setProgressDescription(QString("Epoch %1/%2").arg(QString::number(iter), QString::number(numberOfEpochs)));
        //_computeTask->setSubtaskStarted(iter);
        QCoreApplication::processEvents();
    }

    updateEmbedding(iter);

    qDebug() << "UMAP: total epochs: " << status->epoch() + 1;

    // Flag the analysis task as finished
    _computeTask->setFinished();
    //_computeTask->setSubtaskFinished(numberOfEpochs);

    emit finished();

    resetThread();
}

/// ////////////// ///
/// Plugin Factory ///
/// ////////////// ///

QIcon UMAPAnalysisPluginFactory::getIcon(const QColor& color /*= Qt::black*/) const
{
    return mv::gui::createPluginIcon("UMAP", color);
}

AnalysisPlugin* UMAPAnalysisPluginFactory::produce()
{
    return new UMAPAnalysisPlugin(this);
}

mv::DataTypes UMAPAnalysisPluginFactory::supportedDataTypes() const
{
    DataTypes supportedTypes;

    // This UMAP analysis plugin is compatible with points datasets
    supportedTypes.append(PointType);

    return supportedTypes;
}

mv::gui::PluginTriggerActions UMAPAnalysisPluginFactory::getPluginTriggerActions(const mv::Datasets& datasets) const
{
    PluginTriggerActions pluginTriggerActions;

    const auto getPluginInstance = [this](const Dataset<Points>& dataset) -> UMAPAnalysisPlugin* {
        return dynamic_cast<UMAPAnalysisPlugin*>(plugins().requestPlugin(getKind(), { dataset }));
    };

    const auto numberOfDatasets = datasets.count();

    if (numberOfDatasets >= 1 && PluginFactory::areAllDatasetsOfTheSameType(datasets, PointType)) {
        auto pluginTriggerAction = new PluginTriggerAction(const_cast<UMAPAnalysisPluginFactory*>(this), this, "UMAP Analysis", "Perform an UMAP Analysis", getIcon(), [this, getPluginInstance, datasets](PluginTriggerAction& pluginTriggerAction) -> void {
            for (const auto& dataset : datasets)
                getPluginInstance(dataset);
            });

        pluginTriggerActions << pluginTriggerAction;
    }

    return pluginTriggerActions;
}
