#include "UMAPAnalysisPlugin.h"

#include <util/Icon.h>

#include <PointData/DimensionsPickerAction.h>
#include <PointData/InfoAction.h>

#include <knncolle/knncolle.hpp>
#include <knncolle/find_nearest_neighbors.hpp>
#include <knncolle_annoy/knncolle_annoy.hpp>
#include <knncolle_hnsw/knncolle_hnsw.hpp>
#include <hnswlib/space_ip.h>
#include <hnswlib/hnswlib.h>

// MSVC does not support all openmp functionality
// that the umappp tries to use
#ifdef _OPENMP
#define _OPENMP_CACHED _OPENMP
#undef _OPENMP
#endif
#include <umappp/initialize.hpp>
#include <umappp/Options.hpp>
#ifdef _OPENMP_CACHED
#define _OPENMP _OPENMP_CACHED
#undef _OPENMP_CACHED
#endif

#include <QDebug>
#include <QtCore>

#include <cmath>

#ifdef _OPENMP
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

using knnMatrix         = knncolle::SimpleMatrix<integer_t, integer_t, scalar_t>;
using knnBase           = knncolle::Prebuilt<integer_t, integer_t, scalar_t>;

using knnAnnoyEuclidean = knncolle_annoy::AnnoyBuilder<Annoy::Euclidean, knnMatrix, scalar_t, integer_t, scalar_t>;
using knnAnnoyAngular   = knncolle_annoy::AnnoyBuilder<Annoy::Angular, knnMatrix, scalar_t, integer_t, scalar_t>;
using knnAnnoyDot       = knncolle_annoy::AnnoyBuilder<Annoy::DotProduct, knnMatrix, scalar_t, integer_t, scalar_t>;

using knnHnsw           = knncolle_hnsw::HnswBuilder<knnMatrix, scalar_t, scalar_t>;

static void normalizeData(std::vector<scalar_t>& data) {
    float norm = 0.0f;
    for (const auto& val : data)
        norm += val * val;

    norm = 1.0f / (std::sqrtf(norm) + 1e-30f);

#pragma omp parallel
    for (std::int64_t i = 0; i < data.size(); i++)
        data[i] *= norm;
}

UMAPAnalysisPlugin::UMAPAnalysisPlugin(const PluginFactory* factory) :
    AnalysisPlugin(factory),
    _settingsAction(this),
    _knnSettingsAction(this),
    _advSettingsAction(this),
    _outDimensions(2),
    _numPoints(0),
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
    auto initEmbeddingsAndDimensions = [this](uint32_t numPoints) {
        std::vector<scalar_t> initEmbeddingValues;
        initEmbeddingValues.resize(numPoints * static_cast<size_t>(_outDimensions));
        _outputPoints->setData(initEmbeddingValues.data(), initEmbeddingValues.size() / _outDimensions, _outDimensions);
        events().notifyDatasetDataChanged(_outputPoints);

        // Set the dimension names as visible in the GUI
        if (_outDimensions == 2)
            _outputPoints->setDimensionNames({ "UMAP x", "UMAP y" });
        else if (_outDimensions == 3)
            _outputPoints->setDimensionNames({ "UMAP x", "UMAP y", "UMAP z" });
        else
        {
            std::vector<QString> dimNames;
            for (int i = 1; i <= _outDimensions; ++i)
                dimNames.push_back(QString("UMAP %1").arg(i));

            _outputPoints->setDimensionNames(dimNames);
        }

        events().notifyDatasetDataDimensionsChanged(_outputPoints);
        };

    // Create UMAP output dataset (a points dataset which is derived from the input points dataset) and set the output dataset
    // we do not need to create a new output when loading this plugin from a project
    if (!outputDataInit())
    {
        _outputPoints = Dataset<Points>(mv::data().createDerivedDataset("UMAP Embedding", getInputDataset(), getInputDataset()));
        setOutputDataset(_outputPoints);

        initEmbeddingsAndDimensions(getInputDataset<Points>()->getNumPoints());
    }
    
    _outputPoints   = getOutputDataset<Points>();
    _numPoints      = getInputDataset<Points>()->getNumPoints();

    // Add settings to UI
    _outputPoints->addAction(_settingsAction);
    _outputPoints->addAction(_knnSettingsAction);
    _outputPoints->addAction(_advSettingsAction);
    
    // Automatically focus on the UMAP data set
    _outputPoints->getDataHierarchyItem().select();
    _outputPoints->_infoAction->collapse();

    // Initialize current epoch action
    _settingsAction.getCurrentEpochAction().setString(QString::number(0));

    // Compute suggested number of epoch
    _settingsAction.getNumberOfEpochsAction().setValue(umappp::internal::choose_num_epochs(-1, _numPoints));

    // Create UMAP worker, which will be executed in another thread
    // Start the analysis when the user clicks the start analysis push button
    connect(&_settingsAction.getStartAction(), &mv::gui::TriggerAction::triggered, this, [this, initEmbeddingsAndDimensions] {

        // Disable actions during analysis
        _settingsAction.setStarted();
        _knnSettingsAction.setReadOnly(true);

        deleteWorker();

        getOutputDataset()->getTask().setRunning();

        _outDimensions = _settingsAction.getNumberEmbDimsAction().getValue();

        // if the user sets a different embedding dimension than 2, re-size the output data
        if(getOutputDataset<Points>()->getNumDimensions() != _outDimensions)
            initEmbeddingsAndDimensions(_numPoints);

        Dataset<Points> inputPoints = getInputDataset<Points>();
        _umapWorker = new UMAPWorker(inputPoints, &getOutputDataset()->getTask(), _outDimensions, &_settingsAction, &_knnSettingsAction, &_advSettingsAction);

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

void UMAPAnalysisPlugin::fromVariantMap(const QVariantMap& variantMap)
{
    AnalysisPlugin::fromVariantMap(variantMap);

    mv::util::variantMapMustContain(variantMap, "UMAP Settings");
    mv::util::variantMapMustContain(variantMap, "Knn Settings");
    mv::util::variantMapMustContain(variantMap, "Advanced Settings");

    _settingsAction.fromParentVariantMap(variantMap);
    _knnSettingsAction.fromParentVariantMap(variantMap);
    _advSettingsAction.fromParentVariantMap(variantMap);
}

QVariantMap UMAPAnalysisPlugin::toVariantMap() const
{
    QVariantMap variantMap = AnalysisPlugin::toVariantMap();

    _settingsAction.insertIntoVariantMap(variantMap);
    _knnSettingsAction.insertIntoVariantMap(variantMap);
    _advSettingsAction.insertIntoVariantMap(variantMap);

    return variantMap;
}


/// ////////// ///
/// UMAPWorker ///
/// ////////// ///

UMAPWorker::UMAPWorker(Dataset<Points>& inputPoints, DatasetTask* parentTask, int outDim, SettingsAction* settings, KnnSettingsAction* knnSettings, AdvancedSettingsAction* advSettings):
    _shouldStop(false),
    _inputDataset(inputPoints),
    _computeTask(nullptr),
    _parentTask(parentTask),
    _settingsAction(settings),
    _knnSettingsAction(knnSettings),
    _advSettingsAction(advSettings),
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

#ifdef _OPENMP
    if (_knnSettingsAction->getMultithreadAction().isChecked())
    {
        nThreads = omp_get_max_threads();
        if (nThreads <= 0)
            nThreads = 1;
    }
#endif

    // compute knn
    knncolle::NeighborList<integer_t, scalar_t> nearestNeighbors(numPoints);
    const KnnParameters knnParams = _knnSettingsAction->getKnnParameters();
    const int numNeighbors = knnParams.getK();
    {
        int nDim = numEnabledDimensions;

        qDebug() << "UMAP: compute knn: " << numNeighbors << " neighbors";

        std::unique_ptr<knnBase> searcher;
        auto mat = knnMatrix(nDim, numPoints, data.data());

        if (knnParams.getKnnAlgorithm() == KnnLibrary::ANNOY) {
            knncolle_annoy::AnnoyOptions opt;
            opt.num_trees   = knnParams.getAnnoyNumTrees();
            opt.search_mult = knnParams.getAnnoyNumChecks();

            if (knnParams.getKnnDistanceMetric() == KnnMetric::COSINE)
                searcher = knnAnnoyAngular(opt).build_unique(mat);
            else if (knnParams.getKnnDistanceMetric() == KnnMetric::DOT)
                searcher = knnAnnoyDot(opt).build_unique(mat);
            else
                searcher = knnAnnoyEuclidean(opt).build_unique(mat);
        }
        else // knnParams.getKnnAlgorithm() == KnnLibrary::HNSW
        {
            knncolle_hnsw::HnswOptions<integer_t, scalar_t> opt;
            opt.num_links       = knnParams.getHNSWm();
            opt.ef_search       = knnParams.getHNSWef();
            opt.ef_construction = knnParams.getHNSWef();

            if (knnParams.getKnnDistanceMetric() == KnnMetric::COSINE)
            {
                normalizeData(data);
                searcher = knnHnsw(opt).build_unique(mat);
            }
            else if (knnParams.getKnnDistanceMetric() == KnnMetric::DOT)
            {
                opt.distance_options.create = [](int dim) -> hnswlib::SpaceInterface<float>*{
                    return new hnswlib::InnerProductSpace(dim);
                };
                searcher = knnHnsw(opt).build_unique(mat);
            }
            else // Euclidean distance
                searcher = knnHnsw(opt).build_unique(mat);
        }

        nearestNeighbors = knncolle::find_nearest_neighbors<integer_t, integer_t, scalar_t>(*searcher, numNeighbors, nThreads);

    }

    qDebug() << "UMAP: initializing...";

    const auto advancedSettings = _advSettingsAction->getAdvParameters();

    umappp::Options opt;

    // general settings
    opt.num_neighbors = numNeighbors;
    opt.num_epochs = numberOfEpochs;

    // default is spectral
    if (_settingsAction->getInitializeAction().getCurrentText() == "Random")
        opt.initialize = umappp::InitializeMethod::RANDOM;
    else
        opt.initialize = umappp::InitializeMethod::SPECTRAL;

     // advanced settings
    opt.local_connectivity  = advancedSettings.local_connectivity;
    opt.bandwidth           = advancedSettings.bandwidth;
    opt.mix_ratio           = advancedSettings.mix_ratio;
    opt.spread              = advancedSettings.spread;
    opt.min_dist            = advancedSettings.min_dist;
    opt.a                   = advancedSettings.a;
    opt.b                   = advancedSettings.b;
    opt.repulsion_strength  = advancedSettings.repulsion_strength;
    opt.learning_rate       = advancedSettings.learning_rate;
    opt.negative_sample_rate = advancedSettings.negative_sample_rate;
    opt.seed                = advancedSettings.seed;

    auto status = umappp::initialize<integer_t, scalar_t>(nearestNeighbors, _outDimensions, _embedding.data(), opt);

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

        status.run(iter);

        if (iter % 10 == 0)
            updateEmbedding(iter);

        _computeTask->setProgress(iter / static_cast<float>(numberOfEpochs));
        _computeTask->setProgressDescription(QString("Epoch %1/%2").arg(QString::number(iter), QString::number(numberOfEpochs)));
        QCoreApplication::processEvents();
    }

    updateEmbedding(iter);

    qDebug() << "UMAP: total epochs: " << status.epoch() + 1;

    // Flag the analysis task as finished
    _computeTask->setFinished();

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

QUrl UMAPAnalysisPluginFactory::getReadmeMarkdownUrl() const
{
    return QUrl("https://raw.githubusercontent.com/ManiVaultStudio/UMAP-Plugin/README.md");
}

QUrl UMAPAnalysisPluginFactory::getRespositoryUrl() const
{
    return QUrl("https://github.com/ManiVaultStudio/UMAP-Plugin");
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
