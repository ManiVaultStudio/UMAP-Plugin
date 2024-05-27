#include "UMAPAnalysisPlugin.h"

#include <util/Icon.h>

#include <PointData/DimensionsPickerAction.h>
#include <PointData/InfoAction.h>

#pragma warning(disable:4477)       // annoy internal: print formatting warnings
#include <annoy/annoylib.h>
#include <guiddef.h>
#include <hnswlib/space_ip.h>
#include <knncolle/Annoy/Annoy.hpp>
#include <knncolle/utils/Base.hpp>
#include <knncolle/Hnsw/Hnsw.hpp>
#pragma warning(default:4477)

#include <QDebug>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrent>
#include <QtCore>

#include <memory>

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

using knnAnnoyEuclidean = knncolle::Annoy<Annoy::Euclidean, int, UMAPAnalysisPlugin::scalar_t, UMAPAnalysisPlugin::scalar_t, int32_t, UMAPAnalysisPlugin::scalar_t>;
using knnAnnoyAngular   = knncolle::Annoy<Annoy::Angular, int, UMAPAnalysisPlugin::scalar_t, UMAPAnalysisPlugin::scalar_t, int32_t, UMAPAnalysisPlugin::scalar_t>;
using knnAnnoyDot       = knncolle::Annoy<Annoy::DotProduct, int, UMAPAnalysisPlugin::scalar_t, UMAPAnalysisPlugin::scalar_t, int32_t, UMAPAnalysisPlugin::scalar_t>;

using knnHnswEuclidean  = knncolle::Hnsw<knncolle::hnsw_distances::Euclidean, int, UMAPAnalysisPlugin::scalar_t, UMAPAnalysisPlugin::scalar_t>;
using knnHnswDot        = knncolle::Hnsw<knncolle::hnsw_distances::InnerProduct, int, UMAPAnalysisPlugin::scalar_t, UMAPAnalysisPlugin::scalar_t>;

UMAPAnalysisPlugin::UMAPAnalysisPlugin(const PluginFactory* factory) :
    AnalysisPlugin(factory),
    _settingsAction(this),
    _knnSettingsAction(this),
    _outDimensions(2),
    _outputPoints(nullptr),
    _umap(),
    _shouldStop(false)
{
}

static void normalizData(std::vector<UMAPAnalysisPlugin::scalar_t>& data) {
    float norm = 0.0f;
    for (const auto& val: data)
        norm += val * val;

    norm = 1.0f / (sqrtf(norm) + 1e-30f);

#pragma omp parallel
    for (std::int64_t i = 0; i < data.size(); i++)
        data[i] *= norm;
}

void UMAPAnalysisPlugin::init()
{
    // Create UMAP output dataset (a points dataset which is derived from the input points dataset) and set the output dataset
    setOutputDataset(mv::data().createDerivedDataset("UMAP Embedding", getInputDataset(), getInputDataset()));

    const auto inputPoints  = getInputDataset<Points>();
    _outputPoints = getOutputDataset<Points>();

    // Inform the core (and thus others) that the data changed
    const auto updatePoints = [this]() {
        _outputPoints->setData(_embedding.data(), _embedding.size() / _outDimensions, _outDimensions);
        events().notifyDatasetDataChanged(_outputPoints);
    };

    _embedding.resize(inputPoints->getNumPoints() * static_cast<size_t>(_outDimensions));
    updatePoints();

    // Set the dimension names as visible in the GUI
    _outputPoints->setDimensionNames({ "UMAP x", "UMAP y" });
    events().notifyDatasetDataDimensionsChanged(_outputPoints);

    // Add settings to UI
    _outputPoints->addAction(_settingsAction);
    _outputPoints->addAction(_knnSettingsAction);
    
    // Automatically focus on the UMAP data set
    _outputPoints->getDataHierarchyItem().select();
    _outputPoints->_infoAction->collapse();

    // Update current epoch action
    auto updateCurrentEpochAction = [this](int currentEpoch) {
        _settingsAction.getCurrentEpochAction().setString(QString::number(currentEpoch));
    };

    // Compute suggested number of epoch
    _settingsAction.getNumberOfEpochsAction().setValue(umappp::choose_num_epochs(-1, inputPoints->getNumPoints()));

    auto computeUMAP = [this, updatePoints, updateCurrentEpochAction, inputPoints]() {
        auto& datasetTask = getOutputDataset()->getTask();
        datasetTask.setName("UMAP analysis");
        datasetTask.setRunning();
        datasetTask.setProgress(0.0f);
        datasetTask.setProgressDescription("Initializing...");

        // Get the number of epochs from the settings
        const auto numberOfEpochs = _settingsAction.getNumberOfEpochsAction().getValue();

        // Create list of data from the enabled dimensions
        std::vector<scalar_t> data;
        std::vector<unsigned int> indices;

        // Extract the enabled dimensions from the data
        std::vector<bool> enabledDimensions = getInputDataset<Points>()->getDimensionsPickerAction().getEnabledDimensions();

        const auto numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

        size_t numPoints = inputPoints->isFull() ? inputPoints->getNumPoints() : inputPoints->indices.size();
        data.resize(numPoints * numEnabledDimensions);

        for (int i = 0; i < inputPoints->getNumDimensions(); i++)
            if (enabledDimensions[i])
                indices.push_back(i);

        inputPoints->populateDataForDimensions<std::vector<scalar_t>, std::vector<unsigned int>>(data, indices);

        const KnnParameters knnParams = _knnSettingsAction.getKnnParameters();

        const int numNeighbors = knnParams.getK();

        _umap = UMAP();
        _umap.set_num_neighbors(numNeighbors);
        _umap.set_num_epochs(numberOfEpochs);

#ifdef USE_OPENMP
        if (_settingsAction.getMultithreadAction().isChecked())
        {
            unsigned int nThreads = omp_get_max_threads();
            if (nThreads <= 0)
                nThreads = 1;
            bool useThreads = nThreads > 1;

            _umap.set_parallel_optimization(useThreads);
            _umap.set_num_threads(nThreads);

            qDebug() << "UMAP: parallelize with " << nThreads << " threads";
        }
#endif

        // default is spectral
        if(_settingsAction.getInitializeAction().getCurrentText() == "Random")
            _umap.set_initialize(umappp::InitMethod::RANDOM);

        int nDim = numEnabledDimensions;

        qDebug() << "UMAP: compute knn: " << numNeighbors << " neighbors";

        std::unique_ptr<knncolle::Base<int, UMAPAnalysisPlugin::scalar_t, UMAPAnalysisPlugin::scalar_t>> searcher;

        if (knnParams.getKnnAlgorithm() == KnnLibrary::ANNOY) {

            if(knnParams.getKnnDistanceMetric() == KnnMetric::COSINE)
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

        auto status = std::make_unique<UMAP::Status>(_umap.initialize(searcher.get(), _outDimensions, _embedding.data()));

        auto updateEmbeddingAndUI = [this, updatePoints, updateCurrentEpochAction](int epoch) {
            updatePoints();
            updateCurrentEpochAction(epoch);
            };

        updatePoints();

        if (numberOfEpochs == 0 || _shouldStop)
        {
            _shouldStop = false;
            return;
        }

        datasetTask.setProgressDescription("Computing...");

        qDebug() << "UMAP: start gradient descent: " << numberOfEpochs << " epoch";

        int iter = 1;
        // Iteratively update UMAP embedding
        for (; iter < numberOfEpochs; iter++)
        {
            if (_shouldStop)
                break;

            datasetTask.setProgress(iter / static_cast<float>(numberOfEpochs));
            datasetTask.setProgressDescription(QString("Computing epoch %1/%2").arg(QString::number(iter), QString::number(numberOfEpochs)));

            status->run(iter);

            if (iter % 10 == 0)
                updateEmbeddingAndUI(iter);

            QCoreApplication::processEvents();
        }

        updateEmbeddingAndUI(iter);

        qDebug() << "UMAP: total epochs: " << status->epoch() + 1 ;

        // Flag the analysis task as finished
        datasetTask.setFinished();
        _shouldStop = false;
        };
    
    // Start the analysis when the user clicks the start analysis push button
    connect(&_settingsAction.getStartAction(), &mv::gui::TriggerAction::triggered, this, [this, computeUMAP] {

        // Disable actions during analysis
        _settingsAction.getNumberOfEpochsAction().setEnabled(false);
        _settingsAction.getStartStopAction().setStarted();

        // Run UMAP in another thread
        QFuture<void> future = QtConcurrent::run(computeUMAP);
        QFutureWatcher<void>* watcher = new QFutureWatcher<void>();

        // Enabled actions again once computation is done
        connect(watcher, &QFutureWatcher<int>::finished, [this, watcher]() {
           _settingsAction.getNumberOfEpochsAction().setEnabled(true);
           _settingsAction.getStartStopAction().setFinished();
           watcher->deleteLater();
            });

        watcher->setFuture(future);
        });

    connect(&_settingsAction.getStopAction(), &mv::gui::TriggerAction::triggered, this, [this] {
        _shouldStop = true;
        });

    // Initialize current epoch action
    updateCurrentEpochAction(0);
}


/// ////////////// ///
/// Plugin Factory ///
/// ////////////// ///

QIcon UMAPAnalysisPluginFactory::getIcon(const QColor& color /*= Qt::black*/) const
{
    return mv::gui::createPluginIcon("UMAP", color);
}

AnalysisPlugin* UMAPAnalysisPluginFactory::produce()
{``
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
