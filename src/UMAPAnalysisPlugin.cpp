#include "UMAPAnalysisPlugin.h"

#include <widgets/MarkdownDialog.h>
#include <util/Icon.h>

#include <PointData/DimensionsPickerAction.h>
#include <PointData/InfoAction.h>

#include <hnswlib/hnswlib.h>
#include <hnswlib/space_ip.h>

#include <knncolle/Builder.hpp>
#include <knncolle/Prebuilt.hpp>
#include <knncolle/find_nearest_neighbors.hpp>
#include <knncolle_annoy/knncolle_annoy.hpp>
#include <knncolle_hnsw/distances.hpp>

#include "util/hnsw_space_corr.h"
#include "util/knncolle_matrix_parallel.h"
#include "util/knncolle_hnsw_parallel.h"

#pragma warning(disable:4267) // umapp internal: conversion warning
#include <umappp/initialize.hpp>
#include <umappp/find_ab.hpp>
#include <umappp/Options.hpp>
#pragma warning(default:4267)

#include <QDebug>
#include <QtCore>

#include <algorithm>
#include <cmath>
#include <tuple>

#ifdef _OPENMP
#include <omp.h>
#endif

Q_PLUGIN_METADATA(IID "studio.manivault.UMAPAnalysisPlugin")

using namespace mv;
using namespace mv::plugin;

using DataMatrix        = knncolle::ParallelMatrix< /* observation index */ integer_t, /* data type */ scalar_t>;
using KnnBase           = knncolle::Prebuilt< /* observation index */ integer_t, /* data type */ scalar_t, /* distance type */ scalar_t>;

using KnnAnnoyEuclidean = knncolle_annoy::AnnoyBuilder<integer_t, scalar_t, scalar_t, Annoy::Euclidean>;
using KnnAnnoyAngular   = knncolle_annoy::AnnoyBuilder<integer_t, scalar_t, scalar_t, Annoy::Angular>;
using KnnAnnoyDot       = knncolle_annoy::AnnoyBuilder<integer_t, scalar_t, scalar_t, Annoy::DotProduct>;

using KnnHnsw           = knncolle_hnsw::HnswBuilderParallel<integer_t, scalar_t, scalar_t, DataMatrix>;

static void normalizeData(std::vector<scalar_t>& data) {
    float norm = 0.0f;
    for (const auto& val : data)
        norm += val * val;

    norm = 1.0f / (std::sqrt(norm) + 1e-30f);

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
        std::vector<QString> dimNames;
        if (_outDimensions == 2)
            dimNames = { "UMAP x", "UMAP y" };
        else if (_outDimensions == 3)
            dimNames = { "UMAP x", "UMAP y", "UMAP z" };
        else {
            for (int i = 1; i <= _outDimensions; ++i) {
                dimNames.push_back(QString("UMAP %1").arg(i));
            }
        }

        _outputPoints->setDimensionNames(dimNames); // calls notifyDatasetDataDimensionsChanged
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
    _parentTask(parentTask),
    _settingsAction(settings),
    _knnSettingsAction(knnSettings),
    _advSettingsAction(advSettings),
    _embedding(),
    _outDimensions(outDim)
{
    _embedding.resize(inputPoints->getNumPoints() * static_cast<size_t>(_outDimensions));

    connect(_parentTask, &DatasetTask::requestAbort, this, &UMAPWorker::stop, Qt::DirectConnection);
}

void UMAPWorker::changeThread(QThread* targetThread)
{
    this->moveToThread(targetThread);
}

void UMAPWorker::resetThread()
{
    changeThread(QCoreApplication::instance()->thread());
}

void UMAPWorker::stop()
{
    qDebug() << "UMAP: user requested manual stop";
    _shouldStop = true;
}

// modular helper to extract data from core
static std::tuple< std::vector<scalar_t>, std::vector<unsigned int>, size_t, size_t> extractEnabledDimensions(Dataset<Points>& dataset) {
    // Create list of data from the enabled dimensions
    std::vector<scalar_t> data;
    std::vector<unsigned int> indices;

    // Extract the enabled dimensions from the data
    const std::vector<bool> enabledDimensions = dataset->getDimensionsPickerAction().getEnabledDimensions();
    const auto numEnabledDimensions = static_cast<size_t>(count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; }));

    const size_t numPoints = dataset->isFull() ? dataset->getNumPoints() : dataset->indices.size();
    data.resize(numPoints * numEnabledDimensions);

    for (int i = 0; i < dataset->getNumDimensions(); i++) {
        if (enabledDimensions[i]) {
            indices.push_back(i);
        }
    }

    dataset->populateDataForDimensions<std::vector<scalar_t>, std::vector<unsigned int>>(data, indices);

    return { data, indices, numEnabledDimensions, numPoints };
}

void UMAPWorker::compute()
{
    auto cleanup = [this]() -> void {
        _parentTask->setFinished();
        emit finished();
        resetThread();
        };

    _shouldStop = false;

    _parentTask->setRunning();
    _parentTask->setProgressDescription("Computing knn");

    QCoreApplication::processEvents();

    // get data
    auto [data, indices, numEnabledDimensions, numPoints] = extractEnabledDimensions(_inputDataset);

    // determine threading
    const bool parallel_knn = _knnSettingsAction->getMultithreadAction().isChecked();
    const bool parallel_layout = _advSettingsAction->getMultithreadAction().isChecked();
    unsigned int num_threads_knn = 1;
    unsigned int num_threads_layout = 1;

#ifdef _OPENMP
    {
        unsigned int num_threads_available = omp_get_max_threads();
        if (num_threads_available <= 0) {
            num_threads_available = 1;
        }

        if (parallel_knn) {
            num_threads_knn = num_threads_available;
        }

        if (num_threads_layout) {
            num_threads_layout = num_threads_available;
        }
    }
#endif

    // compute knn
    knncolle::NeighborList<integer_t, scalar_t> nearestNeighbors;
    const KnnParameters knnParams = _knnSettingsAction->getKnnParameters();
    const int numNeighbors = knnParams.getK();
    {
        qDebug() << "UMAP: compute knn: " << numNeighbors << " neighbors based on " << printMetric(knnParams.getKnnMetric()) << " distance with " << printAlgorithm(knnParams.getKnnAlgorithm());
        qDebug() << "UMAP: adding vectors to knn searcher";

        std::unique_ptr<KnnBase> searcher;
        const auto mat = DataMatrix(static_cast<size_t>(numEnabledDimensions), numPoints, data.data());

        if (knnParams.getKnnAlgorithm() == KnnAlgorithm::ANNOY) {
            knncolle_annoy::AnnoyOptions opt;
            opt.num_trees   = knnParams.getAnnoyNumTrees();
            opt.search_mult = knnParams.getAnnoyNumChecks();

            switch (knnParams.getKnnMetric()) {
            case KnnMetric::COSINE:     searcher = KnnAnnoyAngular(opt).build_unique(mat); break;
            case KnnMetric::DOT:        searcher = KnnAnnoyDot(opt).build_unique(mat); break;
            case KnnMetric::EUCLIDEAN:  searcher = KnnAnnoyEuclidean(opt).build_unique(mat); break;
            default:
                qDebug() << "UMAP: unknown metric using euclidean";
                searcher = KnnAnnoyEuclidean(opt).build_unique(mat); break;
            }
        }
        else // knnParams.getKnnAlgorithm() == KnnAlgorithm::HNSW
        {
            knncolle_hnsw::HnswOptions opt;
            opt.num_links       = knnParams.getHNSWm();
            opt.ef_search       = knnParams.getHNSWef();
            opt.ef_construction = knnParams.getHNSWef();

            switch (knnParams.getKnnMetric()) {
            case KnnMetric::COSINE:
                normalizeData(data);
                searcher = KnnHnsw(knncolle_hnsw::makeEuclideanDistanceConfig<scalar_t>(), opt).build_unique(mat);
                break;
            case KnnMetric::DOT: {
                auto inner_config = knncolle_hnsw::DistanceConfig<scalar_t>();
                inner_config.create = [](std::size_t dim) -> hnswlib::SpaceInterface<scalar_t>*{
                    return static_cast<hnswlib::InnerProductSpace*>(new hnswlib::InnerProductSpace(dim));
                    };

                searcher = KnnHnsw(inner_config, opt).build_unique(mat);
                break;
            }
            case KnnMetric::EUCLIDEAN:  
                searcher = KnnHnsw(knncolle_hnsw::makeEuclideanDistanceConfig<scalar_t>(), opt).build_unique(mat);
                break;
            case KnnMetric::CORRELATION: {
                auto correlation_config = knncolle_hnsw::DistanceConfig<scalar_t>();
                correlation_config.create = [](std::size_t dim) -> hnswlib::SpaceInterface<scalar_t>*{
                    return static_cast<hnswlib::CorrelationSpace*>(new hnswlib::CorrelationSpace(dim));
                    };

                searcher = KnnHnsw(correlation_config, opt).build_unique(mat);
                break;
            }
            default:
                qDebug() << "UMAP: unknown metric using euclidean";
                searcher = KnnHnsw(knncolle_hnsw::makeEuclideanDistanceConfig<scalar_t>(), opt).build_unique(mat);
            }
        }

        qDebug() << "UMAP: querying knn in searcher: " << numNeighbors << " neighbors";
        nearestNeighbors = knncolle::find_nearest_neighbors<integer_t, scalar_t, scalar_t>(*searcher, numNeighbors, num_threads_knn);
        qDebug() << "UMAP: finished knn";

    }

    if (_shouldStop) {
        cleanup();
        return;
    }

    qDebug() << "UMAP: initializing layout...";
    _parentTask->setProgressDescription("Initializing layout");
    QCoreApplication::processEvents();

    const auto advancedSettings = _advSettingsAction->getAdvParameters();

    umappp::Options opt;

    // get the number of epochs from the settings
    const auto numberOfEpochs = _settingsAction->getNumberOfEpochsAction().getValue();

    // general settings
    opt.num_neighbors = numNeighbors;
    opt.num_epochs = numberOfEpochs;

    // default is spectral
    if (_settingsAction->getInitializeAction().getCurrentText() == "Random")
        opt.initialize_method = umappp::InitializeMethod::RANDOM;
    else
        opt.initialize_method = umappp::InitializeMethod::SPECTRAL;

    // option that are not exposed
    opt.initialize_random_on_spectral_fail = true;
    opt.initialize_spectral_irlba_options = {};
    opt.initialize_spectral_scale = 10;
    opt.initialize_spectral_jitter = false;
    opt.initialize_spectral_jitter_sd = 0.0001;
    opt.initialize_random_scale = 10;
    opt.optimize_seed = 1234567890;

     // advanced settings
    opt.local_connectivity   = advancedSettings.local_connectivity;
    opt.bandwidth            = advancedSettings.bandwidth;
    opt.mix_ratio            = advancedSettings.mix_ratio;
    opt.spread               = advancedSettings.spread;
    opt.min_dist             = advancedSettings.min_dist;
    opt.a                    = advancedSettings.a;
    opt.b                    = advancedSettings.b;
    opt.repulsion_strength   = advancedSettings.repulsion_strength;
    opt.learning_rate        = advancedSettings.learning_rate;
    opt.negative_sample_rate = advancedSettings.negative_sample_rate;
    opt.initialize_seed      = advancedSettings.seed;

    if (parallel_layout) {
        opt.parallel_optimization   = true;
        opt.num_threads             = num_threads_layout;
    }

    // move this here from umappp::initialize so that we can log the resulting a and b settings
    if (opt.a <= 0 || opt.b <= 0) {
        auto found = umappp::internal::find_ab(opt.spread, opt.min_dist);
        opt.a = found.first;
        opt.b = found.second;
    }

    qDebug() << "UMAP: layout settings: a: " << opt.a << ", b: " << opt.b << ", min_dist: " << opt.min_dist << ", spread: " << opt.spread;

    auto status = umappp::initialize<integer_t, scalar_t>(nearestNeighbors, _outDimensions, _embedding.data(), opt);

    const auto updateEmbedding = [this](int ep) -> void {
        _outEmbedding.assign(_embedding.begin(), _embedding.end());
        emit embeddingUpdate(_outEmbedding, ep);
        };

    updateEmbedding(0);

    if (numberOfEpochs == 0 || _shouldStop) {
        cleanup();
        return;
    }

    qDebug() << "UMAP: start gradient descent: " << numberOfEpochs << " epochs";

    int epoch = 1;
    // Iteratively update UMAP embedding
    for (; epoch < numberOfEpochs; epoch++)
    {
        if (_shouldStop)
            break;

        status.run(epoch);

        // publish the embedding only every 10 iterations
        if (epoch % 10 == 0)
            updateEmbedding(epoch);

        // update status progress each iteration
        _parentTask->setProgress(epoch / static_cast<float>(numberOfEpochs));
        _parentTask->setProgressDescription(QString("Epoch %1/%2").arg(QString::number(epoch), QString::number(numberOfEpochs)));
        _settingsAction->getCurrentEpochAction().setString(QString::number(epoch));
        QCoreApplication::processEvents();
    }

    updateEmbedding(epoch);

    qDebug() << "UMAP: total epochs: " << status.epoch() + 1;

    cleanup();
}

UMAPAnalysisPluginFactory::UMAPAnalysisPluginFactory()
{
    setIcon(StyledIcon(createPluginIcon("UMAP")));

    connect(&getPluginMetadata().getTriggerHelpAction(), &TriggerAction::triggered, this, [this]() -> void {
        if (!getReadmeMarkdownUrl().isValid() || _helpMarkdownDialog.get())
            return;

        _helpMarkdownDialog = new util::MarkdownDialog(getReadmeMarkdownUrl());

        _helpMarkdownDialog->setWindowTitle(QString("%1").arg(getKind()));
        _helpMarkdownDialog->setAttribute(Qt::WA_DeleteOnClose);
        _helpMarkdownDialog->setWindowModality(Qt::NonModal);
        _helpMarkdownDialog->show();
        });
}

QUrl UMAPAnalysisPluginFactory::getReadmeMarkdownUrl() const
{
    return QUrl("https://raw.githubusercontent.com/ManiVaultStudio/UMAP-Plugin/main/README.md");
}

QUrl UMAPAnalysisPluginFactory::getRepositoryUrl() const
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
        auto pluginTriggerAction = new PluginTriggerAction(const_cast<UMAPAnalysisPluginFactory*>(this), this, "UMAP Analysis", "Perform an UMAP Analysis", icon(), [this, getPluginInstance, datasets](PluginTriggerAction& pluginTriggerAction) -> void {
            for (const auto& dataset : datasets)
                getPluginInstance(dataset);
            });

        pluginTriggerActions << pluginTriggerAction;
    }

    return pluginTriggerActions;
}
