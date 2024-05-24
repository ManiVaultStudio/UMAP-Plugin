#include "UMAPAnalysisPlugin.h"

#include <util/Icon.h>

#include <PointData/DimensionsPickerAction.h>
#include <PointData/InfoAction.h>

#include "knncolle/knncolle.hpp"

#include <QDebug>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrent>
#include <QtCore>

#include <thread>

Q_PLUGIN_METADATA(IID "studio.manivault.UMAPAnalysisPlugin")

using namespace mv;
using namespace mv::plugin;

using knnAnnoy = knncolle::Annoy<Annoy::Euclidean, std::int32_t, UMAPAnalysisPlugin::scalar_t, UMAPAnalysisPlugin::scalar_t, std::int32_t, UMAPAnalysisPlugin::scalar_t>;

UMAPAnalysisPlugin::UMAPAnalysisPlugin(const PluginFactory* factory) :
    AnalysisPlugin(factory),
    _settingsAction(),
    _outDimensions(2),
    _outputPoints(nullptr),
    _umap(),
    _umapStatus(nullptr)
{
}

UMAPAnalysisPlugin::~UMAPAnalysisPlugin()
{
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

    _embedding.resize(inputPoints->getNumPoints() * _outDimensions);
    updatePoints();

    // Set the dimension names as visible in the GUI
    _outputPoints->setDimensionNames({ "UMAP x", "UMAP y" });
    events().notifyDatasetDataDimensionsChanged(_outputPoints);

    // Inject the settings action in the output points dataset 
    // By doing so, the settings user interface will be accessible though the data properties widget
    _outputPoints->addAction(_settingsAction);
    
    // Automatically focus on the UMAP data set
    _outputPoints->getDataHierarchyItem().select();
    _outputPoints->_infoAction->collapse();

    // Update current iteration action
    auto updateCurrentIterationAction = [this](int currentIteration) {
        _settingsAction.getCurrentIterationAction().setString(QString::number(currentIteration));
    };

    auto computeUMAP = [this, updatePoints, updateCurrentIterationAction, inputPoints]() {
        auto& datasetTask = getOutputDataset()->getTask();
        datasetTask.setName("UMAP analysis");
        datasetTask.setRunning();
        datasetTask.setProgress(0.0f);
        datasetTask.setProgressDescription("Initializing...");

        // Get the number of iterations from the settings
        const auto numberOfIterations = _settingsAction.getNumberOfIterationsAction().getValue();

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

        unsigned int nThreads = std::thread::hardware_concurrency();
        if (nThreads == 0)
            nThreads = 1;
        bool useThredas = nThreads > 1;

        _umap = UMAP();
        _umap.set_num_neighbors(20);
        _umap.set_num_epochs(numberOfIterations);
        _umap.set_parallel_optimization(useThredas);
        _umap.set_num_threads(nThreads);

        std::vector<scalar_t> embedding(_embedding.size());
        int nDim = numEnabledDimensions;

        knnAnnoy searcher(nDim, numPoints, data.data(), /* ntrees = */ 20);
        _umapStatus = std::make_unique<UMAP::Status>(_umap.initialize(&searcher, _outDimensions, embedding.data()));

        datasetTask.setProgressDescription("Computing...");

        auto updateEmbeddingAndUI = [this, updatePoints, updateCurrentIterationAction](int iter, const std::vector<scalar_t>& emb) {
            // Copy from umap worker and publish to core
            for (size_t i = 0; i < emb.size(); i++)
                _embedding[i] = emb[i];

            updatePoints();
            updateCurrentIterationAction(iter);
        };

        // Iteratively update UMAP embedding
        for (int iter = 1; iter < numberOfIterations; iter++)
        {
            datasetTask.setProgress(iter / static_cast<float>(numberOfIterations));
            datasetTask.setProgressDescription(QString("Computing iteration %1/%2").arg(QString::number(iter), QString::number(numberOfIterations)));

            _umapStatus->run(iter);

            if (iter % 10 == 0)
                updateEmbeddingAndUI(iter, embedding);

            QCoreApplication::processEvents();
        }

        updateEmbeddingAndUI(numberOfIterations, embedding);

        qDebug() << "Total iterations: " << _umapStatus->epoch() << " " << _umapStatus->num_epochs();

        // Flag the analysis task as finished
        datasetTask.setFinished();
        };
    
    auto continueUMAP = [this, updatePoints, updateCurrentIterationAction]() {
        auto& datasetTask = getOutputDataset()->getTask();
        datasetTask.setName("UMAP analysis");
        datasetTask.setRunning();
        datasetTask.setProgress(0.0f);
        datasetTask.setProgressDescription("Computing...");

        const auto numberOfIterations = _settingsAction.getNumberOfIterationsAction().getValue();
        const auto currentIterations = _settingsAction.getCurrentIterationAction().getString().toInt();

        auto updateEmbeddingAndUI = [this, updatePoints, updateCurrentIterationAction](int iter) {
            const scalar_t* embedding = _umapStatus->embedding();

            // Copy from umap worker and publish to core
            for (size_t i = 0; i < _embedding.size(); i++)
                _embedding[i] = *(embedding + i);

            updatePoints();
            updateCurrentIterationAction(iter);
            };

        datasetTask.setProgressDescription("Computing...");

        // Iteratively update UMAP embedding
        for (int iter = currentIterations; iter < numberOfIterations; iter++)
        {
            datasetTask.setProgress(iter / static_cast<float>(numberOfIterations));
            datasetTask.setProgressDescription(QString("Computing iteration %1/%2").arg(QString::number(iter), QString::number(numberOfIterations)));

            _umapStatus->run(iter);

            if (iter % 10 == 0)
                updateEmbeddingAndUI(iter);

            QCoreApplication::processEvents();
        }

        updateEmbeddingAndUI(numberOfIterations);

        qDebug() << "Total iterations (cont.): " << _umapStatus->epoch() << " " << _umapStatus->num_epochs();

        // Flag the analysis task as finished
        datasetTask.setFinished();
        };


    // Start the analysis when the user clicks the start analysis push button
    connect(&_settingsAction.getStartAnalysisAction(), &mv::gui::TriggerAction::triggered, this, [this, computeUMAP] {

        // Disable actions during analysis
        _settingsAction.getNumberOfIterationsAction().setEnabled(false);

        // Run UMAP in another thread
        QFuture<void> future = QtConcurrent::run(computeUMAP);
        QFutureWatcher<void>* watcher = new QFutureWatcher<void>();

        // Enabled actions again once computation is done
        connect(watcher, &QFutureWatcher<int>::finished, [this, watcher]() {
           _settingsAction.getNumberOfIterationsAction().setEnabled(true);
            watcher->deleteLater();
            });

        watcher->setFuture(future);
        });

    connect(&_settingsAction.getUpdateAction(), &mv::gui::TriggerAction::triggered, this, [this, continueUMAP] {

        // Disable actions during analysis
        _settingsAction.getNumberOfIterationsAction().setEnabled(false);

        // Run UMAP in another thread
        QFuture<void> future = QtConcurrent::run(continueUMAP);
        QFutureWatcher<void>* watcher = new QFutureWatcher<void>();

        // Enabled actions again once computation is done
        connect(watcher, &QFutureWatcher<int>::finished, [this, watcher]() {
           _settingsAction.getNumberOfIterationsAction().setEnabled(true);
            watcher->deleteLater();
            });

        watcher->setFuture(future);
        });

    // Initialize current iteration action
    updateCurrentIterationAction(0);
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
