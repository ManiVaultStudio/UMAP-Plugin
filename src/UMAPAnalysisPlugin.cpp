#include "UMAPAnalysisPlugin.h"

#include <PointData/DimensionsPickerAction.h>

#include <QtCore>
#include <QDebug>

#include "umappp/Umap.hpp"

Q_PLUGIN_METADATA(IID "nl.ManiVault.UMAPAnalysisPlugin")

using namespace mv;
using namespace mv::plugin;

// Initialize the random number generator
QRandomGenerator UMAPAnalysisPlugin::Point::rng;

// Initialize the dimension names
std::vector<QString> UMAPAnalysisPlugin::Point::dimensionNames = std::vector<QString>({"UMAP x", "UMAP y" });

// Initialize the number of dimensions in the point
std::uint32_t UMAPAnalysisPlugin::Point::numberOfDimensions = 2;

UMAPAnalysisPlugin::UMAPAnalysisPlugin(const PluginFactory* factory) :
    AnalysisPlugin(factory),
    _settingsAction(),
    _pointHeadings(),
    _outDimensions(2),
    _outputPoints(nullptr)
{
}

void UMAPAnalysisPlugin::init()
{
    // Create UMAP output dataset (a points dataset which is derived from the input points dataset) and set the output dataset
    setOutputDataset(_core->createDerivedDataset("UMAP Embedding", getInputDataset(), getInputDataset()));

    // Retrieve the input dataset for our specific data type (in our case points)
    // The HDPS core sets the input dataset reference when the plugin is instantiated
    const auto inputPoints  = getInputDataset<Points>();

    // Retrieve the output dataset for our specific data type (in our case points)
    _outputPoints = getOutputDataset<Points>();

    // Set the dimension names as visible in the GUI
    _outputPoints->setDimensionNames(Point::dimensionNames);
    
    // Performs points update
    const auto updatePoints = [this]() {

        // Assign the output points to the output dataset
        int d = Point::numberOfDimensions;
        _outputPoints->setData(reinterpret_cast<float*>(_embedding.data()), _embedding.size() / d, d);

        // Inform the core (and thus others) that the data changed
        events().notifyDatasetDataChanged(_output);
    };

    // Initializes the points
    const auto initializePoints = [this, inputPoints, updatePoints]() {

        if (_embedding.size() != inputPoints->getNumPoints() * _outDimensions) {

            // Resize the point positions and headings
            _embedding.resize(inputPoints->getNumPoints() * _outDimensions);
        }
        else {

            std::fill(_embedding.begin(), _embedding.end(), 0.0);
        }

        // Update the points
        updatePoints();
    };

    // Inject the settings action in the output points dataset 
    // By doing so, the settings user interface will be accessible though the data properties widget
    _outputPoints->addAction(_settingsAction);
    
    // Update current iteration action
    const auto updateCurrentIterationAction = [this](const std::int32_t& currentIteration = 0) {
        _settingsAction.getCurrentIterationAction().setString(QString::number(currentIteration));
    };

    // Start the analysis when the user clicks the start analysis push button
    connect(&_settingsAction.getStartAnalysisAction(), &mv::gui::TriggerAction::triggered, this, [this, initializePoints, updatePoints, updateCurrentIterationAction, inputPoints]() {

        // Initialize our points
        initializePoints();

        // Disable actions during analysis
        _settingsAction.getNumberOfIterationsAction().setEnabled(false);

        // Get reference to dataset task for reporting progress
        auto& datasetTask = getOutputDataset()->getTask();

        // Set the task name as it will appear in the data hierarchy viewer
        datasetTask.setName("UMAP analysis");

        // In order to report progress the task status has to be set to running
        datasetTask.setRunning();

        // Zero progress at the start
        datasetTask.setProgress(0.0f);

        // Set task description as it will appear in the data hierarchy viewer
        datasetTask.setProgressDescription("Initializing");

        // Get the number of iterations from the settings
        const auto numberOfIterations = _settingsAction.getNumberOfIterationsAction().getValue();
        
        // Create list of data from the enabled dimensions
        std::vector<double> data;
        std::vector<unsigned int> indices;

        // Extract the enabled dimensions from the data
        std::vector<bool> enabledDimensions = getInputDataset<Points>()->getDimensionsPickerAction().getEnabledDimensions();

        const auto numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

        size_t numPoints = inputPoints->isFull() ? inputPoints->getNumPoints() : inputPoints->indices.size();
        data.resize(numPoints * numEnabledDimensions);

        for (int i = 0; i < inputPoints->getNumDimensions(); i++)
            if (enabledDimensions[i])
                indices.push_back(i);

        inputPoints->populateDataForDimensions<std::vector<double>, std::vector<unsigned int>>(data, indices);
        
        //std::vector<double> input(data.size());
        //for(int i = 0; i < data.size())
        
        umappp::Umap umap;
        umap.set_num_neighbors(20).set_num_epochs(numberOfIterations);
        std::vector<double> embedding(_embedding.size());
        int nDim = numEnabledDimensions;
        auto status = umap.initialize(nDim, numPoints, data.data(), _outDimensions, embedding.data());

        // Do computation
        datasetTask.setProgressDescription("Computing...");

        // Perform the loop
        for (int i = 0; i < numberOfIterations; i++)
        {
            QCoreApplication::processEvents();
            
            // Update task progress
            datasetTask.setProgress(i / static_cast<float>(numberOfIterations));

            // Update task progress
            datasetTask.setProgressDescription(QString("Computing iteration %1/%2").arg(QString::number(i), QString::number(numberOfIterations)));

            // Iterate the UMAP
            status.run(i);//run(i)
            
            // copy double to float
            for(size_t i = 0; i < embedding.size(); i++)
                _embedding[i] = static_cast<float>(embedding[i]);

            // Update the points
            updatePoints();

            // Update current iteration action
            updateCurrentIterationAction(i + 1);
        }

        // Flag the analysis task as finished
        datasetTask.setFinished();

        // Enabled actions again
        _settingsAction.getNumberOfIterationsAction().setEnabled(true);
    });

    // Initialize current iteration action
    updateCurrentIterationAction();

    // Initialize our points
    initializePoints();

    // Register for points datasets events using a custom callback function
    _eventListener.addSupportedEventType(static_cast<std::uint32_t>(EventType::DatasetAdded));
    _eventListener.addSupportedEventType(static_cast<std::uint32_t>(EventType::DatasetDataChanged));
    _eventListener.addSupportedEventType(static_cast<std::uint32_t>(EventType::DatasetRemoved));
    _eventListener.addSupportedEventType(static_cast<std::uint32_t>(EventType::DatasetDataSelectionChanged));

    _eventListener.registerDataEventByType(PointType, std::bind(&UMAPAnalysisPlugin::onDataEvent, this, std::placeholders::_1));
}

void UMAPAnalysisPlugin::onDataEvent(mv::DatasetEvent* dataEvent)
{
    // The data event has a type so that we know what type of data event occurred (e.g. data added, changed, removed, renamed, selection changes)
    switch (dataEvent->getType()) {

        // A points dataset was added
        case EventType::DatasetAdded:
        {
            // Cast the data event to a data added event
            const auto dataAddedEvent = static_cast<DatasetAddedEvent*>(dataEvent);

            // Get the GUI name of the added points dataset and print to the console
            qDebug() << dataAddedEvent->getDataset()->getGuiName() << "was added";

            break;
        }

        // Points dataset data has changed
        case EventType::DatasetDataChanged:
        {
            // Cast the data event to a data changed event
            const auto dataChangedEvent = static_cast<DatasetDataChangedEvent*>(dataEvent);

            // Get the GUI name of the points dataset of which the data changed and print to the console
            qDebug() << dataChangedEvent->getDataset()->getGuiName() << "data changed";

            break;
        }

        // Points dataset data was removed
        case EventType::DatasetRemoved:
        {
            // Cast the data event to a data removed event
            const auto dataRemovedEvent = static_cast<DatasetRemovedEvent*>(dataEvent);

            // Get the GUI name of the removed points dataset and print to the console
            qDebug() << dataRemovedEvent->getDataset()->getGuiName() << "was removed";

            break;
        }

        // Points dataset selection has changed
        case EventType::DatasetDataSelectionChanged:
        {
            // Cast the data event to a data selection changed event
            const auto dataSelectionChangedEvent = static_cast<DatasetDataSelectionChangedEvent*>(dataEvent);

            // Get points dataset
            const auto& changedDataSet = dataSelectionChangedEvent->getDataset();

            // Get the selection set that changed
            const auto selectionSet = changedDataSet->getSelection<Points>();

            // Print to the console
            qDebug() << changedDataSet->getGuiName() << "selection has changed";

            break;
        }

        default:
            break;
    }
}

AnalysisPlugin* UMAPAnalysisPluginFactory::produce()
{
    // Return a new instance of the UMAP analysis plugin
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
            for (auto dataset : datasets)
                getPluginInstance(dataset);
            });

        pluginTriggerActions << pluginTriggerAction;
    }

    return pluginTriggerActions;
}
