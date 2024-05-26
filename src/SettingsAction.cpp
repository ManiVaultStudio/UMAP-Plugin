#include "SettingsAction.h"

SettingsAction::SettingsAction(QObject* parent) :
    GroupAction(parent, "SettingsAction", true),
    _currentIterationAction(this, "Iterations"),
    _numberOfIterationsAction(this, "Number of iterations", 0, 10000, 500),
    _startStopActions(this),
    _initializeActions(this, "Initialization", {"Spectral", "Random"})
{
    setText("UMAP Settings");

    _currentIterationAction.setEnabled(false);

    _currentIterationAction.setToolTip("Current iteration index");
    _numberOfIterationsAction.setToolTip("Number of iterations to compute");
    _startStopActions.setToolTip("Computation control");
    _initializeActions.setToolTip("Use spectral decomposition of the graph Laplacian or random initialization");

    _initializeActions.setCurrentIndex(0);

    addAction(&_currentIterationAction);
    addAction(&_numberOfIterationsAction);
    addAction(&_startStopActions);
    addAction(&_initializeActions);
}


/// ////////////////// ///
/// Start Stop Buttons ///
/// ////////////////// ///

ButtonsGroupAction::ButtonsGroupAction(QObject* parent) :
    HorizontalGroupAction(parent, "UmapComputationAction"),
    _startComputationAction(this, "Start"),
    _stopComputationAction(this, "Stop")
{
    setText("Computation");

    addAction(&_startComputationAction);
    addAction(&_stopComputationAction);

    _startComputationAction.setToolTip("Start computation");
    _stopComputationAction.setToolTip("Stop");

    _stopComputationAction.setEnabled(false);
}