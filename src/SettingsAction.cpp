#include "SettingsAction.h"

SettingsAction::SettingsAction(QObject* parent) :
    GroupAction(parent, "SettingsAction", true),
    _currentEpochAction(this, "Epoch"),
    _numberOfEpochsAction(this, "Epochs", 0, 10000, 500),
    _startStopActions(this),
    _multithreadActions(this, "Use multithread", false),
    _initializeActions(this, "Initialization", {"Spectral", "Random"})
{
    setText("UMAP Settings");

    _currentEpochAction.setEnabled(false);

    _currentEpochAction.setToolTip("Current epoch index");
    _numberOfEpochsAction.setToolTip("Number of epochs to compute");
    _startStopActions.setToolTip("Computation control");
    _initializeActions.setToolTip("Use spectral decomposition of the graph Laplacian or random initialization");
    _multithreadActions.setToolTip("Use more memory to increase computation speed");

    _initializeActions.setCurrentIndex(0);

    addAction(&_currentEpochAction);
    addAction(&_numberOfEpochsAction);
    addAction(&_startStopActions);
    addAction(&_initializeActions);

#ifdef USE_OPENMP
    addAction(&_multithreadActions);
#endif
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