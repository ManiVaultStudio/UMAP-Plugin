#include "SettingsAction.h"

SettingsAction::SettingsAction(QObject* parent) :
    GroupAction(parent, "SettingsAction", true),
    _currentIterationAction(this, "Iterations"),
    _numberOfIterationsAction(this, "Number of iterations", 0, 10000, 500),
    _startStopActions(this)
{
    setText("UMAP Settings");

    _currentIterationAction.setEnabled(false);

    _currentIterationAction.setToolTip("Current iteration index");
    _numberOfIterationsAction.setToolTip("Number of iterations to compute");
    _startStopActions.setToolTip("Computation control");

    addAction(&_currentIterationAction);
    addAction(&_numberOfIterationsAction);
    addAction(&_startStopActions);
}


/// ////////////////// ///
/// Start Stop Buttons ///
/// ////////////////// ///

ButtonsGroupAction::ButtonsGroupAction(QObject* parent) :
    HorizontalGroupAction(parent, "UmapComputationAction"),
    _startComputationAction(this, "Start"),
    _continueComputationAction(this, "Continue"),
    _stopComputationAction(this, "Stop")
{
    setText("Computation");

    addAction(&_startComputationAction);
    addAction(&_continueComputationAction);
    addAction(&_stopComputationAction);

    _startComputationAction.setToolTip("Start computation");
    _continueComputationAction.setToolTip("Continue");
    _stopComputationAction.setToolTip("Stop");

    _continueComputationAction.setEnabled(false);
}