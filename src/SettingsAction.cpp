#include "SettingsAction.h"

SettingsAction::SettingsAction(QObject* parent) :
    GroupAction(parent, "SettingsAction", true),
    _currentIterationAction(this, "Iterations"),
    _numberOfIterationsAction(this, "Number of iterations", 0, 10000, 500),
    _startAnalysisAction(this, "Start analysis"),
    _updateAction(this, "Start analysis")
{
    setText("UMAP Settings");

    _currentIterationAction.setEnabled(false);

    _currentIterationAction.setToolTip("Current iteration index");
    _numberOfIterationsAction.setToolTip("Number of iterations to compute");
    _updateAction.setToolTip("Update");

    addAction(&_currentIterationAction);
    addAction(&_numberOfIterationsAction);
    addAction(&_startAnalysisAction);
    addAction(&_updateAction);
}
