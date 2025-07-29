#include "SettingsAction.h"

SettingsAction::SettingsAction(QObject* parent) :
    GroupAction(parent, "SettingsAction", true),
    _currentEpochAction(this, "Epoch"),
    _numberOfEpochsAction(this, "Epochs", 0, 5000, 500),
    _startStopActions(this),
    _initializeActions(this, "Initialization", {"Spectral", "Random"}),
    _numberEmbDimsAction(this, "Emb dims", 1, 10, 2) // Max may be increased up to data dim
{
    setText("UMAP Settings");
    setSerializationName("UMAP Settings");

    _currentEpochAction.setEnabled(false);

    _currentEpochAction.setToolTip("Current epoch index");
    _numberOfEpochsAction.setToolTip("Number of epochs to compute the gradient descent, i.e., optimization iterations");
    _startStopActions.setToolTip("Computation control");
    _initializeActions.setToolTip("Use spectral decomposition of the graph Laplacian or random initialization");
    _numberEmbDimsAction.setToolTip("Number of embedding output dimensions");

    _initializeActions.setCurrentIndex(0);

    addAction(&_currentEpochAction);
    addAction(&_numberOfEpochsAction);
    addAction(&_initializeActions);
    addAction(&_numberEmbDimsAction);
    addAction(&_startStopActions);
}

void SettingsAction::fromVariantMap(const QVariantMap& variantMap)
{
    GroupAction::fromVariantMap(variantMap);

    _currentEpochAction.fromParentVariantMap(variantMap);
    _numberOfEpochsAction.fromParentVariantMap(variantMap);
    _initializeActions.fromParentVariantMap(variantMap);
    _numberEmbDimsAction.fromParentVariantMap(variantMap);
}

QVariantMap SettingsAction::toVariantMap() const
{
    QVariantMap variantMap = GroupAction::toVariantMap();

    _currentEpochAction.insertIntoVariantMap(variantMap);
    _numberOfEpochsAction.insertIntoVariantMap(variantMap);
    _initializeActions.insertIntoVariantMap(variantMap);
    _numberEmbDimsAction.insertIntoVariantMap(variantMap);

    return variantMap;
}

/// ////////////////// ///
/// Start Stop Buttons ///
/// ////////////////// ///

ButtonsGroupAction::ButtonsGroupAction(QObject* parent) :
    HorizontalGroupAction(parent, "UmapComputationAction"),
    _startComputationAction(this, "Start"),
    _stopComputationAction(this, "Stop")
{
    setText("Compute");

    addAction(&_startComputationAction);
    addAction(&_stopComputationAction);

    _startComputationAction.setToolTip("Start computation");
    _stopComputationAction.setToolTip("Stop computation");

    _stopComputationAction.setEnabled(false);
}