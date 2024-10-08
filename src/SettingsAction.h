#pragma once

#include "actions/GroupAction.h"
#include "actions/HorizontalGroupAction.h"
#include "actions/IntegralAction.h"
#include "actions/OptionAction.h"
#include "actions/StringAction.h"
#include "actions/TriggerAction.h"

using namespace mv::gui;

/// ////////////////// ///
/// Start Stop Buttons ///
/// ////////////////// ///

class ButtonsGroupAction : public HorizontalGroupAction
{
public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    ButtonsGroupAction(QObject* parent);

    void changeEnabled(bool enable)
    {
        _startComputationAction.setEnabled(enable);
        _stopComputationAction.setEnabled(!enable);
    }

    void setStarted()
    {
        changeEnabled(false);
    }

    void setFinished()
    {
        changeEnabled(true);
    }

public: // Action getters

    TriggerAction& getStartComputationAction() { return _startComputationAction; }
    TriggerAction& getStopComputationAction() { return _stopComputationAction; }

protected:
    TriggerAction   _startComputationAction;        /** Start computation action */
    TriggerAction   _stopComputationAction;         /** Stop computation action */
};

/**
 * SettingsAction
 * 
 * Class that houses general settings for the UMAP analysis plugin
 *
 */
class SettingsAction : public GroupAction
{
public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    SettingsAction(QObject* parent = nullptr);

    void changeEnabled(bool enable)
    {
        _numberOfEpochsAction.setEnabled(enable);
        _initializeActions.setEnabled(enable);
    }

    void setStarted()
    {
        _startStopActions.setStarted();
        changeEnabled(false);
    }

    void setFinished()
    {
        _startStopActions.setFinished();
        changeEnabled(true);
    }

public: // Action getters

    StringAction& getCurrentEpochAction() { return _currentEpochAction; }
    IntegralAction& getNumberOfEpochsAction() { return _numberOfEpochsAction; }
    ButtonsGroupAction& getStartStopAction() { return _startStopActions; }
    TriggerAction& getStartAction() { return _startStopActions.getStartComputationAction(); }
    TriggerAction& getStopAction() { return _startStopActions.getStopComputationAction(); }
    OptionAction& getInitializeAction() { return _initializeActions; }
    IntegralAction& getNumberEmbDimsAction() { return _numberEmbDimsAction; }

public: // Serialization

    /**
     * Load plugin from variant map
     * @param Variant map representation of the plugin
     */
    void fromVariantMap(const QVariantMap& variantMap) override;

    /**
     * Save plugin to variant map
     * @return Variant map representation of the plugin
     */
    QVariantMap toVariantMap() const override;

protected:
    StringAction        _currentEpochAction;            /** Current epoch string  */
    IntegralAction      _numberOfEpochsAction;          /** Number of epocs */
    ButtonsGroupAction  _startStopActions;              /** Buttons that control start and top of computation */
    OptionAction        _initializeActions;             /** How to initialize the embedding */
    IntegralAction      _numberEmbDimsAction;           /** Number of emebdding dimenisons */
};
