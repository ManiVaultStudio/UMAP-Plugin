#pragma once

#include "actions/GroupAction.h"
#include "actions/HorizontalGroupAction.h"
#include "actions/IntegralAction.h"
#include "actions/OptionAction.h"
#include "actions/StringAction.h"
#include "actions/ToggleAction.h"
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

    void changeEnabled(bool start, bool stop)
    {
        _startComputationAction.setEnabled(start);
        _stopComputationAction.setEnabled(stop);
    }

    void setStarted()
    {
        changeEnabled(false, true);
    }

    void setFinished()
    {
        changeEnabled(true, false);
    }

public: // Action getters

    TriggerAction& getStartComputationAction() { return _startComputationAction; }
    TriggerAction& getStopComputationAction() { return _stopComputationAction; }

protected:
    TriggerAction   _startComputationAction;        /** Start computation action */
    TriggerAction   _stopComputationAction;         /** Stop computation action */
};

/**
 * UMAPSettingsAction class
 * 
 * Class that houses settings for the UMAP analysis plugin
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

public: // Action getters

    StringAction& getCurrentEpochAction() { return _currentEpochAction; }
    IntegralAction& getNumberOfEpochsAction() { return _numberOfEpochsAction; }
    ButtonsGroupAction& getStartStopAction() { return _startStopActions; }
    TriggerAction& getStartAction() { return _startStopActions.getStartComputationAction(); }
    TriggerAction& getStopAction() { return _startStopActions.getStopComputationAction(); }
    OptionAction& getInitializeAction() { return _initializeActions; }
    ToggleAction& getMultithreadAction() { return _multithreadActions; }

public:
    StringAction        _currentEpochAction;            /** Current epoch string  */
    IntegralAction      _numberOfEpochsAction;          /** Number of iterations */
    ButtonsGroupAction  _startStopActions;              /** Buttons that control start and top of computation */
    OptionAction        _initializeActions;             /** How to initialize the embedding */
    ToggleAction        _multithreadActions;            /** Whether to use multiple threads */
};
