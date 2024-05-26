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

    StringAction& getCurrentIterationAction() { return _currentIterationAction; }
    IntegralAction& getNumberOfIterationsAction() { return _numberOfIterationsAction; }
    ButtonsGroupAction& getStartStopAction() { return _startStopActions; }
    TriggerAction& getStartAction() { return _startStopActions.getStartComputationAction(); }
    TriggerAction& getStopAction() { return _startStopActions.getStopComputationAction(); }
    OptionAction& getInitializeAction() { return _initializeActions; }

public:
    StringAction        _currentIterationAction;        /** Current iteration string action from the mv::gui namespace */
    IntegralAction      _numberOfIterationsAction;      /** Number of iterations action from the mv::gui namespace */
    ButtonsGroupAction  _startStopActions;              /** Buttons that control start and top of computation */
    OptionAction        _initializeActions;             /** How to initialize the embedding */
};
