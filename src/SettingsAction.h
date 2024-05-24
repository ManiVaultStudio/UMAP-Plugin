#pragma once

#include "actions/GroupAction.h"
#include "actions/IntegralAction.h"
#include "actions/StringAction.h"
#include "actions/TriggerAction.h"

/** All GUI related classes are in the HDPS Graphical User Interface namespace */
using namespace mv::gui;

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
    TriggerAction& getStartAnalysisAction() { return _startAnalysisAction; }
    TriggerAction& getUpdateAction() { return _updateAction; }

public:
    StringAction    _currentIterationAction;        /** Current iteration string action from the mv::gui namespace */
    IntegralAction  _numberOfIterationsAction;      /** Number of iterations action from the mv::gui namespace */
    TriggerAction   _startAnalysisAction;           /** Start loop  trigger action from the mv::gui namespace */
    TriggerAction   _updateAction;           /** Start loop  trigger action from the mv::gui namespace */
};
