#pragma once

#include "actions/DecimalAction.h"
#include "actions/GroupAction.h"
#include "actions/IntegralAction.h"

using namespace mv::gui;

struct AdvancedParameters
{
    float     local_connectivity = 1.0;
    float     bandwidth = 1;
    float     mix_ratio = 1;
    float     spread = 1;
    float     min_dist = 0.01;
    float     a = 0;
    float     b = 0;
    float     repulsion_strength = 1;
    float     learning_rate = 1;
    float     negative_sample_rate = 5;
    uint64_t  seed = 1234567890;
};


/// ///////////////// ///
/// Advanced Settings ///
/// ///////////////// ///

class AdvancedSettingsAction : public GroupAction
{
public:

    /**
     * Constructor
     * @param tsneSettingsAction Reference to TSNE settings action
     */
    AdvancedSettingsAction(QObject* parent = nullptr);

    AdvancedParameters getAdvParameters() const { return _advParameters; }

public: // Action getters

    DecimalAction& getLocalConnectivityAction() { return _local_connectivity; };
    DecimalAction& getBandwidthAction() { return _bandwidth; };
    DecimalAction& getMixRatioAction() { return _mix_ratio; };
    DecimalAction& getSpreadAction() { return _spread; };
    DecimalAction& getMinDistAction() { return _min_dist; };
    DecimalAction& getAAction() { return _a; };
    DecimalAction& getBAction() { return _b; };
    DecimalAction& getRepulsionStrengthAction() { return _repulsion_strength; };
    DecimalAction& getLearningRateAction() { return _learning_rate; };
    DecimalAction& getNegativeSampleRateAction() { return _negative_sample_rate; };
    IntegralAction& getSeedAction() { return _seed; };

protected:
    AdvancedParameters      _advParameters;             /** Advanced parameters */

    DecimalAction           _local_connectivity;
    DecimalAction           _bandwidth;
    DecimalAction           _mix_ratio;
    DecimalAction           _spread;
    DecimalAction           _min_dist;
    DecimalAction           _a;
    DecimalAction           _b;
    DecimalAction           _repulsion_strength;
    DecimalAction           _learning_rate;
    DecimalAction           _negative_sample_rate;
    IntegralAction          _seed;
};
