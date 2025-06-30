#pragma once

#include <cstdint>

#include "actions/DecimalAction.h"
#include "actions/GroupAction.h"
#include "actions/IntegralAction.h"

using namespace mv::gui;

// Defaults as in https://umap-learn.readthedocs.io/en/latest/api.html
struct AdvancedParameters
{
    float     local_connectivity   = 1.f;
    float     bandwidth            = 1.f;
    float     mix_ratio            = 1.f;
    float     spread               = 1.f;
    float     min_dist             = 0.1f;
    float     a                    = 0.f;   // if 0, set based on spread, min_dist automatically
    float     b                    = 0.f;   // if 0, set based on spread, min_dist automatically
    float     repulsion_strength   = 1.f;
    float     learning_rate        = 1.f;
    float     negative_sample_rate = 5.f;
    uint64_t  seed                 = 1234567890;
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
