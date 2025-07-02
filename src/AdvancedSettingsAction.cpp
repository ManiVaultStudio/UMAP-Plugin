#include "AdvancedSettingsAction.h"

AdvancedSettingsAction::AdvancedSettingsAction(QObject* parent) :
    GroupAction(parent, "Advanced Settings"),
    _advParameters(),
    _local_connectivity(this, "local connectivity"),
    _bandwidth(this, "bandwidth"),
    _mix_ratio(this, "mix ratio", true),
    _spread(this, "spread"),
    _min_dist(this, "min dist"),
    _a(this, "a"),
    _b(this, "b"),
    _repulsion_strength(this, "repulsion strength"),
    _learning_rate(this, "learning rate"),
    _negative_sample_rate(this, "negative sample rate"),
    _multithreadActions(this, "Multithreading (for many cores)", false),
    _seed(this, "seed")
{
    setText("Advanced Settings");
    setSerializationName("Advanced Settings");

    addAction(&_local_connectivity);
    addAction(&_bandwidth);
    addAction(&_mix_ratio);
    addAction(&_spread);
    addAction(&_min_dist);
    addAction(&_a);
    addAction(&_b);
    addAction(&_repulsion_strength);
    addAction(&_learning_rate);
    addAction(&_negative_sample_rate);
    addAction(&_seed);

#ifdef _OPENMP
    addAction(&_multithreadActions);
#endif

    _local_connectivity.initialize(0, 10, _advParameters.local_connectivity, 2);
    _bandwidth.initialize(0, 10, _advParameters.bandwidth, 2);
    _mix_ratio.initialize(0, 1, _advParameters.mix_ratio, 2);
    _spread.initialize(0.01, 10, _advParameters.spread, 2);
    _min_dist.initialize(0.001, 10, _advParameters.min_dist, 3);
    _a.initialize(0, 10, _advParameters.a, 2);
    _b.initialize(0, 1, _advParameters.b, 2);
    _repulsion_strength.initialize(0, 10, _advParameters.repulsion_strength, 2);
    _learning_rate.initialize(0, 10, _advParameters.learning_rate, 2);
    _negative_sample_rate.initialize(0, 10, _advParameters.negative_sample_rate, 2);
    _seed.initialize(1, 2000000000, _advParameters.seed);

    _multithreadActions.setToolTip("Use more memory to increase computation speed\nThis option is only useful if you have many cores\n4 is not a lot.");

    // Close the advanced settings by default
    collapse();

    const auto update_local_connectivity = [this]() -> void {
        _advParameters.local_connectivity = _local_connectivity.getValue();
        };

    const auto update_bandwidth = [this]() -> void {
        _advParameters.bandwidth = _bandwidth.getValue();
        };

    const auto update_mix_ratio = [this]() -> void {
        _advParameters.mix_ratio = _mix_ratio.getValue();
        };

    const auto update_spread = [this]() -> void {
        _advParameters.spread = _spread.getValue();
        };

    const auto update_min_dist = [this]() -> void {
        _advParameters.min_dist = _min_dist.getValue();
        };

    const auto update_a = [this]() -> void {
        _advParameters.a = _a.getValue();
        };

    const auto update_b = [this]() -> void {
        _advParameters.b = _b.getValue();
        };

    const auto update_repulsion_strength = [this]() -> void {
        _advParameters.repulsion_strength = _repulsion_strength.getValue();
        };

    const auto update_learning_rate = [this]() -> void {
        _advParameters.learning_rate = _learning_rate.getValue();
        };

    const auto update_negative_sample_rate = [this]() -> void {
        _advParameters.negative_sample_rate = _negative_sample_rate.getValue();
        };

    const auto update_seed = [this]() -> void {
        _advParameters.seed = _seed.getValue();
        };

    const auto updateReadOnly = [this]() -> void {
        const auto enable = !isReadOnly();

        _local_connectivity.setEnabled(enable);
        _bandwidth.setEnabled(enable);
        _mix_ratio.setEnabled(enable);
        _spread.setEnabled(enable);
        _min_dist.setEnabled(enable);
        _a.setEnabled(enable);
        _b.setEnabled(enable);
        _repulsion_strength.setEnabled(enable);
        _learning_rate.setEnabled(enable);
        _negative_sample_rate.setEnabled(enable);
        _multithreadActions.setEnabled(enable);
        _seed.setEnabled(enable);
        };

    connect(&_local_connectivity, &DecimalAction::valueChanged, this, [this, update_local_connectivity](const float& value) {
        update_local_connectivity();
        });

    connect(&_bandwidth, &DecimalAction::valueChanged, this, [this, update_bandwidth](const float& value) {
        update_bandwidth();
        });

    connect(&_mix_ratio, &DecimalAction::valueChanged, this, [this, update_mix_ratio](const float& value) {
        update_mix_ratio();
        });

    connect(&_spread, &DecimalAction::valueChanged, this, [this, update_spread](const float& value) {
        update_spread();
        });

    connect(&_min_dist, &DecimalAction::valueChanged, this, [this, update_min_dist](const float& value) {
        update_min_dist();
        });

    connect(&_a, &DecimalAction::valueChanged, this, [this, update_a](const float& value) {
        update_a();
        });

    connect(&_b, &DecimalAction::valueChanged, this, [this, update_b](const float& value) {
        update_b();
        });

    connect(&_repulsion_strength, &DecimalAction::valueChanged, this, [this, update_repulsion_strength](const float& value) {
        update_repulsion_strength();
        });

    connect(&_learning_rate, &DecimalAction::valueChanged, this, [this, update_learning_rate](const float& value) {
        update_learning_rate();
        });

    connect(&_negative_sample_rate, &DecimalAction::valueChanged, this, [this, update_negative_sample_rate](const float& value) {
        update_negative_sample_rate();
        });

    connect(&_seed, &IntegralAction::valueChanged, this, [this, update_seed](const std::int32_t& value) {
        update_seed();
        });

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
        });

    update_local_connectivity();
    update_bandwidth();
    update_mix_ratio();
    update_spread();
    update_min_dist();
    update_a();
    update_b();
    update_repulsion_strength();
    update_learning_rate();
    update_negative_sample_rate();
    update_seed();
    updateReadOnly();
}

void AdvancedSettingsAction::fromVariantMap(const QVariantMap& variantMap)
{
    GroupAction::fromVariantMap(variantMap);

    _local_connectivity.fromParentVariantMap(variantMap);
    _bandwidth.fromParentVariantMap(variantMap);
    _mix_ratio.fromParentVariantMap(variantMap);
    _spread.fromParentVariantMap(variantMap);
    _min_dist.fromParentVariantMap(variantMap);
    _a.fromParentVariantMap(variantMap);
    _b.fromParentVariantMap(variantMap);
    _repulsion_strength.fromParentVariantMap(variantMap);
    _learning_rate.fromParentVariantMap(variantMap);
    _negative_sample_rate.fromParentVariantMap(variantMap);
    _multithreadActions.fromParentVariantMap(variantMap);
    _seed.fromParentVariantMap(variantMap);
}

QVariantMap AdvancedSettingsAction::toVariantMap() const
{
    QVariantMap variantMap = GroupAction::toVariantMap();

    _local_connectivity.insertIntoVariantMap(variantMap);
    _bandwidth.insertIntoVariantMap(variantMap);
    _mix_ratio.insertIntoVariantMap(variantMap);
    _spread.insertIntoVariantMap(variantMap);
    _min_dist.insertIntoVariantMap(variantMap);
    _a.insertIntoVariantMap(variantMap);
    _b.insertIntoVariantMap(variantMap);
    _repulsion_strength.insertIntoVariantMap(variantMap);
    _learning_rate.insertIntoVariantMap(variantMap);
    _negative_sample_rate.insertIntoVariantMap(variantMap);
    _multithreadActions.insertIntoVariantMap(variantMap);
    _seed.insertIntoVariantMap(variantMap);

    return variantMap;
}
