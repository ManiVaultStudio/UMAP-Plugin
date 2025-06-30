#include "KnnSettingsAction.h"

#include <string>

std::string printAlgorithm(const KnnAlgorithm& a) {
    switch (a)
    {
    case KnnAlgorithm::ANNOY: return "Annoy";
    case KnnAlgorithm::HNSW: return "HNSW";
    }

    return "Unkown";
}

std::string printMetric(const KnnMetric& m) {
    switch (m)
    {
    case KnnMetric::EUCLIDEAN: return "Euclidean";
    case KnnMetric::DOT: return "Dot";
    case KnnMetric::COSINE: return "Cosine";
    case KnnMetric::CORRELATION: return "Correlation";
    }

    return "Unkown";
}

KnnSettingsAction::KnnSettingsAction(QObject* parent) :
    GroupAction(parent, "Knn Settings"),
    _knnParameters(),
    _knnAlgorithm(this, "Algorithm"),
    _knnMetric(this, "Metric"),
    _kAction(this, "Number kNN"),
    _multithreadActions(this, "Use multithreading", true),
    _numTreesAction(this, "Annoy Trees"),
    _numChecksAction(this, "Annoy Checks"),
    _mAction(this, "HNSW M"),
    _efAction(this, "HNSW ef")
{
    setText("Knn Settings");
    setSerializationName("Knn Settings");

    addAction(&_knnAlgorithm);
    addAction(&_knnMetric);
    addAction(&_kAction);

#ifdef USE_OPENMP
    addAction(&_multithreadActions);
#endif

    addAction(&_numTreesAction);
    addAction(&_numChecksAction);
    addAction(&_mAction);
    addAction(&_efAction);

    _multithreadActions.setToolTip("Use more memory to increase computation speed");

    _numTreesAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numChecksAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _mAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _efAction.setDefaultWidgetFlags(IntegralAction::SpinBox);

    _knnAlgorithm.initialize({ "Annoy", "HNSW" }, "Annoy");
    _knnMetric.initialize({ "Euclidean", "Cosine", "Inner", "Correlation"}, "Euclidean");
    _kAction.initialize(1, 150, 20);
    _numTreesAction.initialize(1, 10000, 8);
    _numChecksAction.initialize(1, 10000, 512);
    _mAction.initialize(2, 300, 16);
    _efAction.initialize(1, 10000, 200);

#ifndef NDEBUG
    _multithreadActions.setChecked(false);
#endif // !NDEBUG

    const auto updateKnnAlgorithm = [this]() -> void {
        if(_knnAlgorithm.getCurrentIndex() == 0)
            _knnParameters.setKnnAlgorithm(KnnAlgorithm::ANNOY);
        else if (_knnAlgorithm.getCurrentIndex() == 1)
            _knnParameters.setKnnAlgorithm(KnnAlgorithm::HNSW);
        };

    const auto updateKnnMetric = [this]() -> void {
        if(_knnAlgorithm.getCurrentIndex() == 0)
            _knnParameters.setKnnMetric(KnnMetric::EUCLIDEAN);
        else if (_knnAlgorithm.getCurrentIndex() == 1)
            _knnParameters.setKnnMetric(KnnMetric::COSINE);
        else if (_knnAlgorithm.getCurrentIndex() == 2)
            _knnParameters.setKnnMetric(KnnMetric::DOT);
        else if (_knnAlgorithm.getCurrentIndex() == 3) {
            _knnAlgorithm.setCurrentIndex(0); // only implemented for hnsw
            _knnParameters.setKnnMetric(KnnMetric::CORRELATION);
        }
        };

    const auto updateNumTrees = [this]() -> void {
        _knnParameters.setAnnoyNumTrees(_numTreesAction.getValue());
        };

    const auto updateK = [this]() -> void {
        _knnParameters.setK(_kAction.getValue());
        };

    const auto updateNumChecks = [this]() -> void {
        _knnParameters.setAnnoyNumChecks(_numChecksAction.getValue());
        };

    const auto updateM = [this]() -> void {
        _knnParameters.setHNSWm(_mAction.getValue());
        };

    const auto updateEf = [this]() -> void {
        _knnParameters.setHNSWef(_efAction.getValue());
        };

    const auto updateReadOnly = [this]() -> void {
        const auto enable = !isReadOnly();

        _knnAlgorithm.setEnabled(enable);
        _knnMetric.setEnabled(enable);
        _kAction.setEnabled(enable);
        _multithreadActions.setEnabled(enable);
        _numTreesAction.setEnabled(enable);
        _numChecksAction.setEnabled(enable);
        _mAction.setEnabled(enable);
        _efAction.setEnabled(enable);
        };

    connect(&_knnAlgorithm, &OptionAction::currentIndexChanged, this, [this, updateKnnAlgorithm](const std::int32_t& value) {
        updateKnnAlgorithm();
        });

    connect(&_knnMetric, &OptionAction::currentIndexChanged, this, [this, updateKnnMetric](const std::int32_t& value) {
        updateKnnMetric();
        });

    connect(&_kAction, &IntegralAction::valueChanged, this, [this, updateK](const std::int32_t& value) {
        updateK();
        });

    connect(&_numTreesAction, &IntegralAction::valueChanged, this, [this, updateNumTrees](const std::int32_t& value) {
        updateNumTrees();
        });

    connect(&_numChecksAction, &IntegralAction::valueChanged, this, [this, updateNumChecks](const std::int32_t& value) {
        updateNumChecks();
        });

    connect(&_mAction, &IntegralAction::valueChanged, this, [this, updateM](const std::int32_t& value) {
        updateM();
        });

    connect(&_efAction, &IntegralAction::valueChanged, this, [this, updateEf](const std::int32_t& value) {
        updateEf();
        });

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
        });

    updateKnnAlgorithm();
    updateKnnMetric();
    updateK();
    updateNumTrees();
    updateNumChecks();
    updateM();
    updateEf();
    updateReadOnly();
}

void KnnSettingsAction::fromVariantMap(const QVariantMap& variantMap)
{
    GroupAction::fromVariantMap(variantMap);

    _knnAlgorithm.fromParentVariantMap(variantMap);
    _kAction.fromParentVariantMap(variantMap);
    _multithreadActions.fromParentVariantMap(variantMap);
    _numTreesAction.fromParentVariantMap(variantMap);
    _numChecksAction.fromParentVariantMap(variantMap);
    _mAction.fromParentVariantMap(variantMap);
    _efAction.fromParentVariantMap(variantMap);
}

QVariantMap KnnSettingsAction::toVariantMap() const
{
    QVariantMap variantMap = GroupAction::toVariantMap();

    _knnAlgorithm.insertIntoVariantMap(variantMap);
    _kAction.insertIntoVariantMap(variantMap);
    _multithreadActions.insertIntoVariantMap(variantMap);
    _numTreesAction.insertIntoVariantMap(variantMap);
    _numChecksAction.insertIntoVariantMap(variantMap);
    _mAction.insertIntoVariantMap(variantMap);
    _efAction.insertIntoVariantMap(variantMap);

    return variantMap;
}
