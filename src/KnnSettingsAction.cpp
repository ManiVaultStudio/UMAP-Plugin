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
    addAction(&_numTreesAction);
    addAction(&_numChecksAction);
    addAction(&_mAction);
    addAction(&_efAction);

#ifdef _OPENMP
    addAction(&_multithreadActions);
#endif

    _multithreadActions.setToolTip("Use more memory to increase computation speed");
    _numTreesAction.setToolTip("Annoy: provided during build time and affects the build time and the index size.\nA larger value will give more accurate results, but larger indexes.");
    _numChecksAction.setToolTip("Annoy: provided in runtime and affects the search performance.\nA larger value will give more accurate results, but will take longer time to return.");
    _mAction.setToolTip("HNSW: number of bi-directional links created for every new element during construction. Reasonable range for M is 2-100.");
    _efAction.setToolTip("HNSW: Sets both search and construction ef parameter.");

    _numTreesAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numChecksAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _mAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _efAction.setDefaultWidgetFlags(IntegralAction::SpinBox);

    _knnAlgorithm.initialize({ "Annoy", "HNSW" }, "HNSW");
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
        auto currentAlgorithmIntex = _knnAlgorithm.getCurrentIndex();

        KnnAlgorithm newAlg = KnnAlgorithm::HNSW;

        if(currentAlgorithmIntex == 0)
            newAlg = KnnAlgorithm::ANNOY;
        else if (currentAlgorithmIntex == 1)
            newAlg = KnnAlgorithm::HNSW;

        _knnParameters.setKnnAlgorithm(newAlg);
        };

    const auto updateKnnMetric = [this]() -> void {
        auto currentMetricIntex = _knnMetric.getCurrentIndex();

        KnnMetric newMetric = KnnMetric::EUCLIDEAN;

        if(currentMetricIntex == 0)
            newMetric = KnnMetric::EUCLIDEAN;
        else if (currentMetricIntex == 1)
            newMetric = KnnMetric::COSINE;
        else if (currentMetricIntex == 2)
            newMetric = KnnMetric::DOT;
        else if (currentMetricIntex == 3) {
            newMetric = KnnMetric::CORRELATION;
            _knnAlgorithm.setCurrentIndex(1); // only implemented for hnsw
        }

        _knnParameters.setKnnMetric(newMetric);
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
