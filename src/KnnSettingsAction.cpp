#include "KnnSettingsAction.h"

KnnSettingsAction::KnnSettingsAction(QObject* parent) :
    GroupAction(parent, "Knn Settings"),
    _knnParameters(),
    _knnAlgorithm(this, "Algorithm"),
    _kAction(this, "Number kNN"),
    _multithreadActions(this, "Use multithread", true),
    _numTreesAction(this, "Annoy Trees"),
    _numChecksAction(this, "Annoy Checks"),
    _mAction(this, "HNSW M"),
    _efAction(this, "HNSW ef")
{
    addAction(&_knnAlgorithm);
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
    _kAction.initialize(1, 150, 20);
    _numTreesAction.initialize(1, 10000, 8);
    _numChecksAction.initialize(1, 10000, 512);
    _mAction.initialize(2, 300, 16);
    _efAction.initialize(1, 10000, 200);

#ifndef NDEBUG
    _multithreadActions.setChecked(false);
#endif // !NDEBUG

    _multithreadActions.setChecked(false);

    const auto updateKnnAlgorithm = [this]() -> void {
        if(_knnAlgorithm.getCurrentIndex() == 0)
            _knnParameters.setKnnAlgorithm(KnnLibrary::ANNOY);
        else if (_knnAlgorithm.getCurrentIndex() == 1)
            _knnParameters.setKnnAlgorithm(KnnLibrary::HNSW);
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
    updateK();
    updateNumTrees();
    updateNumChecks();
    updateM();
    updateEf();
    updateReadOnly();
}
