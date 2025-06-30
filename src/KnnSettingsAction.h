#pragma once

#include "actions/GroupAction.h"
#include "actions/IntegralAction.h"
#include "actions/OptionAction.h"
#include "actions/ToggleAction.h"

#include <string>

using namespace mv::gui;

enum class KnnAlgorithm {
    ANNOY,
    HNSW,
};

enum class KnnMetric {
    EUCLIDEAN,
    COSINE,
    DOT,
    CORRELATION,     // only works with hnsw
};

std::string printAlgorithm(const KnnAlgorithm& a);
std::string printMetric(const KnnMetric& m);

class KnnParameters
{
public:
    KnnParameters() :
        _knnAlgorithm(KnnAlgorithm::ANNOY),
        _knnMetric(KnnMetric::EUCLIDEAN),
        _k(20),
        _AnnoyNumChecksAknn(512),
        _AnnoyNumTrees(8),
        _HNSW_M(16),
        _HNSW_ef_construction(200)
    {

    }

    void setKnnAlgorithm(KnnAlgorithm knnLibrary) { _knnAlgorithm = knnLibrary; }
    void setKnnMetric(KnnMetric knnDistanceMetric) { _knnMetric = knnDistanceMetric; }
    void setK(int k) { _k = k; }
    void setAnnoyNumChecks(int numChecks) { _AnnoyNumChecksAknn = numChecks; }
    void setAnnoyNumTrees(int numTrees) { _AnnoyNumTrees = numTrees; }
    void setHNSWm(int m) { _HNSW_M = m; }
    void setHNSWef(int ef) { _HNSW_ef_construction = ef; }

    KnnAlgorithm getKnnAlgorithm() const { return _knnAlgorithm; }
    KnnMetric getKnnMetric() const { return _knnMetric; }
    int getK() const { return _k; }
    int getAnnoyNumChecks() const { return _AnnoyNumChecksAknn; }
    int getAnnoyNumTrees() const { return _AnnoyNumTrees; }
    int getHNSWm() const { return _HNSW_M; }
    int getHNSWef() const { return _HNSW_ef_construction; }

private:

    KnnAlgorithm    _knnAlgorithm;              /** Enum specifying which approximate nearest neighbour library to use for the similarity computation */
    KnnMetric       _knnMetric;                 /** Enum specifying which distance to compute knn with */

    int             _k;                         /** Number of neighbors */

    int             _AnnoyNumChecksAknn;        /** Number of checks used in Annoy, more checks means more precision but slower computation */
    int             _AnnoyNumTrees;             /** Number of trees used in Annoy, more checks means more precision but slower computation */

    int             _HNSW_M;                    /** hnsw: construction time/accuracy trade-off  */
    int             _HNSW_ef_construction;      /** hnsw: maximum number of outgoing connections in the graph  */
};


/// //////////// ///
/// KNN Settings ///
/// //////////// ///

class KnnSettingsAction : public GroupAction
{
public:

    KnnSettingsAction(QObject* parent = nullptr);

    KnnParameters getKnnParameters() const { return _knnParameters; }

public: // Action getters

    OptionAction& getKnnAlgorithmAction() { return _knnAlgorithm; };
    IntegralAction& getKAction() { return _kAction; };
    ToggleAction& getMultithreadAction() { return _multithreadActions; }
    IntegralAction& getNumTreesAction() { return _numTreesAction; };
    IntegralAction& getNumChecksAction() { return _numChecksAction; };
    IntegralAction& getMAction() { return _mAction; };
    IntegralAction& getEfAction() { return _efAction; };

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
    KnnParameters           _knnParameters;             /** Knn parameters */

    OptionAction            _knnAlgorithm;              /** Annoy or HNSW */
    OptionAction            _knnMetric;                 /** Euclidean, Cosine, Dot, etc. */
    IntegralAction          _kAction;                   /** Number of kNN */
    ToggleAction            _multithreadActions;        /** Whether to use multiple threads */
    IntegralAction          _numTreesAction;            /** Annoy parameter Trees action */
    IntegralAction          _numChecksAction;           /** Annoy parameter Checks action */
    IntegralAction          _mAction;                   /** HNSW parameter M action */
    IntegralAction          _efAction;                  /** HNSW parameter ef action */
};
