#pragma once

#include "actions/GroupAction.h"
#include "actions/IntegralAction.h"
#include "actions/OptionAction.h"
#include "actions/ToggleAction.h"

using namespace mv::gui;

enum class KnnLibrary {
    ANNOY,
    HNSW,
};

enum class KnnMetric {
    EUCLIDEAN,
    COSINE,
    DOT,
};

class KnnParameters
{
public:
    KnnParameters() :
        _knnLibrary(KnnLibrary::ANNOY),
        _knn_metric(KnnMetric::EUCLIDEAN),
        _k(20),
        _AnnoyNumChecksAknn(512),
        _AnnoyNumTrees(8),
        _HNSW_M(16),
        _HNSW_ef_construction(200)
    {

    }

    void setKnnAlgorithm(KnnLibrary knnLibrary) { _knnLibrary = knnLibrary; }
    void setKnnDistanceMetric(KnnMetric knnDistanceMetric) { _knn_metric = knnDistanceMetric; }
    void setK(int k) { _k = k; }
    void setAnnoyNumChecks(int numChecks) { _AnnoyNumChecksAknn = numChecks; }
    void setAnnoyNumTrees(int numTrees) { _AnnoyNumTrees = numTrees; }
    void setHNSWm(int m) { _HNSW_M = m; }
    void setHNSWef(int ef) { _HNSW_ef_construction = ef; }

    KnnLibrary getKnnAlgorithm() const { return _knnLibrary; }
    KnnMetric getKnnDistanceMetric() const { return _knn_metric; }
    int getK() const { return _k; }
    int getAnnoyNumChecks() const { return _AnnoyNumChecksAknn; }
    int getAnnoyNumTrees() const { return _AnnoyNumTrees; }
    int getHNSWm() const { return _HNSW_M; }
    int getHNSWef() const { return _HNSW_ef_construction; }

private:

    KnnLibrary _knnLibrary;     /** Enum specifying which approximate nearest neighbour library to use for the similarity computation */
    KnnMetric _knn_metric;      /** Enum specifying which distance to compute knn with */

    int _k;                     /** Number of neighbors */

    int _AnnoyNumChecksAknn;                        /** Number of checks used in Annoy, more checks means more precision but slower computation */
    int _AnnoyNumTrees;                             /** Number of trees used in Annoy, more checks means more precision but slower computation */

    int            _HNSW_M;                         /** hnsw: construction time/accuracy trade-off  */
    int            _HNSW_ef_construction;           /** hnsw: maximum number of outgoing connections in the graph  */
};


/// //////////// ///
/// KNN Settings ///
/// //////////// ///

class KnnSettingsAction : public GroupAction
{
public:

    /**
     * Constructor
     * @param tsneSettingsAction Reference to TSNE settings action
     */
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

protected:
    KnnParameters           _knnParameters;             /** Pointer to Knn parameters */

    OptionAction            _knnAlgorithm;              /** Annoy or HNSW */
    IntegralAction          _kAction;                   /** Number of kNN */
    ToggleAction            _multithreadActions;        /** Whether to use multiple threads */
    IntegralAction          _numTreesAction;            /** Annoy parameter Trees action */
    IntegralAction          _numChecksAction;           /** Annoy parameter Checks action */
    IntegralAction          _mAction;                   /** HNSW parameter M action */
    IntegralAction          _efAction;                  /** HNSW parameter ef action */
};
