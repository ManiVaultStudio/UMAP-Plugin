#pragma once

#include <AnalysisPlugin.h>

#include <PointData/PointData.h>

#include "AdvancedSettingsAction.h"
#include "KnnSettingsAction.h"
#include "SettingsAction.h"

#define UMAPPP_CUSTOM_NEIGHBORS // due to an older hsnw version in knncolle
#include "umappp/Umap.hpp"
#undef UMAPPP_CUSTOM_NEIGHBORS

#include <QThread>

#include <memory>

using namespace mv::plugin;
using namespace mv;

using scalar_t = float;
using UMAP = umappp::Umap<scalar_t>;

class UMAPWorker : public QObject
{
    Q_OBJECT

public:
    UMAPWorker(Dataset<Points>& inputPoints, DatasetTask* parentTask, int outDim, SettingsAction* settings, KnnSettingsAction* knnSettings, AdvancedSettingsAction* advSettings);

public: // Setter
    void changeThread(QThread* targetThread);
    void setInitEmbedding(std::vector<scalar_t>&& embedding) { _embedding = std::move(embedding); }

private:
    void resetThread();

public slots:
    void compute();
    void stop() { _shouldStop = true; };

signals:
    void embeddingUpdate(const std::vector<scalar_t> embedding, int epoch);
    void finished();

private:
    UMAP                    _umap;              /** library */
    Dataset<Points>         _inputDataset;

    SettingsAction*         _settingsAction;    /** General settings */
    KnnSettingsAction*      _knnSettingsAction; /** knn settings */
    AdvancedSettingsAction* _advSettingsAction; /** Advanced settings */
    bool                    _shouldStop;
    std::unique_ptr<mv::Task> _computeTask;
    DatasetTask*            _parentTask;

    std::vector<scalar_t>   _embedding;         /** The output embedding as generated by this plugin */
    std::vector<scalar_t>   _outEmbedding;      /** Storage of current embedding */
    int                     _outDimensions;     /** The number of dimensions to reduce to */

};

/**
 * UMAP analysis plugin class
 *
 * This analysis plugin class provides acces to the UMAP implementation from
 * https://github.com/LTLA/umappp as an analysis plugin in HDPS
 *
 * UMAP is a neighborhood-based dimensionality reduction similar to t-SNE
 *
 * @authors T. Höllt, A.Vieth
 */
class UMAPAnalysisPlugin : public AnalysisPlugin
{
Q_OBJECT

public:

    UMAPAnalysisPlugin(const PluginFactory* factory);
    ~UMAPAnalysisPlugin() override;

    void init() override;

private:
    void deleteWorker();

signals:
    void startWorker();
    void stopWorker();

private:
    SettingsAction          _settingsAction;    /** General settings */
    KnnSettingsAction       _knnSettingsAction; /** knn settings */
    AdvancedSettingsAction  _advSettingsAction; /** Advanced settings */

    int                     _numPoints;         /** Numer of data points */
    int                     _outDimensions;     /** The number of dimensions to reduce to */
    Dataset<Points>         _outputPoints;
    
    QThread                 _workerThread;
    UMAPWorker*             _umapWorker;
};

/**
 * UMAP analysis plugin factory class
 */
class UMAPAnalysisPluginFactory : public AnalysisPluginFactory
{
    Q_INTERFACES(mv::plugin::AnalysisPluginFactory mv::plugin::PluginFactory)
    Q_OBJECT
    Q_PLUGIN_METADATA(IID   "studio.manivault.UMAPAnalysisPlugin"
                      FILE  "UMAPAnalysisPlugin.json")

public:

    UMAPAnalysisPluginFactory() {}
    ~UMAPAnalysisPluginFactory() override {}

    QIcon getIcon(const QColor& color = Qt::black) const override;

    /** Creates an instance of the UMAP analysis plugin */
    AnalysisPlugin* produce() override;

    /** Returns the data types that are supported by the UMAP analysis plugin */
    mv::DataTypes supportedDataTypes() const override;

    /** Enable right-click on data set to open analysis */
    PluginTriggerActions getPluginTriggerActions(const mv::Datasets& datasets) const override;
};
