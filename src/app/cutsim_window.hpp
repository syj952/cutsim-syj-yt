#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QFileDialog>
#include <QLabel>
#include <QProgressBar>
#include <QProgressDialog>
#include <QtWidgets/QAction>
#include <QtWidgets/QStatusBar>
#include <QDockWidget>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QToolBar>
#include <QtConcurrent>

#include <src/cutsim/cutsim.hpp>
#include <src/cutsim/glwidget.hpp>

#include "version_string.hpp"
#include "text_area.hpp"
#include "cutsim_app.hpp"

class QAction;
class QLabel;
class QMenu;

typedef enum {
    NO_OPERATION = 0,
    SUM_OPERATION = 1,
    DIFF_OPERATION = 2,
    INTERSECT_OPERATION = 3,
} OperationType;

class StockVolume {
public:
    StockVolume() {};
    virtual ~StockVolume() {};
    cutsim::Volume*	stock;
    int operation;
} ;

/// the main application window for the cutting-simulation
/// this includes menus, toolbars, text-areas for g-code and canon-lines and debug
/// the 3D view of tool/stock.
class CutsimWindow : public QMainWindow {
    Q_OBJECT

public:
    /// create window
    CutsimWindow(QStringList ags);
    ~CutsimWindow();

public slots:


    void requestRedraw(QString s) {
        if (s != QString("")) {
            statusBar()->showMessage(s + "...");
            CutsimApplication::processEvents();
        }
        myCutsim->updateGL();
        bool pre_status;
        if ((pre_status = myGLWidget->doAnimate()) == false)
            myGLWidget->setAnimate(true);
        myGLWidget->reDraw();
        myGLWidget->setAnimate(pre_status);
        if (s != QString(""))
            statusBar()->showMessage(s + "...done.");
    }

private:

    cutsim::Cutsim* myCutsim;

    cutsim::GLWidget* myGLWidget;

    std::vector<cutsim::CutterVolume*> myTools;

    unsigned int currentTool;
    TextArea* debugText;

    QStringList args;
    QString myLastFolder;
    QSettings settings;
    QLabel* myStatus;
    double octree_cube_size;
    unsigned int max_depth;
    cutsim::GLVertex* octree_center;
    double cube_resolution_1,cube_resolution_2;
    double specific_cutting_force;
    double powerCoff;
    double requiredPower;

    std::vector<StockVolume*> myStocks;
    QProgressDialog* waitingDialog;

    double step_size;
    bool   variable_step_mode;
};

#endif
