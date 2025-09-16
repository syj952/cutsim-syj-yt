QT -= gui
QMAKE_CC=mpicxx
QMAKE_CXX=mpicxx
CONFIG += c++17 console
CONFIG -= app_bundle
CONFIG -= -O3
QT       += core gui xml opengl
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
QT += concurrent
TARGET = cutsim
TEMPLATE = app
DEFINES += QT_DEPRECATED_WARNINGS
CONFIG += c++17
# 定义CUDA支持

DEFINES += ENABLE_CUDA  # 启用CUDA宏
DEFINES += QT_DEPRECATED_WARNINGS
# CUDA设置
CUDA_DIR = /usr/local/cuda-12.1
INCLUDEPATH += $$CUDA_DIR/include

# 确保build目录存在
system(mkdir -p $$PWD/build)
message("项目目录: $$PWD")
message("输出目录: $$OUT_PWD")

# 明确指定CUDA编译步骤
cuda_diff_volume.target = $$PWD/build/cuda_diff_volume.o
cuda_diff_volume.depends = $$PWD/src/cutsim/cuda_diff_volume.cu
# 修改CUDA编译命令段并添加详细输出
cuda_diff_volume.commands = echo "开始编译CUDA文件..." && \
                        $$CUDA_DIR/bin/nvcc -v -c $$PWD/src/cutsim/cuda_diff_volume.cu -o $$PWD/build/cuda_diff_volume.o \
                        -I$$PWD/src -I$$PWD -I$$CUDA_DIR/include \
                        -I/usr/include/x86_64-linux-gnu/qt5 \
                        -I/usr/include/x86_64-linux-gnu/qt5/QtCore \
                        -D__CUDACC__ -DENABLE_CUDA \
                        --compiler-options "-fPIC -std=c++14" \
                        -arch=sm_75 && \
                        echo "CUDA编译完成，检查文件是否存在:" && \
                        ls -la $$PWD/build/

# 添加自定义目标
QMAKE_EXTRA_TARGETS += cuda_diff_volume

# 将CUDA编译设为构建依赖
PRE_TARGETDEPS += $$PWD/build/cuda_diff_volume.o

## 添加CUDA目标文件到链接
LIBS += $$PWD/build/cuda_diff_volume.o
LIBS += -L$$CUDA_DIR/lib64 -lcudart -lcuda


SOURCES += \
    src/app/cutsim_window.cpp \
    src/app/levelmeter.cpp \
    src/app/lex_analyzer.cpp \
    src/app/main_app.cpp \
    src/app/text_area.cpp \
    src/cutsim/bbox.cpp \
    src/cutsim/cuda_functions.cpp \
    src/cutsim/cutsim.cpp \
    src/cutsim/gldata.cpp \
    src/cutsim/glwidget.cpp \
    src/cutsim/machine.cpp \
    src/cutsim/marching_cubes.cpp \
    src/cutsim/mfem_analysis1.cpp \
    src/cutsim/octnode.cpp \
    src/cutsim/octree.cpp \
    src/cutsim/stl.cpp \
    src/cutsim/volume.cpp \
    src/g2m/canonLine.cpp \
    src/g2m/canonMotion.cpp \
    src/g2m/canonMotionless.cpp \
    src/g2m/g2m.cpp \
    src/g2m/helicalMotion.cpp \
    src/g2m/linearMotion.cpp \
    src/g2m/machineStatus.cpp \
    src/g2m/nanotimer.cpp

HEADERS += \
    src/app/version_string.hpp \
    src/app/cutsim_app.hpp \
    src/app/cutsim_def.hpp \
    src/app/cutsim_window.hpp \
    src/app/levelmeter.hpp \
    src/app/lex_analyzer.hpp \
    src/app/text_area.hpp \
    src/cutsim/bbox.hpp \
    src/cutsim/cube_wireframe.hpp \
    src/cutsim/cuda_functions.hpp \
    src/cutsim/cutsim.hpp \
    src/cutsim/facet.hpp \
    src/cutsim/gldata.hpp \
    src/cutsim/glwidget.hpp \
    src/cutsim/glvertex.hpp \
    src/cutsim/isosurface.hpp \
    src/cutsim/machine.hpp \
    src/cutsim/marching_cubes.hpp \
    src/cutsim/mfem_analysis1.hpp \
    src/cutsim/octnode.hpp \
    src/cutsim/octree.hpp \
    src/cutsim/stl.hpp \
    src/cutsim/volume.hpp \
    src/g2m/canonLine.hpp \
    src/g2m/canonMotion.hpp \
    src/g2m/canonMotionless.hpp \
    src/g2m/g2m.hpp \
    src/g2m/gplayer.hpp \
    src/g2m/helicalMotion.hpp \
    src/g2m/linearMotion.hpp \
    src/g2m/machineStatus.hpp \
    src/g2m/nanotimer.hpp \
    src/g2m/point.hpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

QMAKE_CFLAGS += -march=native -ftree-vectorize -O0 -g
QMAKE_CXXFLAGS += -march=native -ftree-vectorize -O0 -g
#QMAKE_CFLAGS += -march=native -ftree-vectorize -O3 -g
#QMAKE_CXXFLAGS += -march=native -ftree-vectorize -O3 -g

LIBS += -L/usr/lib/x86_64-linux-gnu/ -lGLU -lgomp
LIBS += -L/usr/lib/x86_64-linux-gnu/ -lQGLViewer-qt5
##"MFEM配置文件"
INCLUDEPATH+=/home/yt/syj/mfem-4.5
INCLUDEPATH+=/home/yt/syj/hypre/src/hypre/include
INCLUDEPATH+=/home/yt/syj/mpich-4.3.1
LIBS+=-L/home/yt/syj/mfem-4.5 -lmfem
LIBS+=-L/home/yt/syj/hypre/src/hypre/lib -lHYPRE
LIBS+=-L/home/yt/syj/metis-4.0 -lmetis
LIBS+=-lrt
LIBS+=-L/home/yt/syj/mpich-4.3.1 -lmpi -lmpi_cxx
QT += widgets

#LIBS *= -L$${LIB_DIR} -lQGLViewer -lGLU -lgomp
#LIB_NAME = QGLViewer-qt5
#LIBS *= -L$${LIB_DIR} -l$${LIB_NAME}

DISTFILES += \
    src/cutsim/cuda_diff_volume.cu
