/*
 *  Copyright 2010-2011 Anders Wallin (anders.e.e.wallin "at" gmail.com)
 *  Copyright 2015      Kazuyasu Hamada (k-hamada@gifu-u.ac.jp)
 *
 *  This file is part of Cutsim / OpenCAMlib.
 *
 *  OpenCAMlib is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  OpenCAMlib is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with OpenCAMlib.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "cutsim_def.hpp"

#include "cutsim_app.hpp"
#include "cutsim_window.hpp"

#include "lex_analyzer.hpp"
#include <src/cutsim/facet.hpp>
#include <src/cutsim/volume.hpp>
#include <src/cutsim/glvertex.hpp>
using namespace cutsim;
CutsimWindow::CutsimWindow(QStringList ags) : args(ags), myLastFolder(tr("")), settings("github.aewallin.cutsim","cutsim") {
    myGLWidget = new cutsim::GLWidget(DEFAULT_SCENE_RADIUS);
    cutsim::GLData* gld = myGLWidget->addGLData();
    this->setCentralWidget(myGLWidget);

    // T0 -- No tool
    cutsim::CylCutterVolume* s0 = new cutsim::CylCutterVolume();
    s0->type=cutsim::CYLINDER_VOLUME;
    s0->cuttertype = cutsim::CYLINDER;
    s0->setRadius(10);
    s0->setLength(150);
    s0->setCenter(cutsim::GLVertex(0.0, 0.0, -10.0));
    s0->setColor(0.5,0.2,0.1);
    s0->setHolderRadius(63);
    s0->setHolderLength(50);
    s0->enableHolder(false);
    s0->calcBB();
    myTools.push_back(s0);

    ///added hust
    #include <cmath>
    //T1
    cutsim::AptCutterVolume* s1 = new cutsim::AptCutterVolume();
    s1->type=cutsim::APT_VOLUME;
    s1->cuttertype = cutsim::APT;
    s1->setCenter(cutsim::GLVertex(0.0, 0.0, 10.0));
    s1->setColor(0.5,0.2,0.1);
    // 设置参数
    //径向、切向和轴向的剪切力系数,径向、切向和轴向的犁耕力系数
    s1->K_rc=2500;
    s1->K_tc=1900;
    s1->K_ac=100;
    s1->K_re=43;
    s1->K_te=24;
    s1->calcBB();
    myTools.push_back(s1);


    s1->K_ae=0;
    // hard-coded stock
    cutsim::CylCutterVolume* stock0 = new cutsim::CylCutterVolume();
    stock0->type=cutsim::CYLINDER_VOLUME;
    stock0->cuttertype = cutsim::CYLINDER;
    stock0->setColor(0.1,0.5,0.2);
    stock0->setRadius(5.0);
    stock0->setLength(3.0);
    stock0->setCenter(cutsim::GLVertex(0.0, 0.0, 0.0));
    stock0->calcBB();

    ///added hust
    cutsim::RectVolume* stock1 = new cutsim::RectVolume();
    stock1->setColor(0.1,0.5,0.2);
    stock1->setlengthX(10.0);
    stock1->setlengthY(10.0);
    stock1->setlengthZ(20.0);
    stock1->setCenter(cutsim::GLVertex(0.0, 0.0, 0.0));
    stock1->calcBB();

    // cutsim
    //octree_cube_size = DEFAULT_CUBE_SIZE / 2.0;
    octree_cube_size=20;
    //max_depth = DEFAULT_MAX_DEPTH;
    max_depth =10;
    octree_center = new cutsim::GLVertex(0.0, 0.0, 0.0);
    s1->max_depth_1=8;//sum
    s1->max_depth_2=8;//diff
    s1->cube_resolution_1 = octree_cube_size * 2.0 / pow(2.0, s1->max_depth_1 - 1);
    s1->cube_resolution_2 = octree_cube_size * 2.0 / pow(2.0, s1->max_depth_2 - 1);
    s1->inv_cube_resolution_2 = 1.0 / s1->cube_resolution_2;

    myCutsim = new cutsim::Cutsim(octree_cube_size, max_depth, octree_center, gld, myGLWidget);
    //    myCutsim->init(4);
    myCutsim->sum_volume_cuda(stock1, s1->max_depth_1);
    //myCutsim->sum_volume(stock0);
    myCutsim->updateGL();
    //step/(2*M_PI):转速,rad/单位时间
    //v_x,v_y:进给速率,mm/转
    s1->step=0.05;
    s1->v_x = 0.0;
    s1->v_y = 10.0;
    s1->v_z = 0.0;
    s1->dx = s1->v_x * s1->step;
    s1->dy = s1->v_y * s1->step;
    s1->dz = s1->v_z * s1->step;
    int blade_sum=2;//刀刃数

    s1->readTestPointsFromFile("../src/data/test_1.txt");

//    s1->setCenter(cutsim::GLVertex(-5,-5,0.0));
     std::vector<cutsim::Octnode*> nearestNodes;
//     for(double i=-1; i<2;i=i+s1->step)
//        {
//            double theata=i;
//            double x = 15.34;
//            double y = -25;
//            s1->rotation_angle=theata;
//            s1->setCenter(cutsim::GLVertex(x,y,0.0));
//            for(double n=0;n<2;n=n+1)
//            {
//                s1->blade_num=n;//刀刃序号
//                s1->setangle(theata+n*2*M_PI/blade_sum);
//                s1->calculatePosition_balde(s1->cube_resolution_2);
//                myCutsim->diff_volume_blade_cuda(s1);
//                s1->cut_h.clear();
//            }
//            qDebug() << "i:" <<i;
//     }
    //主循环函数
    for(double i=-1; i<3;i=i+s1->step)
    {
        double theata=i;
        double x = -5+s1->v_x*i/(2*M_PI);
        double y = -5+s1->v_y*i/(2*M_PI);
        s1->dx=s1->v_x*s1->step/(2*M_PI);
        s1->dy=s1->v_y*s1->step/(2*M_PI);
        //s1->rotation_angle=theata;
        s1->setCenter(cutsim::GLVertex(x,y,-9.0));
        s1->tool_angle=s1->tool_angle+s1->step;
        for(int n=0;n<blade_sum;n=n+1)
        {
            s1->blade_num=n;//刀刃序号
            s1->tool_angle=s1->tool_angle+n*2*M_PI/blade_sum;
            s1->calculatePosition_balde();
            myCutsim->diff_volume_blade_cuda(s1);
            s1->tool_angle=s1->tool_angle-n*2*M_PI/blade_sum;
            s1->calculateForceData(s1->step,s1->cube_resolution_2);
            s1->cut_h.clear();
        }
        s1->calculateTotalForce();
        qDebug() << "i:" <<i;
        std::cout<<"接触节点数量"<<myCutsim->tree->cwenodelist.size()<<std::endl;
        //        s1->calculateVibration();
    }
    //
    std::cout<<"力数据"<<s1->collected_force_data.size()<<std::endl;

    //网格导出
    std::cout << "开始导出网格文件..." << std::endl;
    // 获取悬挂顶点和普通顶点
    std::vector<GLVertex*> hanging_vertices;
    std::vector<GLVertex*> normalvertices;
    myCutsim->tree->get_hangingVertex(hanging_vertices, normalvertices);
    std::cout<<"hanging vertex"<<hanging_vertices.size()<<std::endl;
    std::cout<<"normal vertex"<<normalvertices.size()<<std::endl;

    // 获取悬挂顶点的父节点信息
    myCutsim->tree->get_hanging_vertex_parent();
    // 获取边界面
    std::vector<std::vector<int>> boundaryFaces;
    std::vector<Octnode*> boundarynode;
    std::vector<std::vector<int>> forceBoundaryFaceVertices;//力作用面
    myCutsim->tree->boundary(boundaryFaces, normalvertices,boundarynode);
    std::cout<<"边界node"<<boundarynode.size()<<std::endl;
    // 添加查找最近节点的代码
     nearestNodes = myCutsim->findNearestNodesForForces(s1->collected_force_data,boundarynode);
    std::cout<<"找到的最近节点数量："<<nearestNodes.size()<<std::endl;
    for (size_t i = 0; i < s1->collected_force_data.size() && i < nearestNodes.size(); ++i) {
        const auto& force = s1->collected_force_data[i];
        const auto* node = nearestNodes[i];

        if (node && node->center) {
            std::cout << "力" << i << " 位置:(" << force.force_position.x << ","
                      << force.force_position.y << "," << force.force_position.z
                      << ") -> 最近节点中心:(" << node->center->x << ","
                      << node->center->y << "," << node->center->z << ")" << std::endl;
        }

    }


    myCutsim->tree->getForceBoundaryFace(nearestNodes,boundaryFaces,forceBoundaryFaceVertices,normalvertices);
    // 导出网格到文件
    myCutsim->tree->export_mesh_to_file(normalvertices, boundaryFaces, hanging_vertices,forceBoundaryFaceVertices,"/home/yt/syj/cutsim-master-yt01/tasat2.mesh");
    myCutsim->updateGL();
//    s1->PlotForces(s1->cube_resolution_2);//输出图表
//    // s1->plotVibrationCurves();//输出图表
//    //    currentTool = 1;
//    //    myGLWidget->setTool(myTools[currentTool]);
//    //    myGLWidget->setAnimate(true);
//    //    myGLWidget->reDraw();
}

CutsimWindow::~CutsimWindow()
{
    delete myCutsim;
    delete myGLWidget;
}
