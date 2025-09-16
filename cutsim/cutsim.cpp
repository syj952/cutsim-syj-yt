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

#include "cutsim.hpp"
#include "volume.hpp"
#include "octnode.hpp"
namespace cutsim {

Cutsim::Cutsim (double octree_size, unsigned int octree_max_depth, GLVertex* octree_center, GLData* gld, GLWidget* wid): g(gld), widget(wid) {
    tree = new Octree(octree_size, octree_max_depth, octree_center, g);
    std::cout << "Cutsim() ctor: tree before init: " << tree->str() << "\n";
    tree->init(2u);
    tree->debug=false;
    wid->setTree(tree);
    std::cout << "Cutsim() ctor: tree after init: " << tree->str() << "\n";
#ifndef WIRE_FRAME
    iso_algo = new MarchingCubes(g, tree);
#else
    iso_algo = new CubeWireFrame(g, tree);
#endif
}

Cutsim::~Cutsim() {
    delete iso_algo;
    delete tree;
    delete g;
}

void Cutsim::updateGL() {
	std::chrono::system_clock::time_point start, stop;
    start = std::chrono::system_clock::now();
    iso_algo->updateGL();
    g->swap();
    stop = std::chrono::system_clock::now();
    qDebug() << "cutsim.cpp updateGL()    :" << std::chrono::duration<double>(stop - start).count() << "sec.";
}

//void Cutsim::sum_volume( const Volume* volume ) {
void Cutsim::sum_volume( Volume* volume ) {
	std::chrono::system_clock::time_point start, stop;
    start = std::chrono::system_clock::now();
    tree->sum( volume );
    //tree->check_node( volume );
    stop = std::chrono::system_clock::now();
    qDebug() << "cutsim.cpp sum_volume()  :" << std::chrono::duration<double>(stop - start).count() << "sec.";
}

/// added hust
void Cutsim::diff_volume_cuda( const CylCutterVolume* volume ) {
    std::chrono::system_clock::time_point start, stop;
    start = std::chrono::system_clock::now();
    tree->cuda_diff( volume );
    stop = std::chrono::system_clock::now();
    qDebug() << "cutsim.cpp diff_volume_cuda() :" << std::chrono::duration<double>(stop - start).count() << "sec.";
    //qDebug() << "\n";
    //qDebug() << "";
    //qDebug() <<std::chrono::duration<double>(stop - start).count();
}

void Cutsim::sum_volume_cuda( const Volume* volume ,double max_depth ) {
    std::chrono::system_clock::time_point start, stop;
    start = std::chrono::system_clock::now();
    tree->cuda_sum( volume ,max_depth);
    stop = std::chrono::system_clock::now();
    qDebug() << "cutsim.cpp sum_volume_cuda()  :" << std::chrono::duration<double>(stop - start).count() << "sec.";
}

void Cutsim::diff_volume_blade_cuda( const AptCutterVolume* volume ) {
    std::chrono::system_clock::time_point start, stop;
    start = std::chrono::system_clock::now();
    tree->cuda_blade_diff( volume);
    stop = std::chrono::system_clock::now();
    qDebug() << "cutsim.cpp diff_volume_blade_cuda() :" << std::chrono::duration<double>(stop - start).count() << "sec.";
}

void Cutsim::diff_volume_blade( const Volume* volume ) {
    std::chrono::system_clock::time_point start, stop;
    start = std::chrono::system_clock::now();
    tree->blade_diff( volume);
    stop = std::chrono::system_clock::now();
    qDebug() << "cutsim.cpp diff_volume() :" << std::chrono::duration<double>(stop - start).count() << "sec.";
}


void Cutsim::diff_volume( const Volume* volume ) {
	std::chrono::system_clock::time_point start, stop;
    start = std::chrono::system_clock::now();
    tree->diff( volume );
    stop = std::chrono::system_clock::now();
    qDebug() << "cutsim.cpp diff_volume() :" << std::chrono::duration<double>(stop - start).count() << "sec.";
}

///added hust
// 添加形变数据加载函数
void Cutsim::loadDeformationData(const std::string& deformedPointsFile, const std::string& displacementsFile) {
    // 通过 MarchingCubes 静态函数加载形变数据
    MarchingCubes::loadDeformationData(deformedPointsFile, displacementsFile);
}
///added hust
void Cutsim::intersect_volume( const Volume* volume ) {
	std::chrono::system_clock::time_point start, stop;
    start = std::chrono::system_clock::now();
    tree->intersect( volume );
    stop = std::chrono::system_clock::now();
    qDebug() << "cutsim.cpp intersect_volume() :" << std::chrono::duration<double>(stop - start).count() << "sec.";
}

void Cutsim::treeTransfer(GLVertex parallel, int flip_axis, bool ignore_parts) {
	tree->treeTransfer(parallel, flip_axis, ignore_parts);
}

void Cutsim::clearSim(double octree_size, unsigned int octree_max_depth, GLVertex* octree_center) {
	tree->clearTree(octree_size, octree_max_depth, octree_center);
    tree->init(2u);
}
// 添加根据力位置查找最近节点的方法
   std::vector<Octnode*> Cutsim::findNearestNodesForForces(const std::vector<AptCutterVolume::ForceData>& forceData,std::vector<Octnode*>& boundarynode)const {
       std::vector<Octnode*> nearestNodes;
       const auto& nodeList = boundarynode;

       for (const auto& force : forceData) {
           Octnode* nearestNode = nullptr;
           double minDistance = std::numeric_limits<double>::max();

           // 遍历所有节点找到最近的
           for (Octnode* node : nodeList) {
               if (node && node->center) {
                   // 计算力位置与节点中心的距离
                   double dx = force.force_position.x - node->center->x;
                   double dy = force.force_position.y - node->center->y;
                   double dz = force.force_position.z - node->center->z;
                   double distance = sqrt(dx*dx + dy*dy + dz*dz);

                   if (distance < minDistance) {
                       minDistance = distance;
                       nearestNode = node;
                   }
               }
           }

           nearestNodes.push_back(nearestNode);
       }

       return nearestNodes;
   }
} // end namespace
