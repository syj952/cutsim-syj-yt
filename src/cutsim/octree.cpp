/*
 *  Copyright 2010-2011 Anders Wallin (anders.e.e.wallin "at" gmail.com)
 *  Copyright 2015      Kazuyasu Hamada (k-hamada@gifu-u.ac.jp)
 *
 *  This file is part of OpenCAMlib.
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

#include <list>
#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>

#include <boost/foreach.hpp>

#include <src/app/cutsim_def.hpp>

#include "octnode.hpp"
#include "octree.hpp"
#include "volume.hpp"
#include "marching_cubes.hpp"

#ifdef ENABLE_CUDA
#include "cuda_functions.hpp"
#endif

namespace cutsim {

//**************** Octree ********************/

Octree::Octree(double scale, unsigned int  depth, GLVertex* centerp, GLData* gl) {
    root_scale = scale;
    max_depth = depth;
    g = gl;
    // parent(=root) scale, GLdata, center
    root = new Octnode( centerp, root_scale, g );

    for (int n = 0; n < 8; ++n) {
        root->child[n] = NULL;
    }
    debug = false;
    debug_mc = false;
    for (unsigned int i = 0; i < max_depth; i++) {
        int amount = 0;
        for (int k = (max_depth - 1) - i; k >= 0; k--)
            amount += (int)pow((1 << k), 3);
        processed.push_back(amount);//存储每一层的节点数量
    }
}

Octree::~Octree() {
    delete root;
    root = 0;
}

unsigned int Octree::get_max_depth() const {
    return max_depth;
}

double Octree::get_root_scale() const {
    return root_scale;
}

double Octree::leaf_scale() const {
    return (2.0*root_scale) / pow(2.0, (int)max_depth);
}

/// subdivide the Octree n times
void Octree::init(const unsigned int n) {
    for (unsigned int m = 0; m < n; ++m) {
        std::vector<Octnode*> nodelist;
        get_leaf_nodes(root, nodelist);
        BOOST_FOREACH( Octnode* node, nodelist) {
            node->force_subdivide();
        }
    }
}

void Octree::get_invalid_leaf_nodes( std::vector<Octnode*>& nodelist) const {
    get_invalid_leaf_nodes( root, nodelist );
}

void Octree::get_invalid_leaf_nodes(Octnode* current, std::vector<Octnode*>& nodelist) const {
    if ( current->childcount == 0 ) {
        if ( !current->valid() ) {
            nodelist.push_back( current );
        }
    } else {//surface()surface()
        for (int n = 0; n < 8; ++n) {
            if ( current->hasChild(n) ) {
                if ( !current->valid() ) {
                    get_leaf_nodes( current->child[n], nodelist );
                }
            }
        }
    }
}

/// put leaf nodes into nodelist
void Octree::get_leaf_nodes(Octnode* current, std::vector<Octnode*>& nodelist) const {
    if ( current->isLeaf() ) {
        nodelist.push_back( current );
    } else {
        for (int n = 0; n < 8; ++n) {
            if ( current->child[n] != 0 )
                get_leaf_nodes( current->child[n], nodelist );
        }
    }
}

/// put all nodes into nodelist
void Octree::get_all_nodes(Octnode* current, std::vector<Octnode*>& nodelist) const {
    if ( current ) {
        nodelist.push_back( current );
        for (int n = 0; n < 8; ++n) {
            if ( current->child[n] != 0 )
                get_all_nodes( current->child[n], nodelist );
        }
    }
}

// sum (union) of tree and Volume
//void Octree::sum(Octnode* current, const Volume* vol) {
void Octree::sum(Octnode* current, Volume* vol) {
    if ( current->is_inside() || !vol->bb.overlaps( current->bb ) ) { // if no overlap, or already INSIDE, then quit.
        vol->accumlateProgress(processed[current->depth]);
        return; // abort if no overlap.
    }

    if ((current->depth == (this->max_depth-1)) && current->is_undecided() && !current->color.compareColor(vol->color)) { // return;
        vol->accumlateProgress(1);
        return;
    }

    current->sum(vol);

    if (vol->type == cutsim::STL_VOLUME) {
        if (current->depth == (this->max_depth-1)) {
            current->set_state();
            vol->accumlateProgress(1);
            return;
        } else if (current->check_complete_inside_outside()) { //return;
            vol->accumlateProgress(processed[current->depth]);
            return;
        }
    } else
        current->set_state();

    if ( (current->childcount == 8) ) { // recurse into existing tree
#ifdef MULTI_THREAD_SUM
        QFuture<void> future[8];
        bool dispatch[8];
        for(int m = 0; m < 8; ++m)
            if ( !current->child[m]->is_inside()  ) { // nodes that are already INSIDE cannot change in a sum-operation
                future[m] = QtConcurrent::run(this, &Octree::sum, current->child[m], vol);
                dispatch[m] = true;
            } else
                dispatch[m] = false;
        for(int m = 0; m < 8; ++m)
            if (dispatch[m] == true)
                future[m].waitForFinished();
#else
        for(int m = 0; m < 8; ++m) {
            if ( !current->child[m]->is_inside()  ) // nodes that are already INSIDE cannot change in a sum-operation
                sum( current->child[m], vol); // call sum on children
        }
#endif
    } else { // no children, subdivide it
        if ( (current->depth < (this->max_depth-1)) ) {
            if (!current->is_undecided()) { current->force_setUndecided(); }
            current->subdivide(); // smash into 8 sub-pieces
#ifdef MULTI_THREAD_SUM
            QFuture<void> future[8];
            for(int m = 0; m < 8; ++m)
                future[m] = QtConcurrent::run(this, &Octree::sum, current->child[m], vol);
            for(int m = 0; m < 8; ++m)
                future[m].waitForFinished();
#else
            for(int m = 0; m < 8; ++m)
                sum( current->child[m], vol); // call sum on children
#endif
        }
    }
    // now all children of current have their status set, and we can prune.
    if ( (current->childcount == 8) && ( current->all_child_state(Octnode::INSIDE) || current->all_child_state(Octnode::OUTSIDE) ) ) {
        if (current->all_child_state(Octnode::INSIDE))
            current->state = Octnode::INSIDE;
        else
            current->state = Octnode::OUTSIDE;
        current->delete_children();
    }

    vol->accumlateProgress(1 << ((this->max_depth-1) - current->depth));
    if (current->depth < (this->max_depth-3))
        vol->sendProgress();

    return;
}

/// added hust

void Octree::cuda_diff(Octnode* current, const CylCutterVolume* vol) {
    // 使用CUDA版本
    //qDebug()<<"使用CUDA版本";
    cuda_functions cuda;  // 创建cuda_functions实例
    //qDebug() << "vol:" <<vol->type;
    cuda.diff_volume(current, vol, this->max_depth);  // 通过对象调用成员函数
}

void Octree::cuda_sum(Octnode* current, const Volume* vol,double max_depth) {
    // 使用CUDA版本
    //qDebug()<<"使用CUDA版本cuda_sum";
    cuda_functions cuda;  // 创建cuda_functions实例
    //qDebug() << "vol:" <<vol->type;
    cuda.sum_volume(current, vol, max_depth);  // 通过对象调用成员函数

}

void Octree::cuda_blade_diff(Octnode* current, const AptCutterVolume* vol) {
    // 使用CUDA版本
    //qDebug()<<"使用CUDA版本";
    cuda_functions cuda;  // 创建cuda_functions实例
#ifdef BLADE_MULTI_GPU
    cuda.diff_volume_blade_multi_gpu(current, vol, this->max_depth);
#else
    cuda.diff_volume_blade(current, vol, this->max_depth,this);  // 通过对象调用成员函数
#endif
}

void Octree::blade_diff(Octnode* current, const Volume* vol) {


    if ( current->is_outside() || !vol->bb.overlaps( current->bb )) // if no overlap, or already OUTSIDE, then return.
        return;

#include <cmath>
#include <algorithm>

    auto aptvol = dynamic_cast<const AptCutterVolume*>(vol);

    for (int n = 0; n < 8; ++n)  {
        struct Point2D {
            double x, y;
            Point2D(double x_=0, double y_=0) : x(x_), y(y_) {}
        };
        GLVertex* v = current->vertex[n];
        GLVertex center = aptvol->center;
        double blade_angle = 0.0;
        double r;
        if(v->z<=aptvol->H_1+aptvol->z_start && v->z>aptvol->z_start){
            r = aptvol->r_1  * (v->z-aptvol->z_start ) / (aptvol->H_1);
            blade_angle=aptvol->get_blade_angle_1(v->z);
        }
        else if(v->z<=aptvol->H_2+aptvol->H_1+aptvol->z_start && v->z>aptvol->H_1+aptvol->z_start){
            r = aptvol->r_1 + (aptvol->r_2 - aptvol->r_1) * (v->z - aptvol->H_1-aptvol->z_start) / (aptvol->H_2);
            blade_angle=aptvol->get_blade_angle_2(v->z);
        }
        else {
            continue;
        }
        Point2D a(center.x,center.y), b(center.x-aptvol->dx,center.y-aptvol->dy), c(center.x-aptvol->dx+r*sin(fmod(blade_angle-aptvol->step, 2*M_PI)),center.y-aptvol->dy+r*cos(fmod(blade_angle-aptvol->step, 2*M_PI))),d(center.x+r*sin(fmod(blade_angle, 2*M_PI)),center.y+r*cos(fmod(blade_angle, 2*M_PI)));
        Point2D p(v->x, v->y);
        auto cross = [](const Point2D& a, const Point2D& b, const Point2D& c) {
            return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
        };
        // 判断点p是否在三角形abc内
        auto pointInTriangle = [&](const Point2D& a, const Point2D& b, const Point2D& c, const Point2D& p) {
            double c1 = cross(a, b, p);
            double c2 = cross(b, c, p);
            double c3 = cross(c, a, p);
            return (c1 >= 0 && c2 >= 0 && c3 >= 0) || (c1 <= 0 && c2 <= 0 && c3 <= 0);
        };
        // 判断点p是否在四个三角形之一内
        bool inside = pointInTriangle(a, b, c, p)
                || pointInTriangle(a, b, d, p)
                || pointInTriangle(a, c, d, p)
                || pointInTriangle(b, c, d, p);
        // 计算点到线段的距离
        auto pointToSegmentDist = [](const Point2D& p, const Point2D& a, const Point2D& b) {
            double dx = b.x - a.x;
            double dy = b.y - a.y;
            if (dx == 0 && dy == 0) return std::hypot(p.x - a.x, p.y - a.y);
            double t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx*dx + dy*dy);
            t = std::max(0.0, std::min(1.0, t));
            double proj_x = a.x + t * dx;
            double proj_y = a.y + t * dy;
            return std::hypot(p.x - proj_x, p.y - proj_y);
        };
        double d1 = pointToSegmentDist(p, a, b);
        double d2 = pointToSegmentDist(p, b, c);
        double d3 = pointToSegmentDist(p, c, d);
        double d4 = pointToSegmentDist(p, d, a);
        double d_min;
        if (inside)
            d_min=std::min({d1, d2, d3, d4});      // 内部为正
        else
            d_min=-std::min({d1, d2, d3, d4});     // 外部为负
        //current->color = vol->color;
        if (-d_min  < current->f[n]) {
            current->f[n] = -d_min;
            current->color = vol->color;
        }
    }

    current->set_state();

    if ( ((current->childcount) == 8) /*&& current->is_undecided()*/ ) { // recurse into existing tree
        for(int m = 0; m < 8; ++m) {
            if ( !current->child[m]->is_outside()  ) // nodes that are OUTSIDE don't change
                blade_diff( current->child[m], vol); // call diff on children
        }
    } else { // no children, subdivide it
        if ( (current->depth < (this->max_depth-1)) ) {
            if (!current->is_undecided()) { current->force_setUndecided(); }
            current->subdivide(); // smash into 8 sub-pieces
            for(int m = 0; m < 8; ++m) {
                blade_diff( current->child[m], vol); // call diff on children
            }
        }
    }
    // now all children have their status set, prune.
    if ( (current->childcount == 8) && ( /*current->all_child_state(Octnode::INSIDE) ||*/ current->all_child_state(Octnode::OUTSIDE) ) ) {
        current->state = Octnode::OUTSIDE;
        current->delete_children();
    }
}
///<added hust
// 在octree.cpp中实现颜色更新函数
void Octree::updateDeformationColors() {
    updateDeformationColors(cwenodelist);
}
void Octree::updateDeformationColors(const std::vector<Octnode*>& nodeList) {
    if (nodeList.empty()) {
        std::cout << "节点列表为空，无法更新颜色" << std::endl;
        return;
    }

    // 第一步：计算所有节点的形变距离范围
    float minDistance = FLT_MAX;
    float maxDistance = 0.0f;
    std::vector<float> nodeMaxDistances;

    for (Octnode* node : nodeList) {
        float nodeMaxDist = 0.0f;

        for (int n = 0; n < 8; n++) {
            if (node->vertex[n] != nullptr) {
                float dx = node->deformedVertex[n].x - node->vertex[n]->x;
                float dy = node->deformedVertex[n].y - node->vertex[n]->y;
                float dz = node->deformedVertex[n].z - node->vertex[n]->z;
                float distance = sqrt(dx*dx + dy*dy + dz*dz);

                if (distance > nodeMaxDist) {
                    nodeMaxDist = distance;
                }
            }
        }

        nodeMaxDistances.push_back(nodeMaxDist);

        if (nodeMaxDist < minDistance) minDistance = nodeMaxDist;
        if (nodeMaxDist > maxDistance) maxDistance = nodeMaxDist;
    }

    std::cout << "形变距离范围: [" << minDistance << ", " << maxDistance << "]" << std::endl;

    // 第二步：应用颜色映射
    if (maxDistance > minDistance) {
        //
        for (size_t i = 0; i < nodeList.size(); i++) {
            float normalizedDistance = (nodeMaxDistances[i] - minDistance) /
                    (maxDistance - minDistance);

            // 线性颜色映射：红色表示高形变，绿色表示低形变
            nodeList[i]->color = {
                normalizedDistance,           // 红色分量
                1.0f - normalizedDistance,    // 绿色分量
                0.0f                          // 蓝色分量
            };

            std::cout << "节点 " << i << " 形变距离: " << nodeMaxDistances[i]
                         << ", 颜色: (" << nodeList[i]->color.r << ", "
                         << nodeList[i]->color.g << ", " << nodeList[i]->color.b << ")" << std::endl;
        }
    } else {
        // 所有节点形变距离相同，使用统一颜色
        for (Octnode* node : nodeList) {
            node->color = {0.5f, 0.5f, 0.0f}; // 黄色
        }
        std::cout << "所有节点形变距离相同，使用统一颜色" << std::endl;
    }
}


void Octree::diff(Octnode* current, const Volume* vol) {
    if ( current->is_outside() || !vol->bb.overlaps( current->bb ) ) // if no overlap, or already OUTSIDE, then return.
        return;
    ///added hust
    current->DeformedVertices(MarchingCubes::originalPoints, MarchingCubes::deformedPoints);//更新该节点的形变坐标
    current->diff(vol);

    current->set_state();

    ///added hust
    if (current->currentcutstate==Octnode::CUTTING && current->state==Octnode::UNDECIDED && current->depth==max_depth-1) ///< added hust tool work engaged
    {
        engaged = true;
        //        current->currentcutstate==Octnode::CUTTING;
#pragma omp critical
        {
            cwenodelist.push_back(current);
            current->color = {1.0f, 0.0f, 0.0f};//设置为红色
        }
    }

    if ( ((current->childcount) == 8) /*&& current->is_undecided()*/ ) { // recurse into existing tree
        for(int m = 0; m < 8; ++m) {
            if ( !current->child[m]->is_outside()  ) // nodes that are OUTSIDE don't change
                diff( current->child[m], vol); // call diff on children
        }
    } else { // no children, subdivide it
        if ( (current->depth < (this->max_depth-1)) ) {
            if (!current->is_undecided()) { current->force_setUndecided(); }
            current->subdivide(); // smash into 8 sub-pieces
            for(int m = 0; m < 8; ++m) {
                diff( current->child[m], vol); // call diff on children
            }
        }
    }
    // now all children have their status set, prune.
    if ( (current->childcount == 8) && ( /*current->all_child_state(Octnode::INSIDE) ||*/ current->all_child_state(Octnode::OUTSIDE) ) ) {
        current->state = Octnode::OUTSIDE;
        current->delete_children();
    }
}

// intersect (intersection) of tree and Volume
void Octree::intersect(Octnode* current, const Volume* vol) {
    if ( current->is_outside() ) // if already OUTSIDE, then return.
        return;

    current->intersect(vol);

    current->set_state();

    if ( ((current->childcount) == 8) /*&& current->is_undecided()*/ ) { // recurse into existing tree
        for(int m = 0; m < 8; ++m) {
            //if ( !current->child[m]->is_outside()  ) // nodes that are OUTSIDE don't change
            intersect( current->child[m], vol); // call intersect on children
        }
    } else { // no children, subdivide it
        if (!current->is_undecided()) { current->force_setUndecided(); }
        if ( (current->depth < (this->max_depth-1)) ) {
            current->subdivide(); // smash into 8 sub-pieces
            for(int m = 0; m < 8; ++m) {
                intersect( current->child[m], vol); // call intersect on children
            }
        }
    }
    // now all children have their status set, prune.
    if ( (current->childcount == 8) && ( current->all_child_state(Octnode::INSIDE) || current->all_child_state(Octnode::OUTSIDE) ) ) {
        if (current->all_child_state(Octnode::INSIDE))
            current->state = Octnode::INSIDE;
        else
            current->state = Octnode::OUTSIDE;
        current->delete_children();
    }
}

// diff (intersection with volume's compliment) of tree and Volume for cuttings
CuttingStatus Octree::diff_c(Octnode* current, const Volume* vol) {
    CuttingStatus status = { 0, NO_COLLISION }, childstatus;
    if ( current->is_outside() || (!vol->bb.overlaps( current->bb ) && (!((CutterVolume*)vol)->enableholder || !((CutterVolume*)vol)->bbHolder.overlaps( current->bb ))) )
        return status;

    if (current->depth == (this->max_depth-1)) {
        status = current->diff_cd(vol);
        if (current->set_state() == false) {
            status.cutcount = 0;
            status.collision = NO_COLLISION;
        }
        return status;
    } else {
        current->diff(vol);
        current->set_state();
    }

    if ( ((current->childcount) == 8) /*&& current->is_undecided()*/ ) { // recurse into existing tree
#ifdef MULTI_THREAD_DIFF
        QFuture<CuttingStatus> future[8];
        bool dispatch[8];
        for(int m = 0; m < 8; ++m) {
            if ( !current->child[m]->is_outside() ) {
                future[m] = QtConcurrent::run(this, &Octree::diff_c, current->child[m], vol);
                dispatch[m] = true;
            } else
                dispatch[m] = false;
        }
        for(int m = 0; m < 8; ++m) {
            if (dispatch[m] == true) {
                future[m].waitForFinished();
                childstatus = future[m].result();
                status.cutcount += childstatus.cutcount;
                status.collision |= childstatus.collision;
            }
        }
#else
        for(int m = 0; m < 8; ++m) {
            if ( !current->child[m]->is_outside() ) { // nodes that are OUTSIDE don't change
                childstatus = diff_c( current->child[m], vol); // call diff_c on children
                status.cutcount += childstatus.cutcount;
                status.collision |= childstatus.collision;
            }
        }
#endif
    } else { // no children, subdivide it
        if ( (current->depth < (this->max_depth-1)) ) {
            if (!current->is_undecided()) { current->force_setUndecided(); }
            current->subdivide(); // smash into 8 sub-pieces
#ifdef MULTI_THREAD_DIFF
            QFuture<CuttingStatus> future[8];
            for(int m = 0; m < 8; ++m)
                future[m] = QtConcurrent::run(this, &Octree::diff_c, current->child[m], vol);
            for(int m = 0; m < 8; ++m) {
                future[m].waitForFinished();
                childstatus = future[m].result();
                status.cutcount += childstatus.cutcount;
                status.collision |= childstatus.collision;
            }
#else
            for(int m = 0; m < 8; ++m) {
                childstatus = diff_c( current->child[m], vol); // call diff_c on children
                status.cutcount += childstatus.cutcount;
                status.collision |= childstatus.collision;
            }
#endif
        }
    }
    // now all children have their status set, prune.
    if ( (current->childcount == 8) && ( /*current->all_child_state(Octnode::INSIDE) ||*/ current->all_child_state(Octnode::OUTSIDE) ) ) {
        current->state = Octnode::OUTSIDE;
        current->delete_children();
    }
    return status;
}

bool Octree::check_node(Octnode* current, const Volume* vol)
{
    if (vol->type != cutsim::STL_VOLUME)
        return true;

    if ((current->depth == (this->max_depth-1)) && current->is_undecided()) {
        return current->check_f_value_with_limit();
    }

    if ( ((current->childcount) == 8) && current->is_undecided() ) { // recurse into existing tree
        int state = current->check_node_state();
        if (state == Octnode::UNDECIDED) state = current->parent->check_node_state();
        if (state == Octnode::UNDECIDED) {
            bool outside = true;
            bool inside  = true;
            for(int m = 0; m < 8; ++m) {
                if (current->child[m]->is_inside()) outside = false;
                if (current->child[m]->is_outside()) inside = false;
            }
            state = inside ? Octnode::INSIDE : outside ? Octnode::OUTSIDE : Octnode::UNDECIDED;
        }

        for(int m = 0; m < 8; ++m) {
            bool check = check_node(current->child[m], vol); // call check_node on children
            if (check == false) {
                if (state == Octnode::OUTSIDE) {
                    std::cout << "Force Outside x:" << current->child[m]->center->x << " y:" << current->child[m]->center->y << " z:" << current->child[m]->center->z << "\n";
                    current->child[m]->color.set(1,1,1);
                    current->child[m]->state = Octnode::OUTSIDE;
                } else if (state == Octnode::INSIDE) {
                    std::cout << "Force Inside  x:" << current->child[m]->center->x << " y:" << current->child[m]->center->y << " z:" << current->child[m]->center->z << "\n";
                    current->child[m]->color.set(1,0,0);
                    current->child[m]->state = Octnode::INSIDE;
                } else
                    std::cout << "Can't Decide  x:" << current->child[m]->center->x << " y:" << current->child[m]->center->y << " z:" << current->child[m]->center->z << "\n";
            }
        }
    }

    // now all children have their status set, prune.
    if ( (current->childcount == 8) && (current->check_include_undecided() == false) ) {
        if (current->all_child_state(Octnode::INSIDE))
            current->state = Octnode::INSIDE;
        else if (current->all_child_state(Octnode::OUTSIDE))
            current->state = Octnode::OUTSIDE;
        else {
            int inside = 0, outside = 0;
            for(int m = 0; m < 8; ++m) {
                if (current->child[m]->is_inside())  outside++;
                if (current->child[m]->is_outside()) inside--;
            }
            std::cout << "INSIDE mix OUTSIDE  x:" << current->center->x << " y:" << current->center->y << " z:" << current->center->z << " inside " << inside << " outside " << outside << "\n";
            goto skip_delete_children;
        }
        current->delete_children();
    }
skip_delete_children:
    return true;
}

#ifdef POOL_NODE
extern std::vector<Octnode*> nodePool;
#endif

// string repr
std::string Octree::str() const {
    std::ostringstream o;
    o << " Octree: ";
    std::vector<Octnode*> nodelist;
    Octree::get_all_nodes(root, nodelist);
    std::vector<int> nodelevel(this->max_depth);
    std::vector<int> invalidsAtLevel(this->max_depth);
    std::vector<int> surfaceAtLevel(this->max_depth);
    int totalVertexSize = 0;
    BOOST_FOREACH( Octnode* n, nodelist) {
        ++nodelevel[n->depth];
        if ( !n->valid() )
            ++invalidsAtLevel[n->depth];
        if (n->is_undecided() )
            ++surfaceAtLevel[n->depth];
        for (int i = 0; i < 8; i++)
            if (n->vertex[i] != 0)
                totalVertexSize += sizeof(GLVertex);
    }
    o << "  " << nodelist.size() << " leaf-nodes:\n";
    int m = 0;
    BOOST_FOREACH( int count, nodelevel) {
        o << "depth="<<m <<"  " << count << " nodes, " << invalidsAtLevel[m] << " invalid, surface=" << surfaceAtLevel[m] << " \n";
        ++m;
    }
    o << "  total " << nodelist.size() * sizeof (Octnode) << " bytes comsumed for node(" << sizeof(Octnode) << " bytes). \n";
    extern unsigned int alocation_count;
    extern unsigned int delete_count;
    extern unsigned int delete_childlen_count;
    o << "    alocation count: " << alocation_count << "  delete count: " << delete_count << "  difference: " << alocation_count - delete_count << "\n";
    o << "    delete child count: " << delete_childlen_count << "\n";
    o << "  total " << totalVertexSize << " bytes comsumed for vertex(" << sizeof(GLVertex) << " bytes). \n";
#ifdef POOL_NODE
    o << "  Node Pool size " << nodePool.size() << "\n";
#endif
    return o.str();
}

void Octree::treeTransfer(Octnode* current, GLVertex parallel, int flip_axis, bool ignore_parts) {
    current->nodeTransfer(parallel, flip_axis, ignore_parts);
    if (((current->childcount) == 8))
        for(int m = 0; m < 8; ++m)
            treeTransfer(current->child[m], parallel, flip_axis, ignore_parts);
}

void Octree::setInvalid(Octnode* current) {
    if (current->is_undecided() && current->isLeaf() && current->valid())
        current->setInvalid();
    if (((current->childcount) == 8))
        for(int m = 0; m < 8; ++m)
            setInvalid(current->child[m]);
}


void Octree::clearTree(double root_scale, unsigned int max_depth, GLVertex* centerPoint) {
    clearNode(root);
    root->scale = root_scale;
    root->depth = 0;

    root->center = centerPoint;
    root->state = Octnode::UNDECIDED;
    root->prev_state = Octnode::OUTSIDE;

    const GLVertex direction[8] = {
        GLVertex( 1, 1,-1),   // 0
        GLVertex(-1, 1,-1),   // 1
        GLVertex(-1,-1,-1),   // 2
        GLVertex( 1,-1,-1),   // 3
        GLVertex( 1, 1, 1),   // 4
        GLVertex(-1, 1, 1),   // 5
        GLVertex(-1,-1, 1),   // 6
        GLVertex( 1,-1, 1)    // 7
    };

    for ( int n = 0; n < 8; ++n) {
        root->vertex[n] = new GLVertex(*root->center + direction[n] * root->scale) ;
        root->f[n] = -1;
    }
    root->bb.clear();
#ifdef MULTI_AXIS
    // Multi Axis
    root->bb.addPoint(*root->center + GLVertex(-2.0,-2.0,-2.0) * root->scale); // caluclate the minimum x,y,z coordinates
    root->bb.addPoint(*root->center + GLVertex( 2.0, 2.0, 2.0) * root->scale); // caluclate the maximum x,y,z coordinates
#else
    root->bb.addPoint( *vertex[2] ); // vertex[2] has the minimum x,y,z coordinates
    root->bb.addPoint( *vertex[4] ); // vertex[4] has the max x,y,z
#endif
    root->childcount = 0;
}

void Octree::clearNode(Octnode* current) {
    if (((current->childcount) == 8))
        for(int m = 0; m < 8; ++m)
            clearNode(current->child[m]);
    current->force_delete_children();
}
///added hust
void Octree::get_leaf_nodes2(Octnode* current, std::vector<Octnode*>& nodelist) const {
    if ( current->isLeaf()&&!current->is_outside() ) {
        nodelist.push_back( current );
    } else {
        for (int n = 0; n < 8; ++n) {
            if ( current->child[n] != 0 )
                get_leaf_nodes2( current->child[n], nodelist );
        }
    }
    // 在递归结束后，对最顶层调用进行排序
    if (current == root) {
        // 按照节点深度从小到大排序，如果深度相同则按 idx 排序
        std::sort(nodelist.begin(), nodelist.end(),
                  [](const Octnode* a, const Octnode* b) {
            // 首先按 depth 排序，如果 depth 相同，则按 idx 排序
            if (a->depth == b->depth) {
                return a->idx < b->idx; // 若 depth 相同，则按 idx 排序
            }
            return a->depth < b->depth; // 否则按 depth 排序
        });
    }
}

int Octree::calculate_index(int remx, int remy, int remz) {
    int index;
    if (remx == 0 && remy == 0 && remz == 0) {
        index = 1;
    }
    else if (remx == 0 && remy == 0 && remz == 1) {
        index = 5;
    }
    else if (remx == 0 && remy == 1 && remz == 1) {
        index = 6;
    }
    else if (remx == 1 && remy == 1 && remz == 1) {
        index = 7;
    }
    else if (remx == 1 && remy == 0 && remz == 1) {
        index = 4;
    }
    else if (remx == 1 && remy == 0 && remz == 0) {
        index = 0;
    }
    else if (remx == 1 && remy == 1 && remz == 0) {
        index = 3;
    }
    else if (remx == 0 && remy == 1 && remz == 0) {
        index = 2;
    }
    return index;
}


// 查找父节点
std::vector<std::vector<int>> Octree::find_parent_nodes(int deep, int idxx, int idxy, int idxz) {
    std::vector<std::vector<int>> parent_data;
    while (deep > 0) {
        int original_idxx = idxx;
        int original_idxy = idxy;
        int original_idxz = idxz;
        //计算父节点的坐标和索引
        if (idxx < 0) {//判断小于0的边界
            idxx -= 1;
        }
        if (idxy < 0) {
            idxy -= 1;
        }
        if (idxz < 0) {
            idxz -= 1;
        }
        int quotx = idxx / 2;
        int quoty = idxy / 2;
        int quotz = idxz / 2;

        int remx = idxx % 2;
        int remy = idxy % 2;
        int remz = idxz % 2;

        int index = calculate_index(remx, remy, remz);  // 计算该节点的索引
        // 将当前节点的深度、坐标和索引存储到 parent_data 中
        parent_data.push_back({ deep,original_idxx, original_idxy ,original_idxz, index });//包含被查找的节点本身直到deep=1的节点


        // 向上移动到父节点
        idxx = quotx;
        idxy = quoty;
        idxz = quotz;
        deep--;  // 深度减1
    }
    return parent_data;
}
Octnode* Octree::find_target_node(Octnode* root, int deep, int idxx, int idxy, int idxz)
{


    // 从root节点开始，逐层查找目标节点
    Octnode* current_node = root;  // 从根节点开始
    int target_deep=deep;
    int indexx=idxx;
    int indexy=idxy;
    int indexz=idxz;
    int index;
    std::vector<std::vector<int>> parent_data= find_parent_nodes( target_deep,  indexx, indexy,indexz);
    if (indexx > (1 << target_deep) -1|| indexy > (1 << target_deep)-1 || indexz >(1 << target_deep)-1||
            indexx < 0 || indexy < 0 || indexz < 0)//这个条件判断是用来处理边界情况的，也就是在查找root节点之外的节点时返回root
    {
        return root;
    }
    for (int i = parent_data.size() - 1; i >= 0; --i)// 遍历 parent_data 中存储的每一层的父节点信息
    {

        target_deep = parent_data[i][0];
        indexx = parent_data[i][1];
        indexy = parent_data[i][2];
        indexz= parent_data[i][3];
        index = parent_data[i][4];
        if (!current_node->child[index])//如果删除了是不是空指针
            return root;

        // 根据子节点的 index 获取子节点
        current_node = current_node->child[index];
    }
    return current_node;
}

bool Octree::exist(Octnode* root, int deep, int idxx, int idxy, int idxz)
{
    return find_target_node(root, deep, idxx, idxy, idxz) != root;
}
void Octree::getForceBoundaryFace(const std::vector<Octnode*>& nearestNodes, const std::vector<std::vector<int>>& boundaryFaces, std::vector<std::vector<int>>& forceBoundaryFaceVertices,std::vector<GLVertex*>& normalvertices)
{
    const int faces[6][4] = {
        {0, 1, 2, 3}, // 底面 (z=0)
        {4, 5, 6, 7}, // 顶面 (z=1)
        {0, 1, 5, 4}, // 前面 (y=0)
        {2, 3, 7, 6}, // 后面 (y=1)
        {0, 3, 7, 4}, // 左面 (x=0)
        {1, 2, 6, 5}  // 右面 (x=1)
    };

    // 面名称
    const std::string faceNames[6] = {
        "底面", "顶面", "前面", "后面", "左面", "右面"
    };

    // 清空之前的结果
       forceBoundaryFaceVertices.clear();
       forceBoundaryFaceVertices.reserve(nearestNodes.size());
       for (size_t nodeIdx = 0; nodeIdx < nearestNodes.size(); nodeIdx++) {
              Octnode* node = nearestNodes[nodeIdx];
              bool foundBoundaryFace = false;

              if (!node) {
                  std::cout << "Node " << nodeIdx << " 为空指针" << std::endl;
                  forceBoundaryFaceVertices.push_back(std::vector<int>());
                  continue;
              }

              // 遍历6个面，找到第一个边界面就停止
              for (int faceIdx = 0; faceIdx < 6 && !foundBoundaryFace; faceIdx++) {
                  std::vector<int> currentFaceVertexIds;


                  // 获取当前面的4个顶点ID
                  for (int j = 0; j < 4; j++) {
                      int vertexIdx = faces[faceIdx][j];
                       if(node->vertex[vertexIdx]->id<=0)
                          currentFaceVertexIds.push_back(-node->vertex[vertexIdx]->id);

                       else
                          currentFaceVertexIds.push_back(node->vertex[vertexIdx]->id+normalvertices.size()-1);

                  }

                  // 如果面有效且包含4个顶点，检查是否为边界面
                  if (currentFaceVertexIds.size() == 4) {
                      // 检查这4个顶点是否构成boundaryFaces中的某个面
                      for (const auto& boundaryFace : boundaryFaces) {
                          if (boundaryFace.size() == 4) {
                              // 使用std::is_permutation检查两个面是否包含相同的顶点（顺序可能不同）
                              if (std::is_permutation(currentFaceVertexIds.begin(), currentFaceVertexIds.end(), boundaryFace.begin(),boundaryFace.end()))
                               {forceBoundaryFaceVertices.push_back(currentFaceVertexIds);
                                std::cout << "Node " << nodeIdx << " 的 " << faceNames[faceIdx]<< " 是外表面" << std::endl;
                                foundBoundaryFace = true;
                                break;
                              }
                          }
                      }
                  }
                }
              // 如果没有找到边界面，添加空vector占位
                      if (!foundBoundaryFace) {
                          forceBoundaryFaceVertices.push_back(std::vector<int>());
                          std::cout << "Node " << nodeIdx << " 没有找到边界表面" << std::endl;
                      }
                  }
}






void Octree::boundary(std::vector<std::vector<int>>& boundaryFaces,std::vector<GLVertex*>& normalVertexList,std::vector<Octnode*>& boundarynode)
{
    std::vector<Octnode*> leaf_nodes;
    get_leaf_nodes2(root, leaf_nodes);

    BOOST_FOREACH(Octnode* node, leaf_nodes)
    {
        int index = node->idx;
        int deep = node->depth;
        int idxx = node->indexs->x;
        int idxy = node->indexs->y;
        int idxz = node->indexs->z;
        bool hasboundaryface=false;
        for (int i = 0; i < 6; i++)
        {
            switch(i)
            {
            case 0:
            {
                std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep, idxx, idxy-1, idxz);
                bool is_boundary = false;
                for (int j = 0; j < parents_data1.size(); j++) {
                    if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                        is_boundary = true; // 周围存在比他大的outside节点，是边界
                        break;
                    }
                }
                if (is_boundary) {
                    std::vector<int> vertex;
                    hasboundaryface=true;
                    // 添加顶点0的ID
                    if(node->vertex[0]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[0]->id);
                    else
                        vertex.push_back(node->vertex[0]->id+normalVertexList.size() -1);

                    // 添加顶点1的ID
                    if(node->vertex[1]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[1]->id);
                    else
                        vertex.push_back(node->vertex[1]->id + normalVertexList.size()  -1);

                    // 添加顶点4的ID
                    if(node->vertex[5]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[5]->id);
                    else
                        vertex.push_back(node->vertex[5]->id + normalVertexList.size()  -1);

                    // 添加顶点5的ID
                    if(node->vertex[4]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[4]->id);
                    else
                        vertex.push_back(node->vertex[4]->id + normalVertexList.size()  -1);

                    boundaryFaces.push_back(vertex);
                }

                break;
            }


            case 1:
            {
                std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep, idxx+1, idxy, idxz);
                bool is_boundary = false;
                for (int j = 0; j < parents_data1.size(); j++) {
                    if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                        is_boundary = true; // 存在父节点，不是边界
                        break;
                    }
                }
                if (is_boundary) {
                    std::vector<int> vertex;
                     hasboundaryface=true;
                    // 添加顶点0的ID
                    if(node->vertex[0]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[0]->id);
                    else
                        vertex.push_back(node->vertex[0]->id + normalVertexList.size()  -1);

                    // 添加顶点3的ID
                    if(node->vertex[3]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[3]->id);
                    else
                        vertex.push_back(node->vertex[3]->id + normalVertexList.size()  -1);

                    // 添加顶点7的ID
                    if(node->vertex[7]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[7]->id);
                    else
                        vertex.push_back(node->vertex[7]->id + normalVertexList.size() -1);

                    // 添加顶点4的ID
                    if(node->vertex[4]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[4]->id);
                    else
                        vertex.push_back(node->vertex[4]->id + normalVertexList.size()  -1);
                    std::cout<<"我是右边边界"<<node->indexs->x<<std::endl;
                    boundaryFaces.push_back(vertex);
                }

                break;
            }
            case 2:
            {
                std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep, idxx, idxy+1, idxz);
                bool is_boundary = false;
                for (int j = 0; j < parents_data1.size(); j++) {
                    if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                        is_boundary = true; // 存在父节点，不是边界
                        break;
                    }
                }
                if (is_boundary) {
                    std::vector<int> vertex;
                     hasboundaryface=true;
                    // 添加顶点2的ID
                    if(node->vertex[2]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[2]->id);
                    else
                        vertex.push_back(node->vertex[2]->id + normalVertexList.size()  -1);

                    // 添加顶点3的ID
                    if(node->vertex[3]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[3]->id);
                    else
                        vertex.push_back(node->vertex[3]->id + normalVertexList.size()  -1);

                    // 添加顶点7的ID
                    if(node->vertex[7]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[7]->id);
                    else
                        vertex.push_back(node->vertex[7]->id + normalVertexList.size()  -1);

                    // 添加顶点6的ID
                    if(node->vertex[6]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[6]->id);
                    else
                        vertex.push_back(node->vertex[6]->id + normalVertexList.size()  -1);

                    boundaryFaces.push_back(vertex);
                }

                break;
            }
            case 3:
            {
                std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep, idxx-1, idxy, idxz);
                bool is_boundary = false;
                for (int j = 0; j < parents_data1.size(); j++) {
                    if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                        is_boundary = true; // 存在父节点，不是边界
                        break;
                    }
                }
                if (is_boundary) {
                    std::vector<int> vertex;
                     hasboundaryface=true;
                    // 添加顶点1的ID
                    if(node->vertex[1]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[1]->id);
                    else
                        vertex.push_back(node->vertex[1]->id + normalVertexList.size()  -1);

                    // 添加顶点2的ID
                    if(node->vertex[2]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[2]->id);
                    else
                        vertex.push_back(node->vertex[2]->id + normalVertexList.size()  -1);

                    // 添加顶点6的ID
                    if(node->vertex[6]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[6]->id);
                    else
                        vertex.push_back(node->vertex[6]->id + normalVertexList.size()  -1);

                    // 添加顶点5的ID
                    if(node->vertex[5]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[5]->id);
                    else
                        vertex.push_back(node->vertex[5]->id + normalVertexList.size()  -1);
                    //std::cout<<"我是左边边界"<<node->indexs->x<<std::endl;
                    boundaryFaces.push_back(vertex);
                }

                break;
            }
            case 4:
            {
                std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep, idxx, idxy, idxz+1);
                bool is_boundary = false;
                for (int j = 0; j < parents_data1.size(); j++) {
                    if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                        is_boundary = true; // 存在父节点，不是边界
                        break;
                    }
                }
                if (is_boundary) {
                    std::vector<int> vertex;
                     hasboundaryface=true;
                    // 添加顶点4的ID
                    if(node->vertex[4]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[4]->id);
                    else
                        vertex.push_back(node->vertex[4]->id + normalVertexList.size()  -1);

                    // 添加顶点5的ID
                    if(node->vertex[5]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[5]->id);
                    else
                        vertex.push_back(node->vertex[5]->id + normalVertexList.size()  -1);

                    // 添加顶点6的ID
                    if(node->vertex[6]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[6]->id);
                    else
                        vertex.push_back(node->vertex[6]->id + normalVertexList.size()  -1);

                    // 添加顶点7的ID
                    if(node->vertex[7]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[7]->id);
                    else
                        vertex.push_back(node->vertex[7]->id + normalVertexList.size()  -1);

                    boundaryFaces.push_back(vertex);

                }

                break;
            }
            case 5:
            {
                std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep, idxx, idxy, idxz-1);
                bool is_boundary = false;
                for (int j = 0; j < parents_data1.size(); j++) {
                    if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                        is_boundary = true; // 存在父节点，不是边界
                        break;
                    }
                }
                if (is_boundary) {
                    std::vector<int> vertex;
                     hasboundaryface=true;
                    // 添加顶点0的ID
                    if(node->vertex[0]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[0]->id);
                    else
                        vertex.push_back(node->vertex[0]->id + normalVertexList.size()  -1);

                    // 添加顶点1的ID
                    if(node->vertex[1]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[1]->id);
                    else
                        vertex.push_back(node->vertex[1]->id + normalVertexList.size()  -1);

                    // 添加顶点2的ID
                    if(node->vertex[2]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[2]->id);
                    else
                        vertex.push_back(node->vertex[2]->id + normalVertexList.size()  -1);

                    // 添加顶点3的ID
                    if(node->vertex[3]->id <= 0) // 正常顶点
                        vertex.push_back(-node->vertex[3]->id);
                    else
                        vertex.push_back(node->vertex[3]->id + normalVertexList.size()  -1);

                    boundaryFaces.push_back(vertex);
                    std::cout<<"我是底面"<<node->indexs->z<<std::endl;
                }

                break;
            }



                break;
            }
        }
        if(hasboundaryface)
        {
            boundarynode.push_back(node);
        }
    }
}
bool Octree::check_hanging_vertex(Octnode* root, int deep, int idxx, int idxy, int idxz, int index, int i)
{

    switch (index)
    {case 0:
        switch (i)
        {   case 0://角点悬挂节点判断
        {
            Octnode* parent=find_target_node(root,deep-1,idxx/2,idxy/2,idxz/2);
            if(check_hanging_vertex(root,parent->depth,parent->indexs->x,parent->indexs->y,parent->indexs->z,parent->idx,i))
            {return true;}
            return false;
            break;
        }
        case 1: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2 - 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 - 1);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {

                    return true;
                }
            }

            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }

            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {

                    return true;
                }
            }

            return false;
            break;
        }

        case 2: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 - 1);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 3: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2 - 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 - 1);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }

            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }

            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }

            return false;
            break;
        }

        case 4: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2 - 1, idxz / 2);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }

            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }

            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }

            return false;
            break;
        }

        case 5: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        case 6:
        {
            return false;
            break;
        }
        case 7: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        }


    case 1:
        switch(i)
        {
        case 0: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2 - 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 - 1);
            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        case 1:
        {
            Octnode* parent=find_target_node(root,deep-1,idxx/2,idxy/2,idxz/2);
            if(check_hanging_vertex(root,parent->depth,parent->indexs->x,parent->indexs->y,parent->indexs->z,parent->idx,i))
            {return true;}
            return false;
            break;


        }
        case 2: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2 - 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 - 1);
            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 3: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 - 1);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 4: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 5: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2-1, idxy / 2 - 1, idxz / 2 );
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2-1, idxy / 2, idxz / 2 );
            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 6: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        case 7:
        {
            return false;
            break;
        }

        }
    case 2:
        switch(i)
        {
        case 0: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 - 1);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 1: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2 - 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 - 1);
            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        case 2:
        {Octnode* parent=find_target_node(root,deep-1,idxx/2,idxy/2,idxz/2);
            if(check_hanging_vertex(root,parent->depth,parent->indexs->x,parent->indexs->y,parent->indexs->z,parent->idx,i))
            {return true;}
            return false;
            break;}

        case 3: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2 - 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 - 1);
            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        case 4:
        {
            return false;
            break;
        }
        case 5: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 6: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2 + 1, idxz / 2);
            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 7: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        }
    case 3:
        switch(i)
        {
        case 0: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2 - 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 - 1);
            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 1: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 - 1);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 2: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2 - 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 - 1);
            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 3:
        {Octnode* parent=find_target_node(root,deep-1,idxx/2,idxy/2,idxz/2);
            if(check_hanging_vertex(root,parent->depth,parent->indexs->x,parent->indexs->y,parent->indexs->z,parent->idx,i))
            {return true;}
            return false;
            break;}

        case 4: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        case 5:
        { return false;
            break;}
        case 6: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 7: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2 + 1, idxz / 2);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2);
            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        }
    case 4:
        switch(i)
        {
        case 0: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2 - 1, idxz / 2);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 1: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        case 2:
        { return false;
            break;}
        case 3: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 4:
        {Octnode* parent=find_target_node(root,deep-1,idxx/2,idxy/2,idxz/2);
            if(check_hanging_vertex(root,parent->depth,parent->indexs->x,parent->indexs->y,parent->indexs->z,parent->idx,i))
            {return true;}
            return false;
            break;}

        case 5: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2 + 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 + 1);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 6: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 + 1);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 7: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 + 1);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2 + 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        }

    case 5:
        switch(i)
        {
        case 0: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 1: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2-1, idxy / 2 - 1, idxz / 2 );
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2-1, idxy / 2, idxz / 2 );

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 2: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        case 3:
        { return false;
            break;}
        case 4: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 - 1, idxz / 2 + 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 + 1);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 5:
        {Octnode* parent=find_target_node(root,deep-1,idxx/2,idxy/2,idxz/2);
            if(check_hanging_vertex(root,parent->depth,parent->indexs->x,parent->indexs->y,parent->indexs->z,parent->idx,i))
            {return true;}
            return false;
            break;}

        case 6: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2 + 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 + 1);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 7: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 + 1);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        }
    case 6:
        switch(i)
        {
        case 0:
        { return false;
            break;}
        case 1: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 2: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2 + 1, idxz / 2);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 3: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 4: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 + 1);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 5: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2 - 1, idxy / 2, idxz / 2 + 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 + 1);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 6:
        {
            Octnode* parent=find_target_node(root,deep-1,idxx/2,idxy/2,idxz/2);
            if(check_hanging_vertex(root,parent->depth,parent->indexs->x,parent->indexs->y,parent->indexs->z,parent->idx,i))
            {return true;}
            return false;
            break;}
        case 7: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 + 1);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2 + 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        }
    case 7:
        switch(i){

        case 0: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        case 1:
        { return false;
            break;}
        case 2: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 3: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2 + 1, idxz / 2);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 4: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 + 1);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2 + 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2 + 1, idxy / 2, idxz / 2);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root,parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 5: {
            std::vector<std::vector<int>> parents_data = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 + 1);
            for (int j = 0; j < parents_data.size(); j++) {
                if (exist(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3]) &&
                        find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->isLeaf()&&!find_target_node(root, parents_data[j][0], parents_data[j][1], parents_data[j][2], parents_data[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }

        case 6: {
            std::vector<std::vector<int>> parents_data1 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2, idxz / 2 + 1);
            std::vector<std::vector<int>> parents_data2 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2 + 1);
            std::vector<std::vector<int>> parents_data3 = find_parent_nodes(deep - 1, idxx / 2, idxy / 2 + 1, idxz / 2);

            for (int j = 0; j < parents_data1.size(); j++) {
                if (find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->isLeaf()&&!find_target_node(root, parents_data1[j][0], parents_data1[j][1], parents_data1[j][2], parents_data1[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data2.size(); j++) {
                if (find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->isLeaf()&&!find_target_node(root, parents_data2[j][0], parents_data2[j][1], parents_data2[j][2], parents_data2[j][3])->is_outside()) {
                    return true;
                }
            }
            for (int j = 0; j < parents_data3.size(); j++) {
                if (find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->isLeaf()&&!find_target_node(root, parents_data3[j][0], parents_data3[j][1], parents_data3[j][2], parents_data3[j][3])->is_outside()) {
                    return true;
                }
            }
            return false;
            break;
        }
        case 7:
        {Octnode* parent=find_target_node(root,deep-1,idxx/2,idxy/2,idxz/2);
            if(check_hanging_vertex(root,parent->depth,parent->indexs->x,parent->indexs->y,parent->indexs->z,parent->idx,i))
            {return true;}
            return false;
            break;
        }
        }

    }

}

void Octree::transfer_vertex_id_upward(Octnode* node, Octnode* root, int vertex_id, int i) {
    // 如果是根节点或者深度为0，直接返回
    if (node == root || node->depth == 0) {
        return;
    }

    // 获取当前节点的信息
    int deep = node->depth;
    int idxx = node->indexs->x;
    int idxy = node->indexs->y;
    int idxz = node->indexs->z;
    int idx = node->idx;

    // 判断当前节点的索引idx是否等于顶点索引i
    // 如果相等，说明该顶点与父节点共享，需要向上传递
    if (idx == i) {
        // 获取父节点
        Octnode* parent = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2);
        if (!parent) {
            std::cerr << "警告：找不到父节点" << std::endl;
            return;
        }

        // 更新父节点对应顶点的ID
        parent->vertex[i]->id = vertex_id;
        parent->vertexnotsaved[i] = 0;//传递ID并且取消有效值

        // 递归处理父节点
        transfer_vertex_id_upward(parent, root, vertex_id, i);
    }
    // 如果不相等，则不需要向上传递，直接返回
}




void Octree::transfer_vertex_id(Octnode* node, Octnode* root, int vertex_id, int i) {//向下传递id直到根节点
    if (node == root) return;  // 如果是 root 节点，直接退出

    // 根据 i 设置当前节点的 vertex[i]
    node->vertexnotsaved[i] = 0;
    node->vertex[i]->id = vertex_id;

    // 如果当前节点不是叶子节点，则递归进入它的 child[i]
    if (!node->isLeaf()) {
        transfer_vertex_id(node->child[i], root, vertex_id, i);  // 递归处理 child[1]
    }
}


void Octree::remove_duplicate_vertex(Octnode* node, Octnode* root, int vertex_idx)//顶点去重函数+id赋值
{
    int deep = node->depth;
    int idxx = node->indexs->x;
    int idxy = node->indexs->y;
    int idxz = node->indexs->z;
    switch (vertex_idx)
    {
    case 0:

        find_target_node(root, deep, idxx+1, idxy, idxz)->vertexnotsaved[1] = 0;
        find_target_node(root, deep, idxx+1, idxy, idxz)->vertex[1]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx+1, idxy, idxz), root, node->vertex[vertex_idx]->id, 1);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx+1, idxy, idxz), root, node->vertex[vertex_idx]->id, 1);

        find_target_node(root, deep, idxx+1, idxy, idxz - 1)->vertexnotsaved[5] = 0;
        find_target_node(root, deep, idxx+1, idxy, idxz-1)->vertex[5]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx+1, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 5);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx+1, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 5);

        find_target_node(root, deep, idxx+1, idxy-1, idxz - 1)->vertexnotsaved[6] = 0;
        find_target_node(root, deep, idxx+1, idxy-1, idxz-1)->vertex[6]->id = node->vertex[vertex_idx]->id; // 问题代码
        transfer_vertex_id(find_target_node(root, deep, idxx+1, idxy-1, idxz-1), root, node->vertex[vertex_idx]->id, 6);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx+1, idxy-1, idxz-1), root, node->vertex[vertex_idx]->id, 6);

        find_target_node(root, deep, idxx+1, idxy-1, idxz)->vertexnotsaved[2] = 0;
        find_target_node(root, deep, idxx+1, idxy-1, idxz)->vertex[2]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx+1, idxy-1, idxz), root, node->vertex[vertex_idx]->id, 2);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx+1, idxy-1, idxz), root, node->vertex[vertex_idx]->id, 2);

        find_target_node(root, deep, idxx, idxy-1, idxz)->vertexnotsaved[3] = 0;
        find_target_node(root, deep, idxx, idxy-1, idxz)->vertex[3]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy-1, idxz), root, node->vertex[vertex_idx]->id, 3);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy-1, idxz), root, node->vertex[vertex_idx]->id, 3);

        find_target_node(root, deep, idxx, idxy, idxz - 1)->vertexnotsaved[4] = 0;
        find_target_node(root, deep, idxx, idxy, idxz-1)->vertex[4]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy, idxz-1), root, node->vertex[vertex_idx]->id, 4);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy, idxz-1), root, node->vertex[vertex_idx]->id, 4);

        find_target_node(root, deep, idxx, idxy-1, idxz - 1)->vertexnotsaved[7] = 0;
        find_target_node(root, deep, idxx, idxy-1, idxz - 1)->vertex[7]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy-1, idxz - 1), root, node->vertex[vertex_idx]->id, 7);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy-1, idxz - 1), root, node->vertex[vertex_idx]->id, 7);
        break;

    case 1:
        // 更新第一个目标节点的 vertex[2]
        find_target_node(root, deep, idxx, idxy - 1, idxz)->vertexnotsaved[2] = 0;
        find_target_node(root, deep, idxx, idxy - 1, idxz)->vertex[2]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy - 1, idxz), root, node->vertex[vertex_idx]->id, 2);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy - 1, idxz), root, node->vertex[vertex_idx]->id, 2);

        // 更新第二个目标节点的 vertex[3]
        find_target_node(root, deep, idxx - 1, idxy - 1, idxz)->vertexnotsaved[3] = 0;
        find_target_node(root, deep, idxx - 1, idxy - 1, idxz)->vertex[3]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy - 1, idxz), root, node->vertex[vertex_idx]->id, 3);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy - 1, idxz), root, node->vertex[vertex_idx]->id, 3);

        // 更新第三个目标节点的 vertex[6]
        find_target_node(root, deep, idxx, idxy - 1, idxz - 1)->vertexnotsaved[6] = 0;
        find_target_node(root, deep, idxx, idxy - 1, idxz - 1)->vertex[6]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy - 1, idxz - 1), root, node->vertex[vertex_idx]->id, 6);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy - 1, idxz - 1), root, node->vertex[vertex_idx]->id, 6);

        // 更新第四个目标节点的 vertex[0]
        find_target_node(root, deep, idxx - 1, idxy, idxz)->vertexnotsaved[0] = 0;
        find_target_node(root, deep, idxx - 1, idxy, idxz)->vertex[0]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 0);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 0);

        // 更新第五个目标节点的 vertex[7]
        find_target_node(root, deep, idxx - 1, idxy - 1, idxz - 1)->vertexnotsaved[7] = 0;
        find_target_node(root, deep, idxx - 1, idxy - 1, idxz - 1)->vertex[7]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy - 1, idxz - 1), root, node->vertex[vertex_idx]->id, 7);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy - 1, idxz - 1), root, node->vertex[vertex_idx]->id, 7);

        // 更新第六个目标节点的 vertex[4]
        find_target_node(root, deep, idxx - 1, idxy, idxz - 1)->vertexnotsaved[4] = 0;
        find_target_node(root, deep, idxx - 1, idxy, idxz - 1)->vertex[4]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 4);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 4);

        // 更新第七个目标节点的 vertex[5]
        find_target_node(root, deep, idxx, idxy, idxz - 1)->vertexnotsaved[5] = 0;
        find_target_node(root, deep, idxx, idxy, idxz - 1)->vertex[5]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 5);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 5);

        break;


    case 2:
        // 更新第一个目标节点的 vertex[3]
        find_target_node(root, deep, idxx - 1, idxy, idxz)->vertexnotsaved[3] = 0;
        find_target_node(root, deep, idxx - 1, idxy, idxz)->vertex[3]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 3);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 3);

        // 更新第二个目标节点的 vertex[0]
        find_target_node(root, deep, idxx - 1, idxy + 1, idxz)->vertexnotsaved[0] = 0;
        find_target_node(root, deep, idxx - 1, idxy + 1, idxz)->vertex[0]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 0);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 0);

        // 更新第三个目标节点的 vertex[1]
        find_target_node(root, deep, idxx, idxy + 1, idxz)->vertexnotsaved[1] = 0;
        find_target_node(root, deep, idxx, idxy + 1, idxz)->vertex[1]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 1);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 1);

        // 更新第四个目标节点的 vertex[6]
        find_target_node(root, deep, idxx, idxy, idxz - 1)->vertexnotsaved[6] = 0;
        find_target_node(root, deep, idxx, idxy, idxz - 1)->vertex[6]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 6);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 6);

        // 更新第五个目标节点的 vertex[7]
        find_target_node(root, deep, idxx - 1, idxy, idxz - 1)->vertexnotsaved[7] = 0;
        find_target_node(root, deep, idxx - 1, idxy, idxz - 1)->vertex[7]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 7);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 7);

        // 更新第六个目标节点的 vertex[4]
        find_target_node(root, deep, idxx - 1, idxy + 1, idxz - 1)->vertexnotsaved[4] = 0;
        find_target_node(root, deep, idxx - 1, idxy + 1, idxz - 1)->vertex[4]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy + 1, idxz - 1), root, node->vertex[vertex_idx]->id, 4);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy + 1, idxz - 1), root, node->vertex[vertex_idx]->id, 4);

        // 更新第七个目标节点的 vertex[5]（解决问题部分）
        find_target_node(root, deep, idxx, idxy + 1, idxz - 1)->vertexnotsaved[5] = 0;
        find_target_node(root, deep, idxx, idxy + 1, idxz - 1)->vertex[5]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy + 1, idxz - 1), root, node->vertex[vertex_idx]->id, 5);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy + 1, idxz - 1), root, node->vertex[vertex_idx]->id, 5);

        break;

    case 3:
        // 更新第一个目标节点的 vertex[0]
        find_target_node(root, deep, idxx, idxy + 1, idxz)->vertexnotsaved[0] = 0;
        find_target_node(root, deep, idxx, idxy + 1, idxz)->vertex[0]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 0);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 0);

        // 更新第二个目标节点的 vertex[4]
        find_target_node(root, deep, idxx, idxy + 1, idxz - 1)->vertexnotsaved[4] = 0;
        find_target_node(root, deep, idxx, idxy + 1, idxz - 1)->vertex[4]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy + 1, idxz - 1), root, node->vertex[vertex_idx]->id, 4);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy + 1, idxz - 1), root, node->vertex[vertex_idx]->id, 4);

        // 更新第三个目标节点的 vertex[7]
        find_target_node(root, deep, idxx, idxy, idxz - 1)->vertexnotsaved[7] = 0;
        find_target_node(root, deep, idxx, idxy, idxz - 1)->vertex[7]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 7);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 7);

        // 更新第四个目标节点的 vertex[2]
        find_target_node(root, deep, idxx + 1, idxy, idxz)->vertexnotsaved[2] = 0;
        find_target_node(root, deep, idxx + 1, idxy, idxz)->vertex[2]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx + 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 2);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx + 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 2);

        // 更新第五个目标节点的 vertex[1]
        find_target_node(root, deep, idxx + 1, idxy + 1, idxz)->vertexnotsaved[1] = 0;
        find_target_node(root, deep, idxx + 1, idxy + 1, idxz)->vertex[1]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx + 1, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 1);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx + 1, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 1);

        // 更新第六个目标节点的 vertex[6]
        find_target_node(root, deep, idxx + 1, idxy, idxz - 1)->vertexnotsaved[6] = 0;
        find_target_node(root, deep, idxx + 1, idxy, idxz - 1)->vertex[6]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx + 1, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 6);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx + 1, idxy, idxz - 1), root, node->vertex[vertex_idx]->id, 6);

        // 更新第七个目标节点的 vertex[5]
        find_target_node(root, deep, idxx + 1, idxy + 1, idxz - 1)->vertexnotsaved[5] = 0;
        find_target_node(root, deep, idxx + 1, idxy + 1, idxz - 1)->vertex[5]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx + 1, idxy + 1, idxz - 1), root, node->vertex[vertex_idx]->id, 5);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx + 1, idxy + 1, idxz - 1), root, node->vertex[vertex_idx]->id, 5);

        break;

    case 4:
        // 更新第一个目标节点的 vertex[0]
        find_target_node(root, deep, idxx, idxy, idxz + 1)->vertexnotsaved[0] = 0;
        find_target_node(root, deep, idxx, idxy, idxz + 1)->vertex[0]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 0);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 0);

        // 更新第二个目标节点的 vertex[7]
        find_target_node(root, deep, idxx, idxy - 1, idxz)->vertexnotsaved[7] = 0;
        find_target_node(root, deep, idxx, idxy - 1, idxz)->vertex[7]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy - 1, idxz), root, node->vertex[vertex_idx]->id, 7);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy - 1, idxz), root, node->vertex[vertex_idx]->id, 7);

        // 更新第三个目标节点的 vertex[3]
        find_target_node(root, deep, idxx, idxy - 1, idxz + 1)->vertexnotsaved[3] = 0;
        find_target_node(root, deep, idxx, idxy - 1, idxz + 1)->vertex[3]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy - 1, idxz + 1), root, node->vertex[vertex_idx]->id, 3);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy - 1, idxz + 1), root, node->vertex[vertex_idx]->id, 3);

        // 更新第四个目标节点的 vertex[5]
        find_target_node(root, deep, idxx + 1, idxy, idxz)->vertexnotsaved[5] = 0;
        find_target_node(root, deep, idxx + 1, idxy, idxz)->vertex[5]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx + 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 5);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx + 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 5);

        // 更新第五个目标节点的 vertex[1]
        find_target_node(root, deep, idxx + 1, idxy, idxz + 1)->vertexnotsaved[1] = 0;
        find_target_node(root, deep, idxx + 1, idxy, idxz + 1)->vertex[1]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx + 1, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 1);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx + 1, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 1);

        // 更新第六个目标节点的 vertex[6]
        find_target_node(root, deep, idxx + 1, idxy - 1, idxz)->vertexnotsaved[6] = 0;
        find_target_node(root, deep, idxx + 1, idxy - 1, idxz)->vertex[6]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx + 1, idxy - 1, idxz), root, node->vertex[vertex_idx]->id, 6);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx + 1, idxy - 1, idxz), root, node->vertex[vertex_idx]->id, 6);

        // 更新第七个目标节点的 vertex[2]
        find_target_node(root, deep, idxx + 1, idxy - 1, idxz + 1)->vertexnotsaved[2] = 0;
        find_target_node(root, deep, idxx + 1, idxy - 1, idxz + 1)->vertex[2]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx + 1, idxy - 1, idxz + 1), root, node->vertex[vertex_idx]->id, 2);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx + 1, idxy - 1, idxz + 1), root, node->vertex[vertex_idx]->id, 2);

        break;

    case 5:
        find_target_node(root, deep, idxx, idxy - 1, idxz)->vertexnotsaved[6] = 0;
        find_target_node(root, deep, idxx, idxy - 1, idxz)->vertex[6]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy - 1, idxz), root, node->vertex[vertex_idx]->id, 6);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy - 1, idxz), root, node->vertex[vertex_idx]->id, 6);

        // 更新第二个目标节点的 vertex[2]
        find_target_node(root, deep, idxx, idxy - 1, idxz + 1)->vertexnotsaved[2] = 0;
        find_target_node(root, deep, idxx, idxy - 1, idxz + 1)->vertex[2]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy - 1, idxz + 1), root, node->vertex[vertex_idx]->id, 2);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy - 1, idxz + 1), root, node->vertex[vertex_idx]->id, 2);

        // 更新第三个目标节点的 vertex[1]
        find_target_node(root, deep, idxx, idxy, idxz + 1)->vertexnotsaved[1] = 0;
        find_target_node(root, deep, idxx, idxy, idxz + 1)->vertex[1]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 1);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 1);

        // 更新第四个目标节点的 vertex[4]
        find_target_node(root, deep, idxx - 1, idxy, idxz)->vertexnotsaved[4] = 0;
        find_target_node(root, deep, idxx - 1, idxy, idxz)->vertex[4]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 4);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 4);

        // 更新第五个目标节点的 vertex[7]
        find_target_node(root, deep, idxx - 1, idxy - 1, idxz)->vertexnotsaved[7] = 0;
        find_target_node(root, deep, idxx - 1, idxy - 1, idxz)->vertex[7]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy - 1, idxz), root, node->vertex[vertex_idx]->id, 7);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy - 1, idxz), root, node->vertex[vertex_idx]->id, 7);

        // 更新第六个目标节点的 vertex[3]
        find_target_node(root, deep, idxx - 1, idxy - 1, idxz + 1)->vertexnotsaved[3] = 0;
        find_target_node(root, deep, idxx - 1, idxy - 1, idxz + 1)->vertex[3]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy - 1, idxz + 1), root, node->vertex[vertex_idx]->id, 3);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy - 1, idxz + 1), root, node->vertex[vertex_idx]->id, 3);

        // 更新第七个目标节点的 vertex[0]
        find_target_node(root, deep, idxx - 1, idxy, idxz + 1)->vertexnotsaved[0] = 0;
        find_target_node(root, deep, idxx - 1, idxy, idxz + 1)->vertex[0]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 0);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 0);

        break;

    case 6:
        // 更新第一个目标节点的 vertex[5]
        find_target_node(root, deep, idxx, idxy + 1, idxz)->vertexnotsaved[5] = 0;
        find_target_node(root, deep, idxx, idxy + 1, idxz)->vertex[5]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 5);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 5);

        // 更新第二个目标节点的 vertex[2]
        find_target_node(root, deep, idxx, idxy, idxz + 1)->vertexnotsaved[2] = 0;
        find_target_node(root, deep, idxx, idxy, idxz + 1)->vertex[2]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 2);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 2);

        // 更新第三个目标节点的 vertex[1]
        find_target_node(root, deep, idxx, idxy + 1, idxz + 1)->vertexnotsaved[1] = 0;
        find_target_node(root, deep, idxx, idxy + 1, idxz + 1)->vertex[1]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy + 1, idxz + 1), root, node->vertex[vertex_idx]->id, 1);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy + 1, idxz + 1), root, node->vertex[vertex_idx]->id, 1);

        // 更新第四个目标节点的 vertex[7]
        find_target_node(root, deep, idxx - 1, idxy, idxz)->vertexnotsaved[7] = 0;
        find_target_node(root, deep, idxx - 1, idxy, idxz)->vertex[7]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 7);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 7);

        // 更新第五个目标节点的 vertex[3]
        find_target_node(root, deep, idxx - 1, idxy, idxz + 1)->vertexnotsaved[3] = 0;
        find_target_node(root, deep, idxx - 1, idxy, idxz + 1)->vertex[3]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 3);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 3);

        // 更新第六个目标节点的 vertex[0]
        find_target_node(root, deep, idxx - 1, idxy + 1, idxz + 1)->vertexnotsaved[0] = 0;
        find_target_node(root, deep, idxx - 1, idxy + 1, idxz + 1)->vertex[0]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy + 1, idxz + 1), root, node->vertex[vertex_idx]->id, 0);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy + 1, idxz + 1), root, node->vertex[vertex_idx]->id, 0);

        // 更新第七个目标节点的 vertex[4]
        find_target_node(root, deep, idxx - 1, idxy + 1, idxz)->vertexnotsaved[4] = 0;
        find_target_node(root, deep, idxx - 1, idxy + 1, idxz)->vertex[4]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx - 1, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 4);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx - 1, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 4);

        break;


    case 7: {
        // 更新第一个目标节点的 vertex[4]
        find_target_node(root, deep, idxx, idxy + 1, idxz)->vertexnotsaved[4] = 0;
        find_target_node(root, deep, idxx, idxy + 1, idxz)->vertex[4]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 4);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 4);

        // 更新第二个目标节点的 vertex[3]
        find_target_node(root, deep, idxx, idxy, idxz + 1)->vertexnotsaved[3] = 0;
        find_target_node(root, deep, idxx, idxy, idxz + 1)->vertex[3]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 3);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 3);

        // 更新第三个目标节点的 vertex[0]
        find_target_node(root, deep, idxx, idxy + 1, idxz + 1)->vertexnotsaved[0] = 0;
        find_target_node(root, deep, idxx, idxy + 1, idxz + 1)->vertex[0]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx, idxy + 1, idxz + 1), root, node->vertex[vertex_idx]->id, 0);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx, idxy + 1, idxz + 1), root, node->vertex[vertex_idx]->id, 0);

        // 更新第四个目标节点的 vertex[6]
        find_target_node(root, deep, idxx + 1, idxy, idxz)->vertexnotsaved[6] = 0;
        find_target_node(root, deep, idxx + 1, idxy, idxz)->vertex[6]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx + 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 6);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx + 1, idxy, idxz), root, node->vertex[vertex_idx]->id, 6);

        // 更新第五个目标节点的 vertex[2]
        find_target_node(root, deep, idxx + 1, idxy, idxz + 1)->vertexnotsaved[2] = 0;
        find_target_node(root, deep, idxx + 1, idxy, idxz + 1)->vertex[2]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx + 1, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 2);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx + 1, idxy, idxz + 1), root, node->vertex[vertex_idx]->id, 2);

        // 更新第六个目标节点的 vertex[1]
        find_target_node(root, deep, idxx + 1, idxy + 1, idxz + 1)->vertexnotsaved[1] = 0;
        find_target_node(root, deep, idxx + 1, idxy + 1, idxz + 1)->vertex[1]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx + 1, idxy + 1, idxz + 1), root, node->vertex[vertex_idx]->id, 1);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx + 1, idxy + 1, idxz + 1), root, node->vertex[vertex_idx]->id, 1);

        // 更新第七个目标节点的 vertex[5]
        find_target_node(root, deep, idxx + 1, idxy + 1, idxz)->vertexnotsaved[5] = 0;
        find_target_node(root, deep, idxx + 1, idxy + 1, idxz)->vertex[5]->id = node->vertex[vertex_idx]->id;
        transfer_vertex_id(find_target_node(root, deep, idxx + 1, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 5);
        transfer_vertex_id_upward(find_target_node(root, deep, idxx + 1, idxy + 1, idxz), root, node->vertex[vertex_idx]->id, 5);

        break;
    }

    }
}



/* Octnode* Octree::findLeafChild(Octnode* root, int start_depth,int start_x, int start_y, int start_z,int child_index)
   {

         Octnode* current = find_target_node(root, start_depth, start_x, start_y, start_z);
         if(!exist(root, start_depth, start_x, start_y, start_z)){std::cerr << "警告：角悬挂顶点的父顶点不存在"<<std::endl;
         return nullptr;
         }
         while(!current->isLeaf())
         {
             current = current->child[child_index];//更新current
         }
         return current;
   }*/


Octree::HangingVertexInfo Octree::find_hanging_vertex_parent_node(Octnode* node, int vertex_idx) {//返回该点坐标和所在的node，针对于在顶点的情况
    // 获取顶点的三维坐标
    int x = node->vindexs[vertex_idx]->x;
    int y = node->vindexs[vertex_idx]->y;
    int z = node->vindexs[vertex_idx]->z;
    std::cout<<"原顶点indexs"<<x<<" "<<y<<" "<<z<<std::endl;
    int division_count = 0;
    int original_x = x;
    int original_y = y;
    int original_z = z;
    // 当所有坐标都是偶数（0除外）时继续循环
    while (true) {
        // 检查是否有坐标为奇数
        bool has_odd = (x % 2 != 0 && x != 0) ||
                (y % 2 != 0 && y != 0) ||
                (z % 2 != 0 && z != 0);

        if (has_odd) {
            break;
        }

        // 所有坐标除以2
        x /= 2;
        y /= 2;
        z /= 2;

        division_count++;
    }
    float target_x = (float)x / 2.0f;
    float target_y = (float)y / 2.0f;
    float target_z = (float)z / 2.0f;
    Octnode* target_node = node;
    //std::cout<<node->parent->idx<<" "<<node->idx<<std::endl;
    //std::cout<<vertex_idx<<std::endl;
    for (int i = 0; i <= division_count && target_node->parent != nullptr; i++) {
        target_node = target_node->parent;
    }
    std::cout<<"原节点"<<node->depth<<node->indexs->x<<" "<<node->indexs->y<<" "<<node->indexs->z<<std::endl;
    std::cout<<division_count<<std::endl;
    std::cout<<"目标节点"<<target_node->depth<<target_node->indexs->x<<" "<<target_node->indexs->y<<" "<<target_node->indexs->z<<std::endl;
    return HangingVertexInfo(target_node, target_x, target_y, target_z);
}

Octree::VertexPositionInfo Octree::determine_hanging_vertex_position(const HangingVertexInfo& info) {
    // 获取目标节点和坐标
    Octnode* target_node = info.target_node;
    float target_x = info.target_x;
    float target_y = info.target_y;
    float target_z = info.target_z;
    std::cout<<target_x<<" "<<target_y<<" "<<target_z<<std::endl;
    for (int i = 0; i < 8; i++) {
        if (target_node->vindexs[i] != nullptr) {
            std::cout <<target_node->depth<<","
                     << "顶点 " << i << ": ("
                     << target_node->vindexs[i]->x << ", "
                     << target_node->vindexs[i]->y << ", "
                     << target_node->vindexs[i]->z << ") ID: "
                     << target_node->vertex[i]->id << std::endl;
        }
    }
    // 存储目标节点的8个顶点坐标
    struct Point {
        float x, y, z;
        int idx;
        Point(float _x, float _y, float _z, int _idx) : x(_x), y(_y), z(_z), idx(_idx) {}
    };

    std::vector<Point> vertices;
    for (int i = 0; i < 8; i++) {
        vertices.push_back(Point(
                               target_node->vindexs[i]->x,
                               target_node->vindexs[i]->y,
                               target_node->vindexs[i]->z,
                               i
                               ));
    }

    // 判断点的位置
    std::string position = "未知位置";
    int vertex1_id = -1, vertex2_id = -1;
    float epsilon = 1e-5; // 浮点数比较的误差范围

    // 检查是否是顶点
    for (int i = 0; i < vertices.size(); i++) {//遍历8个顶点，如果有重合的顶点，就说明是顶点
        if (fabs(target_x - vertices[i].x) < epsilon &&
                fabs(target_y - vertices[i].y) < epsilon &&
                fabs(target_z - vertices[i].z) < epsilon) {
            position = "顶点";
            vertex1_id = target_node->vertex[i]->id;
            std::cout<<"顶点"<<std::endl;
            if (vertex1_id ==-9999999 ) {
                std::cout << "错误：找到的父节点顶点ID无效（ID = 0）" << std::endl;
            }
            break;
        }
    }

    // 检查是否在边上
    if (position == "未知位置") {
        // 定义八叉树节点的12条边
        const int edges[12][2] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0},  // 底面的四条边
            {4, 5}, {5, 6}, {6, 7}, {7, 4},  // 顶面的四条边
            {0, 4}, {1, 5}, {2, 6}, {3, 7}   // 连接底面和顶面的四条边
        };

        for (int i = 0; i < 12; i++) {//遍历12条边，判断是否在边中点
            int v1 = edges[i][0];//一条边的两个顶点编号0~7
            int v2 = edges[i][1];

            // 计算边的中点
            float mid_x = (vertices[v1].x + vertices[v2].x) / 2.0f;
            float mid_y = (vertices[v1].y + vertices[v2].y) / 2.0f;
            float mid_z = (vertices[v1].z + vertices[v2].z) / 2.0f;

            if (fabs(target_x - mid_x) < epsilon &&
                    fabs(target_y - mid_y) < epsilon &&
                    fabs(target_z - mid_z) < epsilon) {
                position = "边中点";
                std::cout<<"边中点"<<std::endl;
                vertex1_id = target_node->vertex[v1]->id;
                vertex2_id = target_node->vertex[v2]->id;
                if (vertex1_id == -9999999 || vertex2_id == -9999999) {
                    std::cout << "错误：找到的父节点顶点ID无效（ID = 0）" << std::endl;
                    if (vertex1_id == -9999999) std::cout << "  - 第一个父节点ID无效" << std::endl;
                    if (vertex2_id == -9999999) std::cout << "  - 第二个父节点ID无效" << std::endl;
                }
                break;
            }

        }
    }

    // 检查是否在面上
    if (position == "未知位置") {
        // 定义八叉树节点的6个面
        const int faces[6][4] = {
            {0, 1, 2, 3},  // 底面
            {4, 5, 6, 7},  // 顶面
            {0, 1, 5, 4},  // 前面
            {2, 3, 7, 6},  // 后面
            {0, 3, 7, 4},  // 左面
            {1, 2, 6, 5}   // 右面
        };

        for (int i = 0; i < 6; i++) {
            // 计算面的中心点
            float center_x = 0, center_y = 0, center_z = 0;
            for (int j = 0; j < 4; j++) {
                int v = faces[i][j];//遍历到了第几个面的第几个点
                center_x += vertices[v].x;
                center_y += vertices[v].y;
                center_z += vertices[v].z;
            }
            center_x /= 4.0f;
            center_y /= 4.0f;
            center_z /= 4.0f;

            if (fabs(target_x - center_x) < epsilon &&
                    fabs(target_y - center_y) < epsilon &&
                    fabs(target_z - center_z) < epsilon) {
                position = "面中点";
                std::cout<<"面中点"<<std::endl;
                vertex1_id = target_node->vertex[faces[i][0]]->id;
                vertex2_id = target_node->vertex[faces[i][2]]->id;
                // 检查父节点ID是否有效
                if (vertex1_id == -9999999 || vertex2_id == -9999999) {
                    std::cout << "错误：找到的父节点顶点ID无效（ID = 0）" << std::endl;
                    if (vertex1_id == -9999999) std::cout << "  - 第一个父节点ID无效" << std::endl;
                    if (vertex2_id ==-9999999) std::cout << "  - 第二个父节点ID无效" << std::endl;
                }
                break;
            }
        }
    }

    // 检查是否是体中心点
    if (position == "未知位置") {
        float center_x = 0, center_y = 0, center_z = 0;
        for (const auto& v : vertices) {
            center_x += v.x;
            center_y += v.y;
            center_z += v.z;
        }
        center_x /= 8.0f;
        center_y /= 8.0f;
        center_z /= 8.0f;

        if (fabs(target_x - center_x) < epsilon &&
                fabs(target_y - center_y) < epsilon &&
                fabs(target_z - center_z) < epsilon) {
            position = "体中心点";
            // 对于体中心点，可以选择两个对角顶点作为参考
            std::cout<<"体中心点"<<std::endl;
            vertex1_id = target_node->vertex[0]->id;
            vertex2_id = target_node->vertex[6]->id;
            // 检查父节点ID是否有效
            if (vertex1_id == -9999999|| vertex2_id ==-9999999) {
                std::cerr << "错误：找到的父节点顶点ID无效（ID = 0）" << std::endl;
                if (vertex1_id ==-9999999) std::cerr << "  - 第一个父节点ID无效" << std::endl;
                if (vertex2_id == -9999999) std::cerr << "  - 第二个父节点ID无效" << std::endl;
            }
        }
    }

    // 返回包含位置类型和顶点ID的结构体
    return VertexPositionInfo(position, vertex1_id, vertex2_id);
}



void Octree::get_hangingVertex(std::vector< GLVertex*>& hangingVertexList,std::vector< GLVertex*>& normalVertexList)//针对整个树进行悬挂节点判断
{
    std::vector<Octnode*> leaf_node_list;  // 存储叶子节点的列表
    get_leaf_nodes2(root, leaf_node_list);  // 递归获取叶子节点
    Octnode* root = this->root;
    int hangingVertexId =1;  // 悬挂节点从1开始的正数
    int normalVertexId = 0;
    BOOST_FOREACH(Octnode* node, leaf_node_list)//遍历所有叶子节点
    {
        int index=node->idx;
        int deep = node->depth;
        int idxx = node->indexs->x;
        int idxy = node->indexs->y;
        int idxz = node->indexs->z;
        for (int i = 0; i < 8; i++)
        {
            if (node->vertexnotsaved[i])
            { //Vertex[i] = node.center + Direction[i] * node.scale;//修改
                switch (index)
                {//0-7
                case 0:

                    switch (i)
                    {
                    case 0://0号octnode的0号顶点
                        if(node->depth>1)
                        {
                            if(node->vertex[i]->x==root->scale&&node->vertex[i]->y==-root->scale&&node->vertex[i]->z==-root->scale)//判断是不是8个根顶点
                                                    {
                                                           node->vertex[0]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                                           remove_duplicate_vertex(node, root, i);
                                                           node->vertexnotsaved[i] = 0;
                                                          normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                                          transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                                     }
                            else

                            {
                                if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                             {
                                node->vertex[i]->setId(hangingVertexId++);
                                hangingVertexList.push_back(node->vertex[i]);//悬挂节点存储
                                remove_duplicate_vertex(node, root, i);
                                transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                             }
                            else
                            {//顶点去重问题，思路：只需要考虑同级顶点的重复，8个角点全部node->vertexnotsaved[i] = 0;
                                node->vertex[0]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                remove_duplicate_vertex(node, root, i);
                                node->vertexnotsaved[i] = 0;
                                normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                            }
                            }


                        }
                        //如果是deep=1的角点直接不做任何操作
                        break;

                    case 1://0号octnode的1号顶点，这是一个新生点，去重+判断悬挂

                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {node->vertex[1]->setId(hangingVertexId++);
                            hangingVertexList.push_back(node->vertex[1]);//悬挂节点存储
                            remove_duplicate_vertex(node, root, i);//去重+id复制
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);

                        }
                        else
                        {//顶点去重问题，思路：只需要考虑同级顶点的重复，8个角点全部node->vertexnotsaved[i] = 0;
                            node->vertex[1]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                            remove_duplicate_vertex(node, root, i);
                            node->vertexnotsaved[i] = 0;
                            normalVertexList.push_back(node->vertex[1]);//非悬挂节点存储
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);

                        }

                        break;

                    case 2://这是一个面悬挂节点
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[2]);
                            node->vertex[2]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);

                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[2]);
                            node->vertex[2]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }

                        break;

                    case 3://边悬挂节点
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);

                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }

                        break;

                    case 4:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }

                        break;

                    case 5://面
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }

                        break;

                    case 6://中心点
                        normalVertexList.push_back(node->vertex[i]);
                        node->vertex[i]->setId(normalVertexId--);
                        node->vertexnotsaved[i] = 0;
                        remove_duplicate_vertex(node, root, i);
                        transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);

                        break;

                    case 7://面
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }

                        break;
                    }
                    break;


                case 1:
                    switch (i)
                    {  case 0: // 边悬挂
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index, i)) {
                            node->vertex[i]->setId(hangingVertexId++);  // 先设置 ID
                            remove_duplicate_vertex(node, root, i);    // 然后进行去重
                            hangingVertexList.push_back(node->vertex[i]);  // 存储悬挂节点
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else {
                            node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                            remove_duplicate_vertex(node, root, i);   // 然后进行去重
                            node->vertexnotsaved[i] = 0;
                            normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;


                    case 1://d角点
                        if(node->depth>1)
                        {if(node->vertex[i]->x==-root->scale&&node->vertex[i]->y==-root->scale&&node->vertex[i]->z==-root->scale)//判断是不是8个根顶点
                            {
                                   node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                   remove_duplicate_vertex(node, root, i);
                                   node->vertexnotsaved[i] = 0;
                                  normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                  transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                             }

                            else
                            {  if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                                {
                                    node->vertex[i]->setId(hangingVertexId++);
                                    hangingVertexList.push_back(node->vertex[i]);//悬挂节点存储
                                    remove_duplicate_vertex(node, root, i);//去重+id复制
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                                else
                                {//顶点去重问题，思路：只需要考虑同级顶点的重复，8个角点全部node->vertexnotsaved[i] = 0;
                                    node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                    remove_duplicate_vertex(node, root, i);
                                    node->vertexnotsaved[i] = 0;
                                    normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                            }

                        }
                        //如果是deep=1的角点直接不做任何操作
                        break;

                    case 2: // 边
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index, i)) {
                            node->vertex[i]->setId(hangingVertexId++);  // 先设置 ID
                            hangingVertexList.push_back(node->vertex[i]);  // 存储悬挂节点
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else {
                            node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                            normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 3: // 面
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index, i)) {
                            node->vertex[i]->setId(hangingVertexId++);  // 先设置 ID
                            hangingVertexList.push_back(node->vertex[i]);  // 存储悬挂节点
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else {
                            node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                            normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 4: // 面
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index, i)) {
                            node->vertex[i]->setId(hangingVertexId++);  // 先设置 ID
                            hangingVertexList.push_back(node->vertex[i]);  // 存储悬挂节点
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else {
                            node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                            normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 5: // 边
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index, i)) {
                            node->vertex[i]->setId(hangingVertexId++);  // 先设置 ID
                            hangingVertexList.push_back(node->vertex[i]);  // 存储悬挂节点
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else {
                            node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                            normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 6:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index, i)) {
                            node->vertex[i]->setId(hangingVertexId++);  // 先设置 ID
                            hangingVertexList.push_back(node->vertex[i]);  // 存储悬挂节点
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else {
                            node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                            normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 7:
                        node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                        normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                        node->vertexnotsaved[i] = 0;
                        remove_duplicate_vertex(node, root, i);  // 然后进行去重
                        transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        break;
                    }
                    break;

                case 2:
                    switch (i)
                    {
                    case 0: // 面
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index, i)) {
                            node->vertex[i]->setId(hangingVertexId++);  // 先设置 ID
                            hangingVertexList.push_back(node->vertex[i]);  // 存储悬挂节点
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else {
                            node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                            normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 1: // 边
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index, i)) {
                            node->vertex[i]->setId(hangingVertexId++);  // 先设置 ID
                            hangingVertexList.push_back(node->vertex[i]);  // 存储悬挂节点
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else {
                            node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                            normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 2:
                        if(node->depth>1)
                        {if(node->vertex[i]->x==-root->scale&&node->vertex[i]->y==root->scale&&node->vertex[i]->z==-root->scale)//判断是不是8个根顶点
                            {
                                   node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                   remove_duplicate_vertex(node, root, i);
                                   node->vertexnotsaved[i] = 0;
                                  normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                  transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                             }
                            else
                            {  if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                                {
                                    node->vertex[i]->setId(hangingVertexId++);
                                    hangingVertexList.push_back(node->vertex[i]);//悬挂节点存储
                                    remove_duplicate_vertex(node, root, i);//去重+id复制
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                                else
                                {//顶点去重问题，思路：只需要考虑同级顶点的重复，8个角点全部node->vertexnotsaved[i] = 0;
                                    node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                    remove_duplicate_vertex(node, root, i);
                                    node->vertexnotsaved[i] = 0;
                                    normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                            }

                        }
                        //如果是deep=1的角点直接不做任何操作
                        break;

                    case 3: // 边
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index, i)) {
                            node->vertex[i]->setId(hangingVertexId++);  // 先设置 ID
                            hangingVertexList.push_back(node->vertex[i]);  // 存储悬挂节点
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else {
                            node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                            normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 4:
                        node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                        normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                        node->vertexnotsaved[i] = 0;
                        remove_duplicate_vertex(node, root, i);  // 然后进行去重
                        transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        break;

                    case 5:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index, i)) {
                            node->vertex[i]->setId(hangingVertexId++);  // 先设置 ID
                            hangingVertexList.push_back(node->vertex[i]);  // 存储悬挂节点
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else {
                            node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                            normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 6:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index, i)) {
                            node->vertex[i]->setId(hangingVertexId++);  // 先设置 ID
                            hangingVertexList.push_back(node->vertex[i]);  // 存储悬挂节点
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else {
                            node->vertex[i]->setId(normalVertexId--);  // 先设置 ID
                            normalVertexList.push_back(node->vertex[i]);  // 存储正常节点
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);  // 然后进行去重
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 7:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    }
                    break;


                case 3:
                    switch (i)
                    {
                    case 0:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 1:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 2:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 3:
                        if(node->depth>1)
                        {if(node->vertex[i]->x==root->scale&&node->vertex[i]->y==root->scale&&node->vertex[i]->z==-root->scale)//判断是不是8个根顶点
                            {
                                   node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                   remove_duplicate_vertex(node, root, i);
                                   node->vertexnotsaved[i] = 0;
                                  normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                  transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                             }
                            else
                            {  if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                                {
                                    node->vertex[i]->setId(hangingVertexId++);
                                    hangingVertexList.push_back(node->vertex[i]);//悬挂节点存储
                                    remove_duplicate_vertex(node, root, i);//去重+id复制
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                                else
                                {//顶点去重问题，思路：只需要考虑同级顶点的重复，8个角点全部node->vertexnotsaved[i] = 0;
                                    node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                    remove_duplicate_vertex(node, root, i);
                                    node->vertexnotsaved[i] = 0;
                                    normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                            }

                        }
                        //如果是deep=1的角点直接不做任何操作
                        break;

                    case 4:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 5:

                        normalVertexList.push_back(node->vertex[i]);
                        node->vertex[i]->setId(normalVertexId--);
                        node->vertexnotsaved[i] = 0;
                        remove_duplicate_vertex(node, root, i);
                        transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        break;
                    case 6:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 7:

                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;
                    }
                    break;


                case 4:
                    switch (i)
                    {

                    case 0:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 1:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 2:

                        normalVertexList.push_back(node->vertex[i]);
                        node->vertex[i]->setId(normalVertexId--);
                        node->vertexnotsaved[i] = 0;
                        remove_duplicate_vertex(node, root, i);
                        transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        break;

                    case 3:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 4:
                        if(node->depth>1)
                        {if(node->vertex[i]->x==root->scale&&node->vertex[i]->y==-root->scale&&node->vertex[i]->z==root->scale)//判断是不是8个根顶点
                            {
                                   node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                   remove_duplicate_vertex(node, root, i);
                                   node->vertexnotsaved[i] = 0;
                                  normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                  transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                             }
                            else
                            {if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                                {
                                    node->vertex[i]->setId(hangingVertexId++);
                                    hangingVertexList.push_back(node->vertex[i]);//悬挂节点存储
                                    remove_duplicate_vertex(node, root, i);//去重+id复制
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                                else
                                {//顶点去重问题，思路：只需要考虑同级顶点的重复，8个角点全部node->vertexnotsaved[i] = 0;
                                    node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                    remove_duplicate_vertex(node, root, i);
                                    node->vertexnotsaved[i] = 0;
                                    normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                            }

                        }
                        //如果是deep=1的角点直接不做任何操作
                        break;

                    case 5:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 6:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 7:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    }
                    break;
                case 5:
                    switch (i)
                    {
                    case 0:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;
                    case 1:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 2:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;
                    case 3:

                        normalVertexList.push_back(node->vertex[i]);
                        node->vertex[i]->setId(normalVertexId--);
                        node->vertexnotsaved[i] = 0;
                        remove_duplicate_vertex(node, root, i);
                        transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        break;
                    case 4:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;
                    case 5:
                        if(node->depth>1)
                        {if(node->vertex[i]->x==-root->scale&&node->vertex[i]->y==-root->scale&&node->vertex[i]->z==root->scale)//判断是不是8个根顶点
                            {
                                   node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                   remove_duplicate_vertex(node, root, i);
                                   node->vertexnotsaved[i] = 0;
                                  normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                  transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                             }
                            else
                            {   if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                                {
                                    node->vertex[i]->setId(hangingVertexId++);
                                    hangingVertexList.push_back(node->vertex[i]);//悬挂节点存储
                                    remove_duplicate_vertex(node, root, i);//去重+id复制
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                                else
                                {//顶点去重问题，思路：只需要考虑同级顶点的重复，8个角点全部node->vertexnotsaved[i] = 0;
                                    node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                    remove_duplicate_vertex(node, root, i);
                                    node->vertexnotsaved[i] = 0;
                                    normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                            }

                        }
                        //如果是deep=1的角点直接不做任何操作
                        break;


                    case 6:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 7:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;


                    }
                    break;
                case 6:
                    switch(i)
                    {
                    case 0:

                        normalVertexList.push_back(node->vertex[i]);
                        node->vertex[i]->setId(normalVertexId--);
                        node->vertexnotsaved[i] = 0;
                        remove_duplicate_vertex(node, root, i);
                        transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        break;
                    case 1:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 2:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 3:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 4:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 5:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 6:
                        if(node->depth>1)
                        {if(node->vertex[i]->x==-root->scale&&node->vertex[i]->y==root->scale&&node->vertex[i]->z==root->scale)//判断是不是8个根顶点
                            {
                                   node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                   remove_duplicate_vertex(node, root, i);
                                   node->vertexnotsaved[i] = 0;
                                  normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                  transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                             }
                            else
                            {
                                if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                                {
                                    node->vertex[i]->setId(hangingVertexId++);
                                    hangingVertexList.push_back(node->vertex[i]);//悬挂节点存储
                                    remove_duplicate_vertex(node, root, i);//去重+id复制
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                                else
                                {//顶点去重问题，思路：只需要考虑同级顶点的重复，8个角点全部node->vertexnotsaved[i] = 0;
                                    node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                    remove_duplicate_vertex(node, root, i);
                                    node->vertexnotsaved[i] = 0;
                                    normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }

                            }
                        }
                        //如果是deep=1的角点直接不做任何操作
                        break;

                    case 7:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;
                    }
                    break;

                case 7:
                    switch(i)
                    {
                    case 0:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 1:

                        normalVertexList.push_back(node->vertex[i]);
                        node->vertex[i]->setId(normalVertexId--);
                        node->vertexnotsaved[i] = 0;
                        remove_duplicate_vertex(node, root, i);
                        transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        break;

                    case 2:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 3:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 4:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 5:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 6:
                        if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                        {
                            hangingVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(hangingVertexId++);
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        else
                        {

                            normalVertexList.push_back(node->vertex[i]);
                            node->vertex[i]->setId(normalVertexId--);
                            node->vertexnotsaved[i] = 0;
                            remove_duplicate_vertex(node, root, i);
                            transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                        }
                        break;

                    case 7:
                        if(node->depth>1)
                        {if(node->vertex[i]->x==root->scale&&node->vertex[i]->y==root->scale&&node->vertex[i]->z==root->scale)//判断是不是8个根顶点
                            {
                                   node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                   remove_duplicate_vertex(node, root, i);
                                   node->vertexnotsaved[i] = 0;
                                  normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                  transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                             }
                            else
                            {
                                if (check_hanging_vertex(root, deep, idxx, idxy, idxz, index,i))
                                { std::cout<<"悬挂了"<<std::endl;
                                    node->vertex[i]->setId(hangingVertexId++);
                                    hangingVertexList.push_back(node->vertex[i]);//悬挂节点存储
                                    remove_duplicate_vertex(node, root, i);//去重+id复制
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                                else
                                {
                                    std::cout<<"一般了"<<std::endl;//顶点去重问题，思路：只需要考虑同级顶点的重复，8个角点全部node->vertexnotsaved[i] = 0;
                                    node->vertex[i]->setId(normalVertexId--); // 先赋值为当前值，然后normalVertexId减少1
                                    remove_duplicate_vertex(node, root, i);
                                    node->vertexnotsaved[i] = 0;
                                    normalVertexList.push_back(node->vertex[i]);//非悬挂节点存储
                                    transfer_vertex_id_upward(node,root,node->vertex[i]->id,i);
                                }
                            }
                        }
                        //如果是deep=1的角点直接不做任何操作
                        break;
                    }
                    break;

                }

            }
        }
    }
}


void Octree::get_hanging_vertex_parent()
{
    std::vector<Octnode*> leaf_node_list;  // 存储叶子节点的列表
    get_leaf_nodes2(root, leaf_node_list);  // 递归获取叶子节点
    Octnode* root = this->root;

    BOOST_FOREACH(Octnode* node, leaf_node_list) // 遍历所有叶子节点
    {
        int index = node->idx;
        int deep = node->depth;
        int idxx = node->indexs->x;
        int idxy = node->indexs->y;
        int idxz = node->indexs->z;

        for (int i = 0; i < 8; i++)
        {
            if (node->vertexnotsaved[i])
            {
                switch (index)
                {
                case 0:
                    switch (i)
                    {
                    case 0:
                    {
                        HangingVertexInfo info = find_hanging_vertex_parent_node(node, i);
                        VertexPositionInfo info1 = determine_hanging_vertex_position(info);
                        node->vertex[i]->parentVertices.push_back(info1.vertex1_id);
                        node->vertex[i]->parentVertices.push_back(info1.vertex2_id);


                    }
                        break;
                    case 1:
                    {
                        int parentVertex1 = find_target_node(root, deep - 1, idxx / 2, idxy / 2, idxz / 2)->vertex[1]->id;
                        node->vertex[1]->parentVertices.push_back(parentVertex1); // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep - 1, idxx / 2, idxy / 2, idxz / 2)->vertex[0]->id;
                        node->vertex[1]->parentVertices.push_back(parentVertex2); // 也将另一个父顶点添加
                    }
                        break;
                    case 2:
                    {
                        int parentVertex1 = find_target_node(root, deep - 1, idxx / 2, idxy / 2, idxz / 2)->vertex[2]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep - 1, idxx / 2, idxy / 2, idxz / 2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 3:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[3]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 4:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[4]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 5:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[5]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 6:
                        std::cout<<"遍历到找体心悬挂点的case"<<std::endl;
                        break;
                    case 7:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[7]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    }
                    break;

                case 1:
                    switch (i)
                    {
                    case 0:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[1]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 1:
                    {
                        HangingVertexInfo info = find_hanging_vertex_parent_node(node, i);
                        VertexPositionInfo info1 = determine_hanging_vertex_position(info);
                        node->vertex[i]->parentVertices.push_back(info1.vertex1_id);
                        node->vertex[i]->parentVertices.push_back(info1.vertex2_id);
                    }
                        break;
                    case 2:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[2]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[1]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 3:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[2]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 4:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[5]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 5:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[5]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[1]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 6:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[1]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 7:
                        std::cout<<"遍历到找体心悬挂点的case"<<std::endl;
                        break;
                    }
                    break;

                case 2:
                    switch (i)
                    {
                    case 0:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[2]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 1:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[2]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[1]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 2:
                    {
                        HangingVertexInfo info = find_hanging_vertex_parent_node(node, i);
                        VertexPositionInfo info1 = determine_hanging_vertex_position(info);
                        node->vertex[i]->parentVertices.push_back(info1.vertex1_id);
                        node->vertex[i]->parentVertices.push_back(info1.vertex2_id);
                    }
                        break;
                    case 3:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[2]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[3]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 4:
                        std::cout<<"遍历到找体心悬挂点的case"<<std::endl;
                        break;
                    case 5:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[1]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 6:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[2]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 7:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[3]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    }
                    break;

                case 3:
                    switch (i)
                    {
                    case 0:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[3]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 1:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[2]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 2:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[2]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[3]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 3:
                    {
                        HangingVertexInfo info = find_hanging_vertex_parent_node(node, i);
                        VertexPositionInfo info1 = determine_hanging_vertex_position(info);
                        node->vertex[i]->parentVertices.push_back(info1.vertex1_id);
                        node->vertex[i]->parentVertices.push_back(info1.vertex2_id);
                    }
                        break;
                    case 4:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[7]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 5:
                        std::cout<<"遍历到找体心悬挂点的case"<<std::endl;
                        break;
                    case 6:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[3]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 7:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[7]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[3]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    }
                    break;

                case 4:
                    switch (i)
                    {
                    case 0:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[4]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 1:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[5]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 2:
                        std::cout<<"遍历到找体心悬挂点的case"<<std::endl;
                        break;
                    case 3:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[7]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 4:
                    {
                        HangingVertexInfo info = find_hanging_vertex_parent_node(node, i);
                        VertexPositionInfo info1 = determine_hanging_vertex_position(info);
                        node->vertex[i]->parentVertices.push_back(info1.vertex1_id);
                        node->vertex[i]->parentVertices.push_back(info1.vertex2_id);
                    }
                        break;
                    case 5:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[4]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[5]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 6:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[4]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    case 7:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[4]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[7]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 也将另一个父顶点添加
                    }
                        break;
                    }
                    break;

                case 5:
                    switch (i)
                    {
                    case 0:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[5]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);
                    }
                        break;
                    case 1:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[1]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[5]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);
                    }
                        break;
                    case 2:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[1]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);
                    }
                        break;
                    case 3:
                        std::cout<<"遍历到找体心悬挂点的case"<<std::endl;
                        break;
                    case 4:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[4]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[5]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);
                    }
                        break;
                    case 5:
                    {
                        HangingVertexInfo info = find_hanging_vertex_parent_node(node, i);
                        VertexPositionInfo info1 = determine_hanging_vertex_position(info);
                        node->vertex[i]->parentVertices.push_back(info1.vertex1_id);
                        node->vertex[i]->parentVertices.push_back(info1.vertex2_id);
                    }
                        break;
                    case 6:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[5]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);
                    }
                        break;
                    case 7:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[4]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);
                    }
                        break;
                    }
                    break;

                case 6:
                    switch (i)
                    {
                    case 0:
                        std::cout<<"遍历到找体心悬挂点的case"<<std::endl;
                        break;
                    case 1:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[1]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2); // 这里添加 case 6, i=1 的处理逻辑
                    }
                        break;
                    case 2:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[2]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2); // 这里添加 case 6, i=2 的处理逻辑
                    }
                        break;
                    case 3:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[3]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2); // 这里添加 case 6, i=3 的处理逻辑
                    }
                        break;
                    case 4:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[4]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2); // 这里添加 case 6, i=4 的处理逻辑
                    }
                        break;
                    case 5:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[5]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2);  // 这里添加 case 6, i=5 的处理逻辑
                    }
                        break;
                    case 6:
                    {
                        HangingVertexInfo info = find_hanging_vertex_parent_node(node, i);
                        VertexPositionInfo info1 = determine_hanging_vertex_position(info);
                        node->vertex[i]->parentVertices.push_back(info1.vertex1_id);
                        node->vertex[i]->parentVertices.push_back(info1.vertex2_id);
                    }
                        break;
                    case 7:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[7]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2); // 这里添加 case 6, i=7 的处理逻辑
                    }
                        break;
                    }
                    break;

                case 7:
                    switch (i)
                    {
                    case 0:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[0]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[7]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2); // 这里添加 case 7, i=0 的处理逻辑
                    }
                        break;
                    case 1:
                        std::cout<<"遍历到找体心悬挂点的case"<<std::endl;
                        break;
                    case 2:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[3]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2); // 这里添加 case 7, i=2 的处理逻辑
                    }
                        break;
                    case 3:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[7]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[3]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2); // 这里添加 case 7, i=3 的处理逻辑
                    }
                        break;
                    case 4:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[7]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[4]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2); // 这里添加 case 7, i=4 的处理逻辑
                    }
                        break;
                    case 5:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[4]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2); // 这里添加 case 7, i=5 的处理逻辑
                    }
                        break;
                    case 6:
                    {
                        int parentVertex1 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[7]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex1);  // 将父顶点添加到当前顶点
                        int parentVertex2 = find_target_node(root, deep-1, idxx/2, idxy/2, idxz/2)->vertex[6]->id;
                        node->vertex[i]->parentVertices.push_back(parentVertex2); // 这里添加 case 7, i=6 的处理逻辑
                    }
                        break;
                    case 7:
                    {
                        HangingVertexInfo info = find_hanging_vertex_parent_node(node, i);
                        VertexPositionInfo info1 = determine_hanging_vertex_position(info);
                        node->vertex[i]->parentVertices.push_back(info1.vertex1_id);
                        node->vertex[i]->parentVertices.push_back(info1.vertex2_id);
                    }
                        break;
                    }
                    break;
                }
            }
        }
    }
}
///查找顶点
GLVertex* Octree::findVerticesById(int id)  {
    // 处理正负ID的情况
    int lookupId = id;

    auto it = vertexIdMap.find(lookupId);
    if (it != vertexIdMap.end() && !it->second.empty()) {
        return it->second[0]; // 只返回第一个匹配的顶点
    }
    return nullptr; // 未找到返回空指针
}

// 构建顶点ID映射表
void Octree::buildVertexIdMap(std::vector<GLVertex*>& normalvertices) {
    vertexIdMap.clear();

    // 获取所有叶节点
    std::vector<Octnode*> leaf_nodes;
    get_leaf_nodes2(root, leaf_nodes);

    // 遍历所有叶节点的顶点并添加到映射表中
    for (Octnode* node : leaf_nodes) {
        for (int i = 0; i < 8; i++) {
            if (node->vertex[i]) {
                int id = node->vertex[i]->id;
                int lookupId;
                if (id <= 0) {
                    lookupId = -id; // 对于负ID或0，使用其绝对值
                } else {
                    lookupId = id + normalvertices.size() - 1; // 对于正ID，加上偏移量
                }
                vertexIdMap[lookupId].push_back(node->vertex[i]);
            }
        }
    }
}

std::vector<int> Octree::findNearOriginBoundaryPoints(const std::vector<std::vector<int>>& boundaryFaces, const std::vector<GLVertex*>& normalVertices,double maxDistance)


{
    std::vector<int> nearOriginIds;
    std::set<int> processedIds; // 用于避免重复处理相同的ID

    // 遍历所有边界面
    for (const auto& face : boundaryFaces) {
        // 遍历面中的每个顶点ID
        for (int id : face) {
            // 如果已经处理过这个ID，则跳过
            if (processedIds.find(id) != processedIds.end()) {
                continue;
            }

            processedIds.insert(id);

            // 直接从normalVertices数组获取顶点坐标
            if (id >= 0 && id < normalVertices.size()) {
                GLVertex* vertex = normalVertices[id];

                // 计算顶点到原点的距离
                double distance = sqrt(vertex->x * vertex->x + vertex->y * vertex->y + vertex->z * vertex->z);

                // 如果距离小于指定值，则添加到结果列表
                if (distance < maxDistance) {
                    nearOriginIds.push_back(id);
                    std::cout << "找到距离原点 " << distance << " 的点，ID: " << id
                              << "，坐标: (" << vertex->x << ", " << vertex->y << ", " << vertex->z << ")" << std::endl;
                }
            } else {
                std::cerr << "警告: ID " << id << " 超出normalVertices范围" << std::endl;
            }
        }
    }

    std::cout << "共找到 " << nearOriginIds.size() << " 个距离原点小于 " << maxDistance << " 的边界点" << std::endl;

    return nearOriginIds;
}



void Octree::export_mesh_to_file(std::vector<GLVertex*>& normalvertices, std::vector<std::vector<int>>& boundaryFaces, std::vector<GLVertex*>& hanging_vertices, const std::vector<std::vector<int>>& forceBoundaryFaceVertices ,const std::string& filename)

{
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        return;
    }

    // tital
    out << "MFEM NC mesh v1.0\n";
    out << "dimension\n";
    out << "3\n";


    // elements
    std::vector<Octnode*> leaf_node_list;
    get_leaf_nodes2(root, leaf_node_list);  // 使用 this 而不是 tree

    out << "elements\n" << leaf_node_list.size() << "\n";

    for (Octnode* node : leaf_node_list) {
        out << "0 1 5 0 ";
        int order[] = {1, 0, 3, 2, 5, 4, 7, 6};
        for (int i = 0; i < 8; ++i) {
            int idx = order[i];
            if (node->vertex[idx]->id <= 0)
                out << -node->vertex[idx]->id;
            else
                out << node->vertex[idx]->id + normalvertices.size()-1;
            if (i < 7) out << " ";
            if(node->vertex[idx]->id==-9999999)
            {std::cout<<"有问题node"<<" "<<node->depth<<" "<<node->indexs->x<<" "<<node->indexs->y<<" "<<node->indexs->z<<" "<<idx<<std::endl;}
        }
        out << "\n";
    }

    ///边界判断
//           // boundary
//           out << "boundary\n" << boundaryFaces.size() << "\n";

//           int forceCounter = 2; // 从2开始累加

//           for (const auto& face : boundaryFaces) {
//               bool isForceRelatedFace = false;

//               // 检查当前面是否在forceBoundaryFaceVertices中
//               for (const auto& forceFace : forceBoundaryFaceVertices) {
//                   if (face.size() == forceFace.size()) {
//                       // 检查两个面的顶点是否完全匹配（考虑顺序可能不同）
//                       std::vector<int> sortedFace = face;
//                       std::vector<int> sortedForceFace = forceFace;
//                       //快排
//                       std::sort(sortedFace.begin(), sortedFace.end());
//                       std::sort(sortedForceFace.begin(), sortedForceFace.end());

//                       if (sortedFace == sortedForceFace) {
//                           isForceRelatedFace = true;
//                           break;
//                       }
//                   }
//               }

//               // 根据是否为力相关面输出不同格式
//               if (isForceRelatedFace) {
//                   out << forceCounter << " 3 ";  // forceCounter在第一个位置，3在第二个位置
//                   forceCounter++; // 为下一个力相关面累加
//               } else {
//                   out << "3 3 ";
//               }

//               // 输出面顶点ID
//               for (int idx : face) {
//                   out << idx << " ";
//               }
//               out << "\n";
//           }
    // boundary
    out << "boundary\n" << boundaryFaces.size() << "\n";

    int forceCounter = 3; // 从3开始累加,1是固定表面，2是自由表面

    for (const auto& face : boundaryFaces) {
        bool isForceRelatedFace = false;
        bool isRightBoundaryFace = false;
        bool isBottomBoundaryFace=false;
        // 检查当前面是否在forceBoundaryFaceVertices中
        for (const auto& forceFace : forceBoundaryFaceVertices) {
            if (face.size() == forceFace.size()) {
                // 检查两个面的顶点是否完全匹配（考虑顺序可能不同）
                std::vector<int> sortedFace = face;
                std::vector<int> sortedForceFace = forceFace;
                //快排
                std::sort(sortedFace.begin(), sortedFace.end());
                std::sort(sortedForceFace.begin(), sortedForceFace.end());

                if (sortedFace == sortedForceFace) {
                    isForceRelatedFace = true;
                    break;
                }
            }
        }

        // 检查是否为右边界面或者底面）
        if (!isForceRelatedFace) {
            // 遍历所有叶节点，查找面所属的节点
            for (Octnode* node : leaf_node_list) {
                // 检查这个面的所有顶点是否都属于当前节点
                bool allVerticesInNode = true;
                for (int faceVertexIdx : face) {
                    bool vertexFound = false;
                    for (int i = 0; i < 8; i++) {
                        int nodeVertexId = node->vertex[i]->id;
                        // 检查顶点ID是否匹配（考虑正负ID）
                        if ((nodeVertexId <= 0 && faceVertexIdx == -nodeVertexId) ||
                            (nodeVertexId > 0 && faceVertexIdx == nodeVertexId + normalvertices.size() - 1)) {
                            vertexFound = true;
                            break;
                        }
                    }
                    if (!vertexFound) {
                        allVerticesInNode = false;
                        break;
                    }
                }

                //判断这个node是否是右、底node
                if (allVerticesInNode && node->indexs != nullptr && (node->indexs->x == 80||node->indexs->z==31)) {
                    // 检查是否是Node的右面 (右面顶点索引为0,3,7,4)
                    bool  isRightFace = true;
                    //检查是否是Node的底面(右面顶点索引为0,1,2,3)
                    bool isBottomFace=true;
                    std::vector<int> RightFaceIndices = {0, 3, 7, 4};
                    std::vector<int> BottomFaceIndices = {0,1,2,3};
                    for (int faceVertexIdx : face) {
                        bool foundInRightFace = false;

                        for (int RightIdx : RightFaceIndices) {
                            int nodeVertexId = node->vertex[RightIdx]->id;
                            if ((nodeVertexId <= 0 && faceVertexIdx == -nodeVertexId) ||
                                (nodeVertexId > 0 && faceVertexIdx == nodeVertexId + normalvertices.size() - 1)) {
                                foundInRightFace = true;
                                break;
                            }
                        }
                        if (!foundInRightFace) {
                            isRightFace = false;
                            break;
                        }
                    }
                    for (int faceVertexIdx : face) {
                        bool foundInBottomFace= false;
                        for (int BottomIdx : BottomFaceIndices) {
                              int nodeVertexId = node->vertex[BottomIdx]->id;
                              if ((nodeVertexId <= 0 && faceVertexIdx == -nodeVertexId) ||
                                  (nodeVertexId > 0 && faceVertexIdx == nodeVertexId + normalvertices.size() - 1)) {
                                  foundInBottomFace = true;
                                  break;
                              }
                          }

                        if (!foundInBottomFace) {
                               isBottomFace = false;
                               break;
                           }
                    }

                    if (isRightFace) {
                        isRightBoundaryFace = true;
                        break;
                    }
                    if(isBottomFace){
                        isBottomBoundaryFace = true;
                        break;
                    }
                }
            }
        }

        // 根据边界类型输出不同格式
        if (isForceRelatedFace) {
            out << forceCounter << " 3 ";  // 有力作用的边界按forceCounter累加
            forceCounter++; // 为下一个力相关面累加
        } else if (isBottomBoundaryFace) {
            out << "1 3 ";  // 下边界都设为属性1
        } else {
            out << "2 3 ";
        }

        // 输出面顶点ID
        for (int idx : face) {
            out << idx << " ";
        }
        out << "\n";
    }
    //        ///边界判断
    //        // boundary
    //           out << "boundary\n" << boundaryFaces.size() << "\n";
    //           for (const auto& face : boundaryFaces) {
    //               bool faceProcessed = false;

    //               // 遍历所有叶节点，查找面所属的节点
    //               for (Octnode* node : leaf_node_list) {
    //                   // 检查这个面的所有顶点是否都属于当前节点
    //                   bool allVerticesInNode = true;
    //                   for (int faceVertexIdx : face) {
    //                       bool vertexFound = false;
    //                       for (int i = 0; i < 8; i++) {
    //                           int nodeVertexId = node->vertex[i]->id;
    //                           // 检查顶点ID是否匹配（考虑正负ID）
    //                           if ((nodeVertexId <= 0 && faceVertexIdx == -nodeVertexId) ||
    //                               (nodeVertexId > 0 && faceVertexIdx == nodeVertexId + normalvertices.size() - 1)) {
    //                               vertexFound = true;
    //                               break;
    //                           }
    //                       }
    //                       if (!vertexFound) {
    //                           allVerticesInNode = false;
    //                           break;
    //                       }
    //                   }

    //                   // 如果所有顶点都在这个节点中，检查index->z的值
    //                   if (allVerticesInNode && node->indexs != nullptr) {
    //                       // 检查是否是右面 (底面顶点索引为0,1,2,3)
    //                       bool isBottomFace = true;
    //                       std::vector<int> bottomFaceIndices = {0, 3, 7, 4};
    //                       for (int faceVertexIdx : face) {
    //                           bool foundInBottomFace = false;
    //                           for (int bottomIdx : bottomFaceIndices) {
    //                               int nodeVertexId = node->vertex[bottomIdx]->id;
    //                               if ((nodeVertexId <= 0 && faceVertexIdx == -nodeVertexId) ||
    //                                   (nodeVertexId > 0 && faceVertexIdx == nodeVertexId + normalvertices.size() - 1)) {
    //                                   foundInBottomFace = true;
    //                                   break;
    //                               }
    //                           }
    //                           if (!foundInBottomFace) {
    //                               isBottomFace = false;
    //                               break;
    //                           }
    //                       }

    //                       // 检查是否是左面 (顶面顶点索引为4,5,6,7)
    //                       bool isTopFace = true;
    //                       std::vector<int> topFaceIndices = {1, 2, 6, 5};
    //                       for (int faceVertexIdx : face) {
    //                           bool foundInTopFace = false;
    //                           for (int topIdx : topFaceIndices) {
    //                               int nodeVertexId = node->vertex[topIdx]->id;
    //                               if ((nodeVertexId <= 0 && faceVertexIdx == -nodeVertexId) ||
    //                                   (nodeVertexId > 0 && faceVertexIdx == nodeVertexId + normalvertices.size() - 1)) {
    //                                   foundInTopFace = true;
    //                                   break;
    //                               }
    //                           }
    //                           if (!foundInTopFace) {
    //                               isTopFace = false;
    //                               break;
    //                           }
    //                       }

    //                       // 根据条件输出不同的标识符
    //                       if (node->indexs->x == 170 && isBottomFace) {//右面
    //                           out << "1 3 ";
    //                       } else if (node->indexs->x == 133 && isTopFace) {//左面
    //                           out << "2 3 ";
    //                       } else {
    //                           out << "3 3 ";
    //                       }

    //                       // 输出面的顶点ID
    //                       for (int idx : face) {
    //                           out << idx << " ";
    //                       }
    //                       out << "\n";

    //                       faceProcessed = true;
    //                       break;
    //                   }
    //               }

    //               // 如果没有找到匹配的节点，使用默认输出
    //               if (!faceProcessed) {
    //                   out << "3 3 ";
    //                   for (int idx : face) {
    //                       out << idx << " ";
    //                   }
    //                   out << "\n";
    //               }
    //           }

    // vertex_parents
    out << "vertex_parents\n" << hanging_vertices.size() << "\n";
    for (GLVertex* vertex : hanging_vertices) {
        out << vertex->id + normalvertices.size() - 1 << " ";
        for (int j = 0; j < 2; ++j) {
            if (vertex->parentVertices[j]<= 0)//正常顶点是0，-1，-2。。。
                out << -vertex->parentVertices[j];
            else
                out << vertex->parentVertices[j] + normalvertices.size()-1;//悬挂顶点是1.2.3
            if (j == 0) out << " ";
        }
        out << "\n";
    }

    // coordinates
    out << "coordinates\n";
    out << normalvertices.size() << "\n";
    out << "3\n";
    for (GLVertex* vertex : normalvertices) {
        out <<vertex->x << " " << vertex->y << " " << vertex->z << "\n";
    }

    // end
    out << "mfem_mesh_end\n";
    out.close();
    std::cout << "输出完成: " << filename << std::endl;
}
}




///<added hust

