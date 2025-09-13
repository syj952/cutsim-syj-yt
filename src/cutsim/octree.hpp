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

#ifndef OCTREE_H
#define OCTREE_H

#include <iostream>
#include <list>
#include <cassert>
///<added hust
#include <vector>
#include<unordered_map>
#include<map>
///<added hust
#include "bbox.hpp"
#include "gldata.hpp"
#include "octnode.hpp"
//#include "marching_cubes.hpp"

#ifdef MULTI_THREAD
#include <QtCore>
#include <QFuture>
#endif

#include <QtConcurrent>

namespace cutsim {

class Octnode;
class Volume;

/// Octree class for cutting simulation
/// see http://en.wikipedia.org/wiki/Octree
/// The root node is divided into eight sub-octants, and each sub-octant
/// is recursively further divided into octants.
/// The side-length of the root node is root_scale
/// The dept of the root node is zero.
/// Subdivision is continued unti max_depth is reached.
/// A node at tree-dept n is a cube with side-length root_scale/pow(2,n)
///
/// This class stores the root Octnode and allows operations on the tree
///
class Octree {

    public:

///added hust
        void cuda_diff(const CylCutterVolume* vol){ cuda_diff( this->root, vol); }
        void cuda_sum(const Volume* vol,double max_depth){ cuda_sum( this->root, vol, max_depth); }
        void blade_diff(const Volume* vol) { blade_diff( this->root, vol); }
        void cuda_blade_diff(const AptCutterVolume* vol){ cuda_blade_diff( this->root, vol); };
        /// create an octree with a root node with scale=root_scale, maximum
        /// tree-depth of max_depth and centered at centerp.
        Octree(double root_scale, unsigned int max_depth, GLVertex* centerPoint, GLData* gl);
        virtual ~Octree();

    // bolean operations on tree
        /// sum given Volume to tree
//        void sum(const Volume* vol) { sum( this->root, vol); }
        void sum(Volume* vol) { sum( this->root, vol); }
        /// diff given Volume from tree
        void diff(const Volume* vol) { 
                   engaged =false; /// added hust
            cwenodelist.clear();///added hust更新接触Node
             diff( this->root, vol);
         }
        /// intersect tree with given Volume
        void intersect(const Volume* vol) { intersect( this->root, vol); }
        /// diff given Volume from tree for cuttings
        CuttingStatus diff_c(const Volume* vol) { return diff_c( this->root, vol); }
        /// check f-values of tree and correct the result
        bool check_node(const Volume* vol) { return check_node(this->root, vol); }

// debug, can be removed?
        /// put all leaf-nodes in a list
        void get_leaf_nodes( std::vector<Octnode*>& nodelist) const { get_leaf_nodes( root,  nodelist); }
        /// put all leaf-nodes in a list
        void get_leaf_nodes(Octnode* current, std::vector<Octnode*>& nodelist) const;
        /// put all invalid nodes in a list
        void get_invalid_leaf_nodes(std::vector<Octnode*>& nodelist) const;
        /// put all invalid nodes in a list
        void get_invalid_leaf_nodes( Octnode* current, std::vector<Octnode*>& nodelist) const;
        /// put all nodes in a list
        void get_all_nodes(Octnode* current, std::vector<Octnode*>& nodelist) const;

        /// initialize by recursively calling subdivide() on all nodes n times
        void init(const unsigned int n);
        /// return max depth
        unsigned int get_max_depth() const;
        /// return the maximum cube side-length, (i.e. at depth=0)
        double get_root_scale() const;
        /// return the minimum cube side-length (i.e. at maximum depth)
        double leaf_scale() const;
        /// string output
        std::string str() const;

void treeTransfer(GLVertex parallel, int flip_axis = 0, bool ignore_parts = false) { treeTransfer(this->root, parallel, flip_axis, ignore_parts); }
void setInvalid(void) { setInvalid(this->root); }

void clearTree(double root_scale, unsigned int max_depth, GLVertex* centerPoint);

        /// flag for debug mode
        bool debug;
        /// flag for debug-mode of marching-cubes
        bool debug_mc;
        /// the root scale, i.e. side-length of depth=0 cube
        double root_scale;
        /// the maximum tree-depth
        unsigned int max_depth;
        /// pointer to the root node
        Octnode* root;
        /// the GLData used to draw this tree
        GLData* g;
 ///< added hust for calculate_index
        struct HangingVertexInfo {
            Octnode* target_node;
            float target_x;
            float target_y;
            float target_z;

            HangingVertexInfo(Octnode* node = nullptr, float x = 0.0f, float y = 0.0f, float z = 0.0f)
                : target_node(node), target_x(x), target_y(y), target_z(z) {}
        };

        //储存父顶点信息
        struct VertexPositionInfo {
            std::string position_type; // "顶点", "边中点", "面中点", "体中心点"
            int vertex1_id;
            int vertex2_id;

            VertexPositionInfo(std::string type = "未知位置", int id1 = -1, int id2 = -1)
                : position_type(type), vertex1_id(id1), vertex2_id(id2) {}
        };

        void get_leaf_nodes2(Octnode* current, std::vector<Octnode*>& nodelist) const;
        int calculate_index(int remx, int remy, int remz);
         std::vector<std::vector<int>> find_parent_nodes(int deep, int idxx, int idxy, int idxz);
         Octnode* find_target_node(Octnode* root, int deep, int idxx, int idxy, int idxz);        
          bool exist(Octnode* root, int deep, int idxx, int idxy, int idxz);
         void get_hangingVertex(std::vector< GLVertex*>& hangingVertexList,std::vector< GLVertex*>& normalVertexList);
         void get_hanging_vertex_parent();
          bool check_hanging_vertex(Octnode* root, int deep, int idxx, int idxy, int idxz, int index, int i);
          void remove_duplicate_vertex(Octnode* node, Octnode* root, int vertex_idx);
          void transfer_vertex_id_upward(Octnode* node, Octnode* root, int vertex_id, int i);
          HangingVertexInfo find_hanging_vertex_parent_node(Octnode* node, int vertex_idx);
          void collectNonLeafNodes(Octnode* node, std::vector<Octnode*>& result) ;
          void boundary(std::vector<std::vector<int>>& boundaryFaces, std::vector<GLVertex *>&normalvertices,std::vector<Octnode*>& boundarynode);
          void getForceBoundaryFace(const std::vector<Octnode*>& nearestNodes, const std::vector<std::vector<int>>& boundaryFaces,std::vector<std::vector<int>>& forceBoundaryFaceVertices,std::vector<GLVertex*>& normalvertices);
          void transfer_vertex_id(Octnode* node, Octnode* root, int vertex_id, int i);
         void export_mesh_to_file(std::vector<GLVertex*>& normalvertices, std::vector<std::vector<int>>& boundaryFaces, std::vector<GLVertex*>& hanging_vertices,  const std::vector<std::vector<int>>& forceBoundaryFaceVertices,const std::string& filename);
         VertexPositionInfo determine_hanging_vertex_position(const HangingVertexInfo& info);
            GLVertex *findVerticesById(int id);//通过ID查找顶点
             void buildVertexIdMap(std::vector<GLVertex*>& normalvertices);//构建ID映射表
              std::unordered_map<int, std::vector<GLVertex*>> vertexIdMap;
              std::vector<int> findNearOriginBoundaryPoints(const std::vector<std::vector<int>>& boundaryFaces,const std::vector<GLVertex*>& normalVertices,double maxDistance) ;
              void updateDeformationColors();
                               void updateDeformationColors(const std::vector<Octnode*>& nodeList);
         ///< added hust
    protected:
        ///added hust
        void cuda_diff(Octnode* current, const CylCutterVolume* vol);
        void cuda_sum(Octnode* current, const Volume* vol,double max_depth);
        void blade_diff(Octnode* current, const Volume* vol);
        void cuda_blade_diff(Octnode* current, const AptCutterVolume* vol);
        /// recursively traverse the tree subtracting Volume
        void diff(Octnode* current, const Volume* vol);
        /// union Octnode with Volume
//        void sum(Octnode* current, const Volume* vol);
        void sum(Octnode* current, Volume* vol);
        /// intersect Octnode with Volume
        void intersect(Octnode* current, const Volume* vol);
        // diff (intersection with volume's compliment) of tree and Volume for cuttings
        CuttingStatus diff_c(Octnode* current, const Volume* vol);
        /// check f-values of Octnode and correct the result
        bool check_node(Octnode* current, const Volume* vol);


void treeTransfer(Octnode* current, GLVertex parallel, int flip_axis = 0, bool ignore_parts = false);
void setInvalid(Octnode* current);
void clearNode(Octnode* current);
        // DATA
        /// the GLData used to draw this tree
//        GLData* g;

    private:
std::vector<int> processed;
  public:
bool engaged =false;///< added hust
std::vector<Octnode *>cwenodelist;///< added hust cwe node list
};

} // end namespace
#endif
// end file octree.hpp
