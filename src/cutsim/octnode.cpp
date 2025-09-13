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

#include <boost/foreach.hpp>

#include <src/app/cutsim_def.hpp>

#include "octnode.hpp"

namespace cutsim {

//**************** Octnode ********************/

// this defines the position of each octree-vertex with relation to the center of the node
// this also determines in which direction the center of a child node is
///<added hust
const GLVertex Octnode::direction[8] = {
    GLVertex( 1, -1,-1),   // 0
      GLVertex(-1, -1,-1),   // 1
      GLVertex(-1,1,-1),   // 2
      GLVertex( 1,1,-1),   // 3
      GLVertex( 1, -1, 1),   // 4
      GLVertex(-1, -1, 1),   // 5
      GLVertex(-1,1, 1),   // 6
      GLVertex( 1,1, 1)    // 7
     };
             

// surface enumeration
// surf     vertices  vertices
// 0:       2,3,7     2,6,7
// 1:       0,4,7     0,3,7
// 2:       0,1,4     1,4,5
// 3:       1,5,6     1,2,6
// 4:       0,2,3     0,1,2
// 5:       4,6,7     4,5,6

// bit-mask for setting valid() property of child-nodes
//const char Octnode::octant[8] = {
const unsigned char Octnode::octant[8] = {
                    0x1,
                    0x2,
                    0x4,
                    0x8,
                    0x10,
                    0x20,
                    0x40,
                    0x80
                };

unsigned int alocation_count = 0;
unsigned int delete_count = 0;
unsigned int delete_childlen_count = 0;


#ifdef MULTI_THREAD
	static QMutex nodePoolMutex;
#endif

// For child nodes
Octnode::Octnode(Octnode* nodeparent, unsigned int index, double nodescale, unsigned int nodedepth, GLData* gl) {
    parent = nodeparent;
    idx = index;
    scale = nodescale;
    depth = nodedepth;
    g = gl;
    ///added hust
    currentcutstate = UNCUTTING;
    if (parent) {
        center = parent->childcenter(idx);
        state = parent->prev_state;//OUtside但是应该重新判断
        prev_state = state;//undecide
        color = parent->color;
        indexs = new GLVertex(0, 0, 0);
        for (int n = 0; n < 8; ++n) {
                   vindexs[n] = new GLVertex(0, 0, 0);}
    } else { //node has no parent
		assert( parent == NULL );
    }

    for ( int n = 0; n < 8; ++n) {
        child[n] = NULL;
        vertex[n] = new GLVertex(*center + direction[n] * scale) ;//这是对八个顶点进行初始化
        deformedVertex[n] = *vertex[n];
        ///< added hust
        vertexnotsaved[n] = 1;//顶点遍历状态
        vertex[n]->id=-9999999;//对顶点ID进行初始化，方便以后查找
        //子结点顶点坐标变换
        switch (idx) {
        case 0:
            vindexs[n]->x = parent->vindexs[0]->x*2 + ((direction[n].x < 0) ? -1 : 0);
                  vindexs[n]->y = parent->vindexs[0]->y*2 + ((direction[n].y > 0) ? 1 : 0);
                  vindexs[n]->z = parent->vindexs[0]->z*2 + ((direction[n].z > 0) ? 1 : 0);
            break;

        case 1:
            vindexs[n]->x = parent->vindexs[1]->x*2 + ((direction[n].x > 0) ? 1 : 0);
                    vindexs[n]->y = parent->vindexs[1]->y*2 + ((direction[n].y > 0) ? 1 : 0);
                    vindexs[n]->z = parent->vindexs[1]->z*2 + ((direction[n].z > 0) ? 1 : 0);
            break;
        case 2:
            vindexs[n]->x = parent->vindexs[2]->x*2 + ((direction[n].x > 0) ? 1 : 0);
                       vindexs[n]->y = parent->vindexs[2]->y*2 + ((direction[n].y > 0) ? 0 : -1);
                       vindexs[n]->z = parent->vindexs[2]->z*2 + ((direction[n].z > 0) ? 1 : 0);
            break;
        case 3:
            vindexs[n]->x = parent->vindexs[3]->x*2 + ((direction[n].x > 0) ? 0 : -1);
                       vindexs[n]->y = parent->vindexs[3]->y*2 + ((direction[n].y > 0) ? 0 : -1);
                       vindexs[n]->z = parent->vindexs[3]->z*2 + ((direction[n].z > 0) ? 1 : 0);
            break;
        case 4:
            vindexs[n]->x=parent->vindexs[4]->x*2+((direction[n].x > 0) ? 0 : -1);
            vindexs[n]->y=parent->vindexs[4]->y*2+((direction[n].y > 0) ? 1 : 0);
            vindexs[n]->z=parent->vindexs[4]->z*2+((direction[n].z > 0) ? 0 : -1);
            break;
        case 5:
                  vindexs[n]->x = parent->vindexs[5]->x*2 + ((direction[n].x > 0) ? 1 : 0);
                  vindexs[n]->y = parent->vindexs[5]->y*2 + ((direction[n].y > 0) ? 1 : 0);
                  vindexs[n]->z = parent->vindexs[5]->z*2 + ((direction[n].z > 0) ? 0 : -1);
                  break;
              case 6:
                  vindexs[n]->x = parent->vindexs[6]->x*2 + ((direction[n].x > 0) ? 1 : 0);
                  vindexs[n]->y = parent->vindexs[6]->y*2 + ((direction[n].y > 0) ? 0 : -1);
                  vindexs[n]->z = parent->vindexs[6]->z*2 + ((direction[n].z > 0) ? 0 : -1);
                  break;
              case 7:
                  vindexs[n]->x = parent->vindexs[7]->x*2 + ((direction[n].x > 0) ? 0 : -1);
                  vindexs[n]->y = parent->vindexs[7]->y*2 + ((direction[n].y > 0) ? 0 : -1);
                  vindexs[n]->z = parent->vindexs[7]->z*2 + ((direction[n].z > 0) ? 0 : -1);
            break;

        }

        ///< added hust
            assert( parent->state == UNDECIDED );
            assert( parent->prev_state != UNDECIDED );//父亲的状态由非undecided变成undecided
            //std::cout << parent->prev_state << "\n";
            if (parent->prev_state == INSIDE) {//子结点的state继承父节点的prev_state
                f[n] = 1.0;
                state = INSIDE;
            }
            else if (parent->prev_state == OUTSIDE) {
                f[n] = -1.0;
                state = OUTSIDE;
            }
            else
                assert(0);

             //f[n]= parent->f[n];  // why does this make a big diggerence in the speed of sum() and dif() ??
             // sum() sum(): 0.15s + 0.27s   compared to 1.18 + 0.47
             // sum() (): 0.15 + 0.2     compared to 1.2 + 0.46
    }
    bb.clear();
#ifdef MULTI_AXIS
// Multi Axis
    bb.addPoint(*center + GLVertex(-2.0,-2.0,-2.0) * scale); // caluclate the minimum x,y,z coordinates
    bb.addPoint(*center + GLVertex( 2.0, 2.0, 2.0) * scale); // caluclate the maximum x,y,z coordinates
#else
    bb.addPoint( *vertex[2] ); // vertex[2] has the minimum x,y,z coordinates
    bb.addPoint( *vertex[4] ); // vertex[4] has the max x,y,z
#endif
    isosurface_valid = false;

    childcount = 0;
    childStatus = 0;
alocation_count++;
}

// For Root node
Octnode::Octnode(GLVertex* root_center, double nodescale, GLData* gl) {
    parent = NULL;
    idx = 0;
    scale = nodescale;
    depth = 0;
    g = gl;

    center = root_center;
    state = UNDECIDED;//根节点的state
    prev_state = OUTSIDE;
   ///<added hust
   vindexs[0]=new GLVertex(1, 0, 0);
   vindexs[1]=new GLVertex(0, 0, 0);
   vindexs[2]=new GLVertex(0, 1, 0);
   vindexs[3]=new GLVertex(1, 1, 0);
   vindexs[4]=new GLVertex(1, 0, 1);
   vindexs[5]=new GLVertex(0, 0, 1);
   vindexs[6]=new GLVertex(0, 1, 1);
   vindexs[7]=new GLVertex(1, 1, 1);
indexs = new GLVertex(0, 0, 0);//在根节点初始化indexs
///<added hust
    for ( int n = 0; n < 8; ++n) {
        child[n] = NULL;
        vertex[n] = new GLVertex(*center + direction[n] * scale) ;
        f[n] = -1;
        ///< added hust
        vertexnotsaved[n] = 1;//在根节点初始化
        ///< added hust
    }
    bb.clear();
#ifdef MULTI_AXIS
// Multi Axis
    bb.addPoint(*center + GLVertex(-2.0,-2.0,-2.0) * scale); // caluclate the minimum x,y,z coordinates
    bb.addPoint(*center + GLVertex( 2.0, 2.0, 2.0) * scale); // caluclate the maximum x,y,z coordinates
#else
    bb.addPoint( *vertex[2] ); // vertex[2] has the minimum x,y,z coordinates
    bb.addPoint( *vertex[4] ); // vertex[4] has the max x,y,z
#endif
    isosurface_valid = false;

    childcount = 0;
    childStatus = 0;

    alocation_count++;
}

// call delete on children, vertices, and center
Octnode::~Octnode() {
    if (childcount != 0 ) {
        for(int n = 0; n < 8; ++n) {
            if (child[n] != 0) {
        		if (child[n]->childcount != 0) {
//std::cout << "\nchild[n]->childcount : " << child[n]->childcount << "\n";
                    for (int m = 0; m < 8; ++m)
        				if (child[n]->child[m] != 0) {
        					child[n]->child[m]->clearVertexSet();
        					delete child[n]->child[m];
        				}
        			child[n]->childcount = 0;
        		}
        		assert( child[n]->childcount == 0);
        		delete child[n];
        		child[n] = 0;
        		delete vertex[n];
        		vertex[n] = 0;
        	}
        }
    }
    delete center;
    center = 0;

    delete_count++;
}

// return centerpoint of child with index n by pointer
GLVertex* Octnode::childcenter(int n) {
    return  new GLVertex(*center + ( direction[n] * 0.5 * scale ));
}

// return centerpoint of child with index n by value
GLVertex Octnode::childcenterValue(int n) {
    return  *center + ( direction[n] * 0.5*scale );
}

// create the 8 children of this node
void Octnode::subdivide() {
    if (this->childcount == 0) {
        if( state != UNDECIDED )
            std::cout << " subdivide() error: state==" << state << "\n";

        assert( state == UNDECIDED );
        for( int n = 0; n < 8; ++n ) {
//#ifdef POOL_NODE///是一个宏性能优化算法，但其目的都是一样的
//        	Octnode* newnode = createOctnode( this, n , scale*0.5 , depth+1 , g); // parent,  idx, scale,   depth, GLdata
//#else
        	Octnode* newnode = new Octnode( this, n , scale*0.5 , depth+1 , g); // parent,  idx, scale,   depth, GLdata
//#endif
            ///< added hust
            //newnode->indexs = new GLVertex(0,0,0);
            newnode->indexs->x = this->indexs->x*2 + int(direction[n].x>0);
            newnode->indexs->y = this->indexs->y*2 + int(direction[n].y>0);
            newnode->indexs->z = this->indexs->z*2 + int(direction[n].z>0);
            child[n] = newnode;
            ++childcount;
        }
    } else {
        std::cout << " DON'T subdivide a non-leaf node \n";
        assert(0);
    }
}

void Octnode::sum(const Volume* vol) {
    double d;
    for (int n = 0; n < 8; ++n) {
        if ((d = vol->dist(*(vertex[n]))) > f[n]) {
            f[n] = d;
            color = vol->color;
        }
    }
//    set_state();
}

void Octnode::diff(const Volume* vol) {
    currentcutstate=UNCUTTING;///added hust
    double d;
    for (int n = 0; n < 8; ++n)  {
        if ((d = -vol->dist(deformedVertex[n])) < f[n]) {
            f[n] = d;
       //     color = vol->color;
    currentcutstate =CUTTING;///added hust
        }
    }
//   set_state();
}

void Octnode::intersect(const Volume* vol) {
    double d;
    for (int n = 0; n < 8; ++n) {
        if ((d = vol->dist(*(vertex[n]))) < f[n]) {
            f[n] = d;
            color = vol->color;
        }
    }
//    set_state();
}

CuttingStatus Octnode::diff_cd(const Volume* vol) {
    Cutting r;
    CuttingStatus status = { 0, NO_COLLISION };
    for (int n = 0; n < 8; ++n)  {
        r = ((CutterVolume*)vol)->dist_cd(*(vertex[n]));
        if (-r.f < f[n]) {
            f[n] = -r.f;
            status.collision |= r.collision;
            if (color.isGray())
                status.collision |= PARTS_COLLISION;
            status.cutcount += r.count;
            color = vol->color;
        }
    }
//    set_state();

    if (status.collision) color.set(COLLISION_COLOR);

    return status;
}

// look at the f-values in the corner of the cube and
// check it within the limit
bool Octnode::check_f_value_with_limit() {
	double limit = scale*(4.0 + TOLERANCE);
	if ((fabs(f[0]) > limit) || (fabs(f[1]) > limit) || (fabs(f[2]) > limit) || (fabs(f[3]) > limit) ||
	    (fabs(f[4]) > limit) || (fabs(f[5]) > limit) || (fabs(f[6]) > limit) || (fabs(f[7]) > limit))
			return false;
	else {
//		limit = scale*(2.0 + TOLERANCE);
		limit = scale*(2.2 + TOLERANCE);
		if (f[0] < 0.0) {
			if ((f[1] >= 0.0 && (f[1] - f[0]) > limit) || (f[3] >= 0.0 && (f[3] - f[0]) > limit) || (f[4] >= 0.0 && (f[4] - f[0]) > limit))
				return false;
		} else if (f[1] < 0.0) {
			if ((f[0] >= 0.0 && (f[0] - f[1]) > limit) || (f[2] >= 0.0 && (f[2] - f[1]) > limit) || (f[5] >= 0.0 && (f[5] - f[1]) > limit))
				return false;
		} else if (f[2] < 0.0) {
			if ((f[1] >= 0.0 && (f[1] - f[2]) > limit) || (f[3] >= 0.0 && (f[3] - f[2]) > limit) || (f[6] >= 0.0 && (f[6] - f[2]) > limit))
				return false;
		} else if (f[3] < 0.0) {
			if ((f[0] >= 0.0 && (f[0] - f[3]) > limit) || (f[2] >= 0.0 && (f[2] - f[3]) > limit) || (f[7] >= 0.0 && (f[7] - f[3]) > limit))
				return false;
		} else if (f[4] < 0.0) {
			if ((f[5] >= 0.0 && (f[5] - f[4]) > limit) || (f[7] >= 0.0 && (f[7] - f[4]) > limit) || (f[0] >= 0.0 && (f[0] - f[4]) > limit))
				return false;
		} else if (f[5] < 0.0) {
			if ((f[4] >= 0.0 && (f[4] - f[5]) > limit) || (f[6] >= 0.0 && (f[6] - f[5]) > limit) || (f[1] >= 0.0 && (f[1] - f[5]) > limit))
				return false;
		} else if (f[6] < 0.0) {
			if ((f[5] >= 0.0 && (f[5] - f[6]) > limit) || (f[7] >= 0.0 && (f[7] - f[6]) > limit) || (f[2] >= 0.0 && (f[2] - f[6]) > limit))
				return false;
		} else if (f[7] < 0.0) {
			if ((f[6] >= 0.0 && (f[6] - f[7]) > limit) || (f[4] >= 0.0 && (f[4] - f[7]) > limit) || (f[3] >= 0.0 && (f[3] - f[7]) > limit))
				return false;
		}
	}
	return true;
}

// look at the f-values in the corner of the cube and set state
// to inside, outside、when those are obviously decided
bool Octnode::check_complete_inside_outside() {
    bool inside = true;
    bool outside = true;
    double limit = scale*4.0;
    for ( int n = 0; n < 8; n++) {
        if (f[n] <= limit)   // if one vertex is not far inside
            inside = false; // then it may not be an inside-node
        if (-limit <= f[n])  // if one vertex is not far outside
            outside = false; // then it may not be an outside-node
    }
    assert( !( outside && inside) ); // sanity check
    if (inside)
	setInside();
    else if (outside)
	setOutside();

    return inside || outside;
}

// check whether the node has no undecided child or not
bool Octnode::check_include_undecided() {
    if ( childcount == 8 ) {
        return ( child[0]->state == UNDECIDED ) ||
               ( child[1]->state == UNDECIDED ) ||
               ( child[2]->state == UNDECIDED ) ||
               ( child[3]->state == UNDECIDED ) ||
               ( child[4]->state == UNDECIDED ) ||
               ( child[5]->state == UNDECIDED ) ||
               ( child[6]->state == UNDECIDED ) ||
               ( child[7]->state == UNDECIDED ) ;
    } else {
        return false;
    }
}

// look at the current f-values in the corner of the cube and check out
// it's inside, outside, or undecided in spite of child's state
int Octnode::check_node_state() {
    bool outside = true;
    bool inside = true;
    for ( int n = 0; n < 8; n++) {
        if ( f[n] >= 0.0 ) {// if one vertex is inside
            outside = false; // then it's not an outside-node
        } else { // if one vertex is outside
            inside = false; // then it's not an inside node anymore
        }
    }
    assert( !( outside && inside) ); // sanity check

    return inside ? INSIDE : outside ? OUTSIDE : UNDECIDED;
}

// look at the f-values in the corner of the cube and set state
// to inside, outside, or undecided
bool Octnode::set_state() {
    NodeState old_state = state;
    bool outside = true;
    bool inside = true;
    for ( int n = 0; n < 8; n++) {
        if ( f[n] >= 0.0 ) {// if one vertex is inside
            outside = false; // then it's not an outside-node
        } else { // if one vertex is outside
            inside = false; // then it's not an inside node anymore
        }
    }
    assert( !( outside && inside) ); // sanity check

    if ( (inside) && (!outside) )
        setInside();
    else if ( (outside) && (!inside) )
        setOutside();
    else if ( (!inside) && (!outside) )
        setUndecided();
    else
        assert(0);

    if ( ((old_state == INSIDE) && (state == INSIDE)) ||
        ((old_state == OUTSIDE) && (state == OUTSIDE)) ) {
        // do nothing if state did not change
    	return false;
    } else {
        setInvalid();
        return true;
    }
}

void Octnode::setInside() {
    if ( (state != INSIDE) && ( all_child_state(INSIDE)   ) ) {
        state = INSIDE;
        if (parent && ( parent->state != INSIDE) )
            parent->setInside();
    }
}

void Octnode::setOutside() {
    if ( (state != OUTSIDE) && ( all_child_state(OUTSIDE)   ) )  {
        state = OUTSIDE;
        if (parent && ( parent->state != OUTSIDE ) )
            parent->setOutside();
    }
}

void Octnode::setUndecided() {
    if (state != UNDECIDED) {
        prev_state = state;
        state = UNDECIDED;
    }
}

bool Octnode::all_child_state(NodeState s) const {
    if ( childcount == 8 ) {
        return ( child[0]->state == s ) &&
               ( child[1]->state == s ) &&
               ( child[2]->state == s ) &&
               ( child[3]->state == s ) &&
               ( child[4]->state == s ) &&
               ( child[5]->state == s ) &&
               ( child[6]->state == s ) &&
               ( child[7]->state == s ) ;
    } else {
        return true;
    }
}

void Octnode::delete_children() {
    if (childcount == 8) {
        NodeState s0 = child[0]->state;
        for (int n = 0; n < 8; n++) {
//            if ( s0 != child[n]->state ) {
//                std::cout << " delete_children() error: ";
//                std::cout << "\n";
//                std::cout << " s0= " << s0 << " \n";
//            }
            assert( s0 == child[n]->state );
            child[n]->clearVertexSet();
#ifdef POOL_NODE
            deleteOctnode(child[n]);
#else
            delete child[n];
#endif
            child[n] = 0;
            childcount--;
        }
        assert( childcount == 0);
        delete_childlen_count++;
    }
}

void Octnode::force_delete_children() {
    if (childcount == 8) {
        for (int n = 0; n < 8; n++) {
            child[n]->clearVertexSet();
#ifdef POOL_NODE
            deleteOctnode(child[n]);
#else
            delete child[n];
#endif
            child[n] = 0;
            childcount--;
        }
        assert( childcount == 0);
        delete_childlen_count++;
    }
}


void Octnode::setValid() {
    isosurface_valid = true;
    //std::cout << spaces() << depth << ":" << idx << " setValid()\n";
    if (parent)
        parent->setChildValid( idx ); // try to propagate valid up the tree:
}

void Octnode::setChildValid( unsigned int id ) {
    childStatus |= octant[id]; // OR with mask
    if (childStatus == 255) { // all children valid...
        setValid(); // ...so this valid
    }
}

void Octnode::setChildInvalid( unsigned int id ) {
    childStatus &= ~octant[id]; // AND with not-mask
    setInvalid();
}

void Octnode::setInvalid() {
    isosurface_valid = false;
    if ( parent && parent->valid() )  {// update parent status also
        parent->setChildInvalid(idx);
    }
}

bool Octnode::valid() const {
    return isosurface_valid;
}

void Octnode::addIndex(unsigned int id) {
#ifndef NDEBUG
    std::set<unsigned int>::iterator found = vertexSet.find( id );
    assert( found == vertexSet.end() ); // we should not have id
#endif
    vertexSet.insert(id);
}

void Octnode::swapIndex(unsigned int oldId, unsigned int newId) {
#ifndef NDEBUG
    std::set<unsigned int>::iterator found = vertexSet.find(oldId);
    assert( found != vertexSet.end() ); // we must have oldId
#endif
    vertexSet.erase(oldId);
    vertexSet.insert(newId);
}

void Octnode::removeIndex(unsigned int id) {
#ifndef NDEBUG
    std::set<unsigned int>::iterator found = vertexSet.find( id );
    assert( found != vertexSet.end() ); // we must have id
#endif
    vertexSet.erase(id);
}

void Octnode::clearVertexSet( ) {
    while( !vertexSetEmpty() ) {
        unsigned int delId = vertexSetTop();
        removeIndex( delId );
#ifdef MULTI_THREAD
        g->workMutex.lock();
#endif
        g->removeVertex( delId );
#ifdef MULTI_THREAD
        g->workMutex.unlock();
#endif
    }
    assert( vertexSetEmpty() ); // when done, set should be empty
}

// string repr
std::ostream& operator<<(std::ostream &stream, const Octnode &n) {
    stream << " node "; //c=" << *(n.center) << " depth=" << n.depth ;
    return stream;
}

std::string Octnode::printF() {
    std::ostringstream o;
    for (int n = 0; n < 8; n++) {
        o << "f[" << n <<"] = " << f[n] << "\n";
    }
    return o.str();
}

std::string Octnode::spaces() const {
    std::ostringstream stream;
    for (unsigned int m = 0; m < this->depth; m++)
        stream << " ";
    return stream.str();
}

std::string Octnode::type() const {
    std::ostringstream stream;
    if (state == INSIDE)
        stream << "inside";
    else if (state == OUTSIDE)
        stream << "outside";
    else if (state == UNDECIDED)
        stream << "undecided";
    else
        assert(0);
    return stream.str();
}

// string repr
std::string Octnode::str() const {
    std::ostringstream o;
    o << *this;
    return o.str();
}

void Octnode::nodeTransfer(GLVertex parallel, int flip_axis, bool ignore_parts) {
	if ((ignore_parts == true) && color.isGray())
		return;

	*center += parallel;
    for (int n = 0; n < 8; ++n)
		*vertex[n] += parallel;

	switch (flip_axis) {
	case X_AXIS: // flip axis X
		center->y = -(center->y);
		center->z = -(center->z);
        for (int n = 0; n < 8; ++n) {
			vertex[n]->y = -(vertex[n]->y);
			vertex[n]->z = -(vertex[n]->z);
		}
		break;
	case Y_AXIS: // flip axis Y
		center->z = -(center->z);
		center->x = -(center->x);
        for (int n = 0; n < 8; ++n) {
			vertex[n]->z = -(vertex[n]->z);
			vertex[n]->x = -(vertex[n]->x);
		}
		break;
	case Z_AXIS: // flip axis Z
		center->x = -(center->x);
		center->y = -(center->y);
        for (int n = 0; n < 8; ++n) {
			vertex[n]->x = -(vertex[n]->x);
			vertex[n]->y = -(vertex[n]->y);
		}
		break;
	default: break;
	}
	bb.clear();
#ifdef MULTI_AXIS
// Multi Axis
	 bb.addPoint(*center + GLVertex(-2.0,-2.0,-2.0) * scale); // calculate the minimum x,y,z coordinates
	 bb.addPoint(*center + GLVertex( 2.0, 2.0, 2.0) * scale); // calculate the maximum x,y,z coordinates
#else
	 bb.addPoint( *vertex[2] ); // vertex[2] has the minimum x,y,z coordinates
	 bb.addPoint( *vertex[4] ); // vertex[4] has the max x,y,z
#endif
	 if (is_undecided() && isLeaf() && valid())
		 setInvalid();
}

#ifdef POOL_NODE

std::vector<Octnode*> nodePool;

Octnode* Octnode::createOctnode(Octnode* nodeparent, unsigned int index, double nodescale, unsigned int nodedepth, GLData* gl)
{
	Octnode* node;

#ifdef MULTI_THREAD
	nodePoolMutex.lock();
#endif

	if (nodePool.size() != 0) {
		node = nodePool[nodePool.size()-1];
		nodePool.resize(nodePool.size()-1);
#ifdef MULTI_THREAD
		nodePoolMutex.unlock();
#endif
		node->parent = nodeparent;
		node->idx = index;
		node->scale = nodescale;
		node->depth = nodedepth;
		node->g = gl;
	    if (node->parent) {
			*node->center = node->parent->childcenterValue(node->idx);
			node->state = node->parent->prev_state;
			node->prev_state = node->state;
			node->color = node->parent->color;
	    } else { // node has no parent
	    	assert( node->parent == NULL );
	    }

        for (int n = 0; n < 8; ++n) {
			node->child[n] = NULL;
			*node->vertex[n] = *(node->center) + node->direction[n] * node->scale;
	        if (node->parent) {
	            assert( node->parent->state == UNDECIDED );
	            assert( node->parent->prev_state != UNDECIDED );
	            //std::cout << parent->prev_state << "\n";
	            if (node->parent->prev_state == INSIDE) {
	            	node->f[n] = 1.0;
	                node->state = INSIDE;
				}
	            else if (node->parent->prev_state == OUTSIDE) {
	            	node->f[n] = -1.0;
	            	node->state = OUTSIDE;
				}
	            else
	                assert(0);
	        } else {
	        	node->f[n] = -1.0;
	        }
	    }
	 node->bb.clear();
	#ifdef MULTI_AXIS
	// Multi Axis
	  node->bb.addPoint(*center + GLVertex(-2.0,-2.0,-2.0) * scale); // calculate the minimum x,y,z coordinates
	  node->bb.addPoint(*center + GLVertex( 2.0, 2.0, 2.0) * scale); // calculate the maximum x,y,z coordinates
	#else
	  node->bb.addPoint( *vertex[2] ); // vertex[2] has the minimum x,y,z coordinates
	  node->bb.addPoint( *vertex[4] ); // vertex[4] has the max x,y,z
	#endif
	  node->isosurface_valid = false;

	  node->childcount = 0;
	  node->childStatus = 0;

	} else {
		node = new Octnode( nodeparent, index , nodescale , nodedepth , gl);
#ifdef MULTI_THREAD
		nodePoolMutex.unlock();
#endif
	}

	return node;
}

void Octnode::deleteOctnode(Octnode* node)
{
#ifdef MULTI_THREAD
	nodePoolMutex.lock();
#endif
	nodePool.push_back(node);
#ifdef MULTI_THREAD
	nodePoolMutex.unlock();
#endif
}
///added hust
void Octnode::DeformedVertices(const std::vector<double>& originalPoints, const std::vector<double>& deformedPoints)
{
    if (originalPoints.empty() || deformedPoints.empty()) {
            // 如果数据为空，使用原始顶点
            for (int i = 0; i < 8; i++) {
                deformedVertex[i] = *(vertex[i]);
            }
            return;
        }

        int numPoints = originalPoints.size() / 3;

        // 为每个顶点找到对应的形变后坐标
        for (int i = 0; i < 8; i++) {
            // 获取当前顶点的原始坐标
            double x = vertex[i]->x;
            double y = vertex[i]->y;
            double z = vertex[i]->z;

            // 在原始坐标列表中查找匹配的点
            int foundIndex = -1;

            for (int j = 0; j < numPoints; j++) {
                double ox = originalPoints[j];
                double oy = originalPoints[j + numPoints];
                double oz = originalPoints[j + 2 * numPoints];

                // 直接比较坐标是否相等，使用小的误差范围
                if (fabs(x - ox) < 1e-10 && fabs(y - oy) < 1e-10 && fabs(z - oz) < 1e-10) {
                    foundIndex = j;
                    break; // 找到完全匹配的点，立即退出循环
                }
            }

            // 如果找到了匹配的点，使用其形变后的坐标
            if (foundIndex != -1) {
                deformedVertex[i].x = deformedPoints[foundIndex];
                deformedVertex[i].y = deformedPoints[foundIndex + numPoints];
                deformedVertex[i].z = deformedPoints[foundIndex + 2 * numPoints];
                 std::cout << "找到与坐标 (" << x << ", " << y << ", " << z << ") 完全匹配的顶点！" << std::endl;
            } else {
                // 如果没有找到匹配的点，输出警告并使用原始坐标
                std::cout << "警告：没有找到与坐标 (" << x << ", " << y << ", " << z << ") 完全匹配的顶点！" << std::endl;
                deformedVertex[i] = *(vertex[i]);
            }
        }
    }
#endif //POOL_NODE

} // end namespace
// end of file octnode.cpp
