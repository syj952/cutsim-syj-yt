
// 文件名: src/cutsim/cuda_functions.cpp
#include "cuda_functions.hpp"
#include "octree.hpp"
#include "octnode.hpp"
#include "volume.hpp"
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include <stack>
//#include "cuda_diff_volume.hpp"

// 确保结构体内存对齐一致
#pragma pack(push, 8)

// 修改VolumeParams结构体，使其与CUDA文件一致
struct CudaNodeData {
    float x[8];  // 节点顶点的 x 坐标
    float y[8];  // 节点顶点的 y 坐标
    float z[8];  // 节点顶点的 z 坐标
    float f[8];  // 每个顶点的距离值
    int node_idx;  // 节点索引
};

// 定义 VolumeParams 结构体
struct VolumeParams {
    // 修改枚举定义，使用int类型确保大小一致
    int type;  // 使用整数代替枚举，避免类型不匹配问题

    // 定义常量，与C++代码保持一致
    static const int SPHERE_VOLUME = 0;
    static const int CYLINDER_VOLUME = 1;
    static const int RECTANGLE_VOLUME = 2;

    // 为每个参数结构体添加类型声明
    struct SphereParams {
        float x, y, z;
        float radius;
        float r, g, b;
    };

    struct CylinderParams {
        float x, y, z;
        float radius;
        float length;
        float angle_x, angle_y, angle_z;
        float r, g, b;
        double holderradius;
    };
    struct RectangleParams {
            float center_x, center_y, center_z;
            float length_x, length_y, length_z;
            float r, g, b;
        };

    union {
        SphereParams sphere;
        CylinderParams cylinder;
        RectangleParams rectangle;
    } params;
};

struct GLVertex_xyz {
    float x, y, z;
    // 可以添加其他需要的成员
};

struct BladePoint {
    GLVertex_xyz v1, v2, v3, v4;
};

struct BladeParams {
    float center_x, center_y, center_z;
    float dx, dy, dz;
    float cube_resolution;
    double* point_r_blade;  // 设备指针
    int point_r_count;      // 点的数量
    GLVertex_xyz* blade_points;  // 改为GLVertex数组指针
    int blade_points_count;  // 点的数量
    int device_id;
    // 这里可以根据需要添加更多参数
};


extern "C" void cuda_diff_volume_blade(CudaNodeData* host_nodes, int numNodes, BladeParams host_blade,
                                       float** host_z_array, float** host_dmin_array, int* host_record_count);
extern "C" void cuda_diff_volume(CudaNodeData* host_nodes, int numNodes, VolumeParams host_volume);


#pragma pack(pop)


namespace cutsim {


    // 构造函数，初始化CUDA状态
    cuda_functions::cuda_functions() : deviceCount(0) {
        cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
            if (cudaStatus != cudaSuccess || deviceCount == 0) {
                qDebug() << "没有找到CUDA设备或CUDA初始化失败";
            } else {
                //qDebug() << "找到" << deviceCount << "个CUDA设备";
                cudaSetDevice(0);
                // 获取设备属性
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, 0);
                //qDebug() << "使用设备:" << deviceProp.name;
                //qDebug() << "计算能力:" << deviceProp.major << "." << deviceProp.minor;
            }
    }

    // 在类定义中添加这个函数声明
    void cuda_functions::clean_outside_nodes(Octnode* current) {
        if (!current) {
            return; // 如果节点为空，直接返回
        }

        // 如果当前节点是叶子节点，直接返回
        if (current->isLeaf()) {
            return;
        }

        // 检查是否有子节点
        if (current->childcount == 8) {
            // 先递归处理所有子节点
            bool all_outside = true;
            for (int n = 0; n < 8; ++n) {
                if (current->child[n]) {
                    clean_outside_nodes(current->child[n]);
                    // 检查子节点是否是OUTSIDE
                    if (!current->child[n]->is_outside()) {
                        all_outside = false;
                    }
                }
            }

            // 如果所有子节点都是OUTSIDE，则删除子节点并设置当前节点为OUTSIDE
            if (all_outside) {
                //qDebug() << "清理: 删除全部为OUTSIDE的子节点 (深度:" << current->depth << ")";
                current->state = Octnode::OUTSIDE;
                current->delete_children();
            }
        }
    }

    // 析构函数，清理CUDA资源
    cuda_functions::~cuda_functions() {}

    void cuda_functions::get_leaf_nodes_diff(Octnode* current, std::vector<Octnode*>& nodes_to_process,const Volume* vol, unsigned int max_depth) {
        // 如果节点已经在外部或没有与体积重叠，则直接返回
        if (current->is_outside() || !vol->bb.overlaps(current->bb)) {
            return;
        }


        // 首先判断是否有子节点
        if (current->childcount ==8) {
            // 检查子节点是否是outside
            for (int n = 0; n < 8; ++n) {
                if (!current->child[n]->is_outside()) {
                    get_leaf_nodes_diff(current->child[n], nodes_to_process,vol, max_depth);
                }
            }
            // 如果所有子节点都是outside，则不添加到nodes_to_process
        } else {
            // 没有子节点
                // 如果不是undecided状态且不是outside，判断是否达到最大深度
                if (current->depth < (max_depth - 1)) {
                    if (!current->is_undecided()) {current->force_setUndecided();}
                    // 未达到最大深度，设置为undecided并加入处理列表
                     // 如果是undecided状态，不加入到处理列表
                    current->subdivide();
                    for(int m = 0; m < 8; ++m) {
                        get_leaf_nodes_diff(current->child[m], nodes_to_process,vol, max_depth);
                    }
                }
                else
                // 如果已达到最大深度，则加入处理列表
                {nodes_to_process.push_back(current);}
        }
    }

    // 判断点是否在三角形内
    bool cuda_functions::pointInTriangle(float2 a, float2 b, float2 c, float2 p) {
        auto cross = [](float2 a, float2 b, float2 c) {
            return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
        };
        float c1 = cross(a, b, p);
        float c2 = cross(b, c, p);
        float c3 = cross(c, a, p);
        return (c1 >= 0 && c2 >= 0 && c3 >= 0) || (c1 <= 0 && c2 <= 0 && c3 <= 0);
    }

    // 计算点到线段距离
    float cuda_functions::pointToSegmentDist(float2 p, float2 a, float2 b) {
        float dx = b.x - a.x;
        float dy = b.y - a.y;
        if (dx == 0 && dy == 0) return hypotf(p.x - a.x, p.y - a.y);
        float t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx*dx + dy*dy);
        t = fmaxf(0.0f, fminf(1.0f, t));
        float proj_x = a.x + t * dx;
        float proj_y = a.y + t * dy;
        return hypotf(p.x - proj_x, p.y - proj_y);
    }

    void cuda_functions::get_leaf_nodes_diff_blade(Octnode* current, std::vector<Octnode*>& nodes_to_process,const AptCutterVolume* vol, unsigned int max_depth,std::chrono::duration<double>& elapsed) {
        // 如果节点已经在外部或没有与体积重叠，则直接返回
        if (current->depth > (vol->max_depth_2-1 )||current->is_outside()) {
            return;
        }

        GLVertex* v = current->center;
        float3 p{ v->x, v->y, v->z };

        // 获取包围盒的8个顶点
        const auto& bb = vol->bb_points;
        float3 min_coord = {
            std::min({std::get<0>(bb).x, std::get<1>(bb).x, std::get<2>(bb).x, std::get<3>(bb).x,
                        std::get<4>(bb).x, std::get<5>(bb).x, std::get<6>(bb).x, std::get<7>(bb).x}),
            std::min({std::get<0>(bb).y, std::get<1>(bb).y, std::get<2>(bb).y, std::get<3>(bb).y,
                        std::get<4>(bb).y, std::get<5>(bb).y, std::get<6>(bb).y, std::get<7>(bb).y}),
            std::min({std::get<0>(bb).z, std::get<1>(bb).z, std::get<2>(bb).z, std::get<3>(bb).z,
                        std::get<4>(bb).z, std::get<5>(bb).z, std::get<6>(bb).z, std::get<7>(bb).z})
        };
        float3 max_coord = {
            std::max({std::get<0>(bb).x, std::get<1>(bb).x, std::get<2>(bb).x, std::get<3>(bb).x,
                        std::get<4>(bb).x, std::get<5>(bb).x, std::get<6>(bb).x, std::get<7>(bb).x}),
            std::max({std::get<0>(bb).y, std::get<1>(bb).y, std::get<2>(bb).y, std::get<3>(bb).y,
                        std::get<4>(bb).y, std::get<5>(bb).y, std::get<6>(bb).y, std::get<7>(bb).y}),
            std::max({std::get<0>(bb).z, std::get<1>(bb).z, std::get<2>(bb).z, std::get<3>(bb).z,
                        std::get<4>(bb).z, std::get<5>(bb).z, std::get<6>(bb).z, std::get<7>(bb).z})
        };

        // 检查点是否在包围盒内
        bool inside = (p.x >= min_coord.x - current-> scale && p.x <= max_coord.x + current->scale &&
            p.y >= min_coord.y - current->scale  && p.y <= max_coord.y + current->scale &&
            p.z >= min_coord.z - current->scale  && p.z <= max_coord.z + current->scale );

        if (!inside) {
            return;
        }

        // 首先判断是否有子节点
        if (current->childcount ==8 && current->depth < (vol->max_depth_2-1)) {
            // 检查子节点是否是outside
            for (int n = 0; n < 8; ++n) {
                if (!current->child[n]->is_outside()) {
                    get_leaf_nodes_diff_blade(current->child[n], nodes_to_process,vol, max_depth,elapsed);
                }
            }
            // 如果所有子节点都是outside，则不添加到nodes_to_process
        } else {
            // 没有子节点
                // 如果不是undecided状态且不是outside，判断是否达到最大深度
                if (current->depth < (vol->max_depth_2-1)) {
                    if (!current->is_undecided()) {current->force_setUndecided();}
                    // 未达到最大深度，设置为undecided并加入处理列表
                     // 如果是undecided状态，不加入到处理列表
                    current->subdivide();
                    for(int m = 0; m < 8; ++m) {
                        get_leaf_nodes_diff_blade(current->child[m], nodes_to_process,vol, max_depth,elapsed);
                    }
                }
                else
                // 如果已达到最大深度，则加入处理列表
                {
                    nodes_to_process.push_back(current);
                }

        }
        if ( (current->childcount == 8) && ( /*current->all_child_state(Octnode::INSIDE) ||*/ current->all_child_state(Octnode::OUTSIDE) ) ) {
            current->state = Octnode::OUTSIDE;
            current->delete_children();
        }
    }

    void cuda_functions::get_leaf_nodes_diff_blade_second(Octnode* current, std::vector<Octnode*>& nodes_to_process,const AptCutterVolume* vol, unsigned int max_depth,std::chrono::duration<double>& elapsed){

        // 首先判断是否有子节点
        if (current->childcount ==8) {
            #pragma omp parallel for schedule(dynamic)  // 动态调度优化负载均衡
            // 检查子节点是否是outside
            for (int n = 0; n < 8; ++n) {
                if (!current->child[n]->is_outside()) {
                    #pragma omp task untied  // 允许线程窃取任务
                    get_leaf_nodes_diff_blade_second(current->child[n], nodes_to_process,vol, max_depth,elapsed);
                }
            }
            #pragma omp taskwait  // 等待所有子任务完成
            // 如果所有子节点都是outside，则不添加到nodes_to_process
        }
        else {
            if (current->depth < (vol->max_depth_2-1)) {
                if (!current->is_undecided()) {current->force_setUndecided();}
                // 未达到最大深度，设置为undecided并加入处理列表
                 // 如果是undecided状态，不加入到处理列表
                auto start1 = std::chrono::system_clock::now();
                current->subdivide();
                auto end1 = std::chrono::system_clock::now();
                elapsed += (end1 - start1);
                for(int m = 0; m < 8; ++m) {
                    get_leaf_nodes_diff_blade_second(current->child[m], nodes_to_process,vol, max_depth,elapsed);
                }
            }
                else
                // 如果已达到最大深度，则加入处理列表
                {
                    if(!current->is_outside())
                    {
                    nodes_to_process.push_back(current);
                    }
                }
        }
        if ( (current->childcount == 8) && ( /*current->all_child_state(Octnode::INSIDE) ||*/ current->all_child_state(Octnode::OUTSIDE) ) ) {
            current->state = Octnode::OUTSIDE;
            current->delete_children();
        }

    }

    void cuda_functions::get_leaf_nodes_sum(Octnode* current, std::vector<Octnode*>& nodes_to_process,const Volume* vol, unsigned int max_depth) {
        //qDebug() << "vol->type():"<<vol->type;
                // 如果节点已经在外部或没有与体积重叠，则直接返回
                if (current->is_inside() || !vol->bb.overlaps(current->bb)) {
                    return;
                }

                if ((current->depth == (max_depth-1)) && current->is_undecided() && !current->color.compareColor(vol->color)) { // return;
                        return;
                    }
                //qDebug() << "get_leaf_nodes_sum():";
                // 首先判断是否有子节点
                if (current->childcount ==8) {
                    // 检查子节点是否是inside
                    for (int n = 0; n < 8; ++n) {
                        if (!current->child[n]->is_inside()) {
                            get_leaf_nodes_sum(current->child[n], nodes_to_process,vol, max_depth);
                        }
                    }
                    // 如果所有子节点都是inside，则不添加到nodes_to_process
                } else {
                    // 没有子节点
                        // 如果不是undecided状态且不是inside，判断是否达到最大深度
                        if (current->depth < (max_depth - 1)) {
                            if (!current->is_undecided()) {current->force_setUndecided();}
                            // 未达到最大深度，设置为undecided并加入处理列表
                             // 如果是undecided状态，不加入到处理列表
                            current->subdivide();
                            for(int m = 0; m < 8; ++m) {
                                get_leaf_nodes_sum(current->child[m], nodes_to_process,vol, max_depth);
                            }
                        }
                        else
                        //qDebug() << "current->scale:" <<current->scale;
                        // 如果已达到最大深度，则加入处理列表
                        {nodes_to_process.push_back(current);}
                }


    }


    void cuda_functions::diff_volume(Octnode* current, const CylCutterVolume* vol, unsigned int max_depth) { // 添加max_depth参数


        // 准备要处理的节点列表
        std::vector<Octnode*> nodes_to_process;

        std::chrono::system_clock::time_point start, stop;
        start = std::chrono::system_clock::now();

        // 收集所有需要处理的叶节点
        get_leaf_nodes_diff(current, nodes_to_process,vol, max_depth);

        stop = std::chrono::system_clock::now();
        //qDebug() << std::chrono::duration<double>(stop - start).count();


        // 如果没有需要处理的节点，直接返回
        if (nodes_to_process.empty()) {
            qDebug() << "列表为空,计算结束";
            return;
        }

        // 分配节点数据数组，用于传输到GPU
        size_t node_count = nodes_to_process.size();
        // 在文件顶部添加类型别名
        // 修改类型别名为直接使用命名空间中的VolumeType
        typedef cutsim::VolumeType VolumeType;

        // 修改数组声明部分（约92行）
        CudaNodeData* host_nodes = new CudaNodeData[node_count];  // 改为动态数组

        // 修改参数填充循环（约94-103行）
        for (size_t i = 0; i < node_count; i++) {
            Octnode* node = nodes_to_process[i];
            for (int j = 0; j < 8; j++) {
                host_nodes[i].x[j] = static_cast<float>(node->vertex[j]->x);
                host_nodes[i].y[j] = static_cast<float>(node->vertex[j]->y);
                host_nodes[i].z[j] = static_cast<float>(node->vertex[j]->z);
                host_nodes[i].f[j] = static_cast<float>(node->f[j]);
            }
            host_nodes[i].node_idx = static_cast<int>(i);
        }

        // 修改类型转换部分（约113行和126行）
        // 在参数填充循环之后添加变量声明（约109行）
        VolumeParams volume_params;  // 声明结构体实例

        // 修改类型转换部分（原110行）
        volume_params.type = vol->type;

        bool supported_volume = true;

        // 根据体积类型填充参数
        switch (vol->type) {

            case CYLINDER_VOLUME: {
                const CylCutterVolume* cv = vol;
                volume_params.type = 1;

                // 使用公有访问方法获取私有成员
                GLVertex center = cv->center;
                GLVertex angle = cv->angle;          // 需要在volume.hpp中添加对应方法


                volume_params.params.cylinder.x = center.x;
                volume_params.params.cylinder.y = center.y;
                volume_params.params.cylinder.z = center.z;
                volume_params.params.cylinder.radius =cv->radius;
                volume_params.params.cylinder.length =cv->length;

                // 修改角度参数赋值
                volume_params.params.cylinder.angle_x = angle.x;
                volume_params.params.cylinder.angle_y = angle.y;
                volume_params.params.cylinder.angle_z = angle.z;

                volume_params.params.cylinder.r = cv->color.r;
                volume_params.params.cylinder.g = cv->color.g;
                volume_params.params.cylinder.b = cv->color.b;

                volume_params.params.cylinder.holderradius= cv->holderradius;
                break;
            }
             }


        //qDebug() << "调用cuda_diff_volume";

        // 添加CUDA错误检查
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            qDebug() << "CUDA初始化错误:" << cudaGetErrorString(cudaStatus);
            delete[] host_nodes;
            return;
        }

        try {
                // 调用CUDA实现
                //qDebug() << "开始调用CUDA函数...";

                start = std::chrono::system_clock::now();

                cuda_diff_volume(host_nodes, node_count, volume_params);

                stop = std::chrono::system_clock::now();
                //qDebug() << "cuda_diff_volume() :" << std::chrono::duration<double>(stop - start).count() << "sec.";
                //qDebug() << std::chrono::duration<double>(stop - start).count();
                //qDebug() << "CUDA函数调用成功";

                // 检查CUDA执行错误
                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    qDebug() << "CUDA执行错误:" << cudaGetErrorString(cudaStatus);
                    throw std::runtime_error("CUDA执行错误");
                }

                // 同步设备
                cudaStatus = cudaDeviceSynchronize();
                if (cudaStatus != cudaSuccess) {
                    qDebug() << "CUDA同步错误:" << cudaGetErrorString(cudaStatus);
                    throw std::runtime_error("CUDA同步错误");
                }
            } catch (const std::exception& e) {
                qDebug() << "CUDA函数调用异常:" << e.what();
                // 释放内存并回退到CPU实现
                delete[] host_nodes;
                for (size_t i = 0; i < node_count; i++) {
                    nodes_to_process[i]->diff(vol);
                    nodes_to_process[i]->set_state();
                }
                return;
            }

        // 更新节点数据
        for (size_t i = 0; i < node_count; i++) {
            Octnode* node = nodes_to_process[i];
    //        qDebug() << "node_count:" <<i;
//            if(i==0){
//                qDebug() << "node更新:" <<node->f[0];
//            }
            if (node->f[0]==host_nodes[i].f[0]){
                qDebug() << "error：dist未更新:" << (node->f[0]==host_nodes[i].f[0]);

}

            bool updated = false; // 标记是否有更新
            for (int j = 0; j < 8; j++) {
                // 只有当新的距离值小于原来的距离值时才更新
                //qDebug() << "dist:" << host_nodes[i].f[j];
                //qDebug() << "f[n]:" << node->f[j];
                if (static_cast<double>(-host_nodes[i].f[j]) < node->f[j]) {
                    node->f[j] = static_cast<double>(-host_nodes[i].f[j]);
                    updated = true; // 标记有更新
                   // qDebug() << "更新:" << node->f[j];
                }
                //node->f[j]=-1.0;
            }

            // 如果有更新，则更新节点颜色
            if (updated) {
                //qDebug() << "颜色更新";
                node->color = vol->color; // 更新颜色为体积的颜色
            }

            node->set_state(); // 更新节点状态


        }

        // 释放内存A
        delete[] host_nodes;

        //current->set_state();

        start = std::chrono::system_clock::now();

        //clean_outside_nodes(current);

        stop = std::chrono::system_clock::now();
        //qDebug() << "clean_outside_nodes():" << std::chrono::duration<double>(stop - start).count() << "sec.";

        //diff_volume(current, vol, max_depth);

     }

    void cuda_functions::sum_volume(Octnode* current, const Volume* vol, unsigned int max_depth) { // 添加max_depth参数


        // 准备要处理的节点列表
        std::vector<Octnode*> nodes_to_process;

        //std::chrono::system_clock::time_point start, stop;
        //start = std::chrono::system_clock::now();
        qDebug() << "sum_volume():" ;
        // 收集所有需要处理的叶节点
        get_leaf_nodes_sum(current, nodes_to_process,vol, max_depth);

        //stop = std::chrono::system_clock::now();
        //qDebug() << "get_leaf_nodes():" << std::chrono::duration<double>(stop - start).count() << "sec.";
        //qDebug() << std::chrono::duration<double>(stop - start).count();


        // 如果没有需要处理的节点，直接返回
        if (nodes_to_process.empty()) {
            qDebug() << "列表为空,计算结束";
            return;
        }

        // 分配节点数据数组，用于传输到GPU
        size_t node_count = nodes_to_process.size();
        // 在文件顶部添加类型别名
        // 修改类型别名为直接使用命名空间中的VolumeType
        typedef cutsim::VolumeType VolumeType;

        // 修改数组声明部分（约92行）
        CudaNodeData* host_nodes = new CudaNodeData[node_count];  // 改为动态数组

        // 修改参数填充循环（约94-103行）
        for (size_t i = 0; i < node_count; i++) {
            Octnode* node = nodes_to_process[i];
            for (int j = 0; j < 8; j++) {
                host_nodes[i].x[j] = static_cast<float>(node->vertex[j]->x);
                host_nodes[i].y[j] = static_cast<float>(node->vertex[j]->y);
                host_nodes[i].z[j] = static_cast<float>(node->vertex[j]->z);
                host_nodes[i].f[j] = static_cast<float>(node->f[j]);
            }
            host_nodes[i].node_idx = static_cast<int>(i);
        }

        // 修改类型转换部分（约113行和126行）
        // 在参数填充循环之后添加变量声明（约109行）
        VolumeParams volume_params;  // 声明结构体实例

        // 修改类型转换部分（原110行）
        volume_params.type = vol->type;

        bool supported_volume = true;

        // 根据体积类型填充参数
        switch (vol->type) {

            case CYLINDER_VOLUME: {
                const CylCutterVolume* cv = static_cast<const CylCutterVolume*>(vol);  // 安全转换
                volume_params.type = VolumeParams::CYLINDER_VOLUME;

                // 使用公有访问方法获取私有成员
                GLVertex center = cv->center;
                GLVertex angle = cv->angle;          // 需要在volume.hpp中添加对应方法


                volume_params.params.cylinder.x = center.x;
                volume_params.params.cylinder.y = center.y;
                volume_params.params.cylinder.z = center.z;
                volume_params.params.cylinder.radius =cv->radius;
                volume_params.params.cylinder.length =cv->length;

                // 修改角度参数赋值
                volume_params.params.cylinder.angle_x = angle.x;
                volume_params.params.cylinder.angle_y = angle.y;
                volume_params.params.cylinder.angle_z = angle.z;

                volume_params.params.cylinder.r = cv->color.r;
                volume_params.params.cylinder.g = cv->color.g;
                volume_params.params.cylinder.b = cv->color.b;

                volume_params.params.cylinder.holderradius= cv->holderradius;
                break;
            }

            case RECTANGLE_VOLUME: {  // 新增矩形体积分支
                        const RectVolume* rv = static_cast<const RectVolume*>(vol);
                        volume_params.type = VolumeParams::RECTANGLE_VOLUME;

                        // 获取矩形中心坐标
                        GLVertex center = rv->getCenter();
                        volume_params.params.rectangle.center_x = center.x;
                        volume_params.params.rectangle.center_y = center.y;
                        volume_params.params.rectangle.center_z = center.z;

                        // 设置矩形尺寸
                        volume_params.params.rectangle.length_x = rv->getLengthX();
                        volume_params.params.rectangle.length_y = rv->getLengthY();
                        volume_params.params.rectangle.length_z = rv->getLengthZ();

                        // 设置颜色参数
                        volume_params.params.rectangle.r = rv->color.r;
                        volume_params.params.rectangle.g = rv->color.g;
                        volume_params.params.rectangle.b = rv->color.b;
                        break;
                    }
            }


        //qDebug() << "调用cuda_diff_volume";

        // 添加CUDA错误检查
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            qDebug() << "CUDA初始化错误:" << cudaGetErrorString(cudaStatus);
            delete[] host_nodes;
            return;
        }

        try {
                // 调用CUDA实现
                //qDebug() << "开始调用CUDA函数...";

                //start = std::chrono::system_clock::now();

                cuda_diff_volume(host_nodes, node_count, volume_params);

                //stop = std::chrono::system_clock::now();
                //qDebug() << "cuda_diff_volume() :" << std::chrono::duration<double>(stop - start).count() << "sec.";
                //qDebug() << std::chrono::duration<double>(stop - start).count();
                //qDebug() << "CUDA函数调用成功";

                // 检查CUDA执行错误
                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    qDebug() << "CUDA执行错误:" << cudaGetErrorString(cudaStatus);
                    throw std::runtime_error("CUDA执行错误");
                }

                // 同步设备
                cudaStatus = cudaDeviceSynchronize();
                if (cudaStatus != cudaSuccess) {
                    qDebug() << "CUDA同步错误:" << cudaGetErrorString(cudaStatus);
                    throw std::runtime_error("CUDA同步错误");
                }
            } catch (const std::exception& e) {
                qDebug() << "CUDA函数调用异常:" << e.what();
                // 释放内存并回退到CPU实现
                delete[] host_nodes;
                for (size_t i = 0; i < node_count; i++) {
                    nodes_to_process[i]->diff(vol);
                    nodes_to_process[i]->set_state();
                }
                return;
            }

        // 更新节点数据
        for (size_t i = 0; i < node_count; i++) {
            Octnode* node = nodes_to_process[i];
    //        qDebug() << "node_count:" <<i;
//            if(i==0){
//                qDebug() << "node更新:" <<node->f[0];
//            }
            if (node->f[0]==host_nodes[i].f[0]){
                qDebug() << "error：dist未更新:" << (node->f[0]==host_nodes[i].f[0]);

}

            bool updated = false; // 标记是否有更新
            for (int j = 0; j < 8; j++) {
                if (static_cast<double>(host_nodes[i].f[j]) > node->f[j]) {
                    node->f[j] = static_cast<double>(host_nodes[i].f[j]);
                    updated = true; // 标记有更新
                }
            }

            // 如果有更新，则更新节点颜色
            if (updated) {
                //qDebug() << "颜色更新";
                node->color = vol->color; // 更新颜色为体积的颜色
            }

            node->set_state(); // 更新节点状态


        }

        // 释放内存A
        delete[] host_nodes;

     }


    // ... 其他代码 ...
    void cuda_functions::diff_volume_blade(Octnode* current, const AptCutterVolume* vol, unsigned int max_depth, Octree* octree) {
        // 收集叶节点
        std::vector<Octnode*> nodes_to_process;

        std::chrono::system_clock::time_point start, stop;
        start = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed(0);

        get_leaf_nodes_diff_blade(current, nodes_to_process, vol, max_depth,elapsed);

        stop = std::chrono::system_clock::now();
        qDebug() << "get_leaf_nodes_diff_blade():" << std::chrono::duration<double>(stop - start).count() << "sec.";
 //       qDebug() << "get_leaf_nodes_diff_blade_second():" << elapsed.count() << "sec.";

       start = std::chrono::system_clock::now();

        size_t node_count = nodes_to_process.size();
        if (node_count == 0) return;

        CudaNodeData* host_nodes = new CudaNodeData[node_count];
        for (size_t i = 0; i < node_count; i++) {
            Octnode* node = nodes_to_process[i];
            for (int j = 0; j < 8; j++) {
                host_nodes[i].x[j] = static_cast<float>(node->vertex[j]->x);
                host_nodes[i].y[j] = static_cast<float>(node->vertex[j]->y);
                host_nodes[i].z[j] = static_cast<float>(node->vertex[j]->z);
                host_nodes[i].f[j] = static_cast<float>(node->f[j]);
            }
            host_nodes[i].node_idx = static_cast<int>(i);
        }

        BladeParams blade;
        blade.center_x = static_cast<float>(vol->center.x);
        blade.center_y = static_cast<float>(vol->center.y);
        blade.center_z = static_cast<float>(vol->center.z);
        blade.dx = static_cast<float>(vol->dx);  // 新增dx赋值
        blade.dy = static_cast<float>(vol->dy);  // 新增dy赋值
        blade.dz = static_cast<float>(vol->dz);  // 新增dz赋值
        blade.cube_resolution = static_cast<float>(vol->cube_resolution_2);
        blade.device_id = 0;  // 设置设备ID

        // 处理blade_points数据
        if (!vol->blade_points.empty()) {
            // 分配主机内存
            blade.blade_points = new GLVertex_xyz[vol->blade_points.size()];
            // 拷贝数据
            for (size_t i = 0; i < vol->blade_points.size(); ++i) {
                blade.blade_points[i].x = vol->blade_points[i].x;
                blade.blade_points[i].y = vol->blade_points[i].y;
                blade.blade_points[i].z = vol->blade_points[i].z;
            }

            blade.blade_points_count = static_cast<int>(vol->blade_points.size());
        }
        else {
            blade.blade_points = nullptr;
            blade.blade_points_count = 0;
        }

        float* host_z_array = nullptr;
        float* host_dmin_array = nullptr;
        int host_record_count = 0;

        // 添加CUDA错误检查
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            qDebug() << "CUDA初始化错误:" << cudaGetErrorString(cudaStatus);
            delete[] host_nodes;
            return;
        }

        // 配置BladeParams
        blade.device_id = 0;  // 设置设备ID

        // 处理设备内存指针（关键修改）
        if (!vol->blade_points.empty()) {
            GLVertex_xyz* dev_points = nullptr;
            cudaMalloc(&dev_points, blade.blade_points_count * sizeof(GLVertex));
            cudaMemcpy(dev_points, blade.blade_points,
                blade.blade_points_count * sizeof(GLVertex),
                cudaMemcpyHostToDevice);
            blade.blade_points = dev_points;
        }


        stop = std::chrono::system_clock::now();
        qDebug() << "设备内存操作():" << std::chrono::duration<double>(stop - start).count() << "sec.";


        try {
                cudaSetDevice(0);
                cudaDeviceSynchronize();
                cudaError_t preLaunchErr = cudaGetLastError();
                if (preLaunchErr != cudaSuccess) {
                    qDebug() << "Pre-launch error:" << cudaGetErrorString(preLaunchErr);
                    return;
                }

                start = std::chrono::system_clock::now();

                cuda_diff_volume_blade (host_nodes, node_count, blade,
                                        &host_z_array, &host_dmin_array, &host_record_count);

                stop = std::chrono::system_clock::now();
                qDebug() << "cuda计算时间():" << std::chrono::duration<double>(stop - start).count() << "sec.";

                // 检查CUDA执行错误
                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    qDebug() << "CUDA执行错误:" << cudaGetErrorString(cudaStatus);
                    throw std::runtime_error("CUDA执行错误");
                }

                // 同步设备
                cudaStatus = cudaDeviceSynchronize();
                if (cudaStatus != cudaSuccess) {
                    qDebug() << "CUDA同步错误:" << cudaGetErrorString(cudaStatus);
                    throw std::runtime_error("CUDA同步错误");
                }
            } catch (const std::exception& e) {
                qDebug() << "CUDA函数调用异常:" << e.what();
                // 释放内存并回退到CPU实现
                delete[] host_nodes;
                for (size_t i = 0; i < node_count; i++) {
                    nodes_to_process[i]->diff(vol);
                    nodes_to_process[i]->set_state();
                }
                return;
            }

        start = std::chrono::system_clock::now();
        octree->cwenodelist.clear();
        // 更新Octnode
        for (size_t i = 0; i < node_count; i++) {
            Octnode* node = nodes_to_process[i];
            bool updated = false;
            for (int j = 0; j < 8; j++) {
                if (static_cast<double>(-host_nodes[i].f[j]) < node->f[j]) {
                    node->f[j] = static_cast<double>(-host_nodes[i].f[j]);
                    octree->cwenodelist.push_back(node);//与刀具接触的node
                    updated = true;
                }
            }
            if (updated) node->color = vol->color;
            node->set_state();
        }

        delete[] host_nodes;

        // 主机端处理：map记录每个z的最大d_min
        std::unordered_map<float, float> z2dmin;
        //qDebug() << "host_record_count:"<<host_record_count;
        for (int i = 0; i < host_record_count; ++i) {
            int inside_index = host_z_array[i];
            float dmin = host_dmin_array[i];
            if (z2dmin.find(inside_index) == z2dmin.end() || dmin > z2dmin[inside_index]) {
                z2dmin[inside_index] = dmin;
            }
        }
        // 输出或保存z2dmin
        for (const auto& kv : z2dmin) {
            const_cast<AptCutterVolume*>(vol)->addcut_h(kv.first, kv.second);
        }

        free(host_z_array);
        free(host_dmin_array);

        stop = std::chrono::system_clock::now();
        qDebug() << "结果处理：():" << std::chrono::duration<double>(stop - start).count() << "sec.";


    }



} // end namespace cutsim

