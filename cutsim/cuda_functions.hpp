// 文件名: src/cutsim/cuda_functions.hpp
#ifndef CUDA_DIFF_VOLUME_HPP
#define CUDA_DIFF_VOLUME_HPP

#include "octree.hpp"
#include "volume.hpp"
#include <vector>


namespace cutsim {

    // 前向声明
    class Octnode;
    class Volume;

    // 定义cuda_functions类
    class cuda_functions {
    public:
        // 构造函数，初始化CUDA状态
        cuda_functions();

        // 析构函数，清理CUDA资源
        ~cuda_functions();

        // 计算体积差异并更新节点状态
       void diff_volume(Octnode* current, const CylCutterVolume* vol, unsigned int max_depth);

       void sum_volume(Octnode* current, const Volume* vol, unsigned int max_depth);

       void diff_volume_blade(Octnode* current, const AptCutterVolume* vol, unsigned int max_depth, Octree* octree);
       void diff_volume_blade_multi_gpu(Octnode* current, const AptCutterVolume* vol, unsigned int max_depth);
    private:
        void clean_outside_nodes(Octnode* current);
        // 获取叶节点
        void get_leaf_nodes_diff(Octnode* current, std::vector<Octnode*>& nodes_to_process,const Volume* vol, unsigned int max_depth);
        struct float2 {
            float x;
            float y;
        };
        float pointToSegmentDist(float2 p, float2 a, float2 b);
        bool pointInTriangle(float2 a, float2 b, float2 c, float2 p);
        void get_leaf_nodes_diff_blade(Octnode* current, std::vector<Octnode*>& nodes_to_process,const AptCutterVolume* vol, unsigned int max_depth,std::chrono::duration<double>& elapsed);
        void get_leaf_nodes_diff_blade_second(Octnode* current, std::vector<Octnode*>& nodes_to_process,const AptCutterVolume* vol, unsigned int max_depth,std::chrono::duration<double>& elapsed);
        void get_leaf_nodes_sum(Octnode* current, std::vector<Octnode*>& nodes_to_process,const Volume* vol, unsigned int max_depth);

        // 设备数量
        int deviceCount;
        std::chrono::duration<double> total_elapsed; // 新增总耗时记录

        // 其他私有成员变量和方法（如果有的话）
    };

} // end namespace cutsim
#endif // CUDA_DIFF_VOLUME_HPP
