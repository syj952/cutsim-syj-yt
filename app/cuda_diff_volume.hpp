// cuda_diff_volume.hpp
#ifndef CUDA_DIFF_VOLUME_HPP
#define CUDA_DIFF_VOLUME_HPP

namespace cutsim {

// 节点数据结构
struct CudaNodeData {
    float x[8];            // 顶点的x坐标
    float y[8];            // 顶点的y坐标
    float z[8];            // 顶点的z坐标
    float f[8];            // 距离值
    int node_idx;          // 节点索引
};

// 球体体积参数
struct SphereParams {
    float x, y, z;         // 中心点
    float radius;          // 半径
    float r, g, b;         // 颜色
};

// 圆柱体体积参数
struct CylinderParams {
    float x, y, z;         // 中心点
    float radius;          // 半径
    float length;          // 长度
    float angle_x, angle_y, angle_z; // 旋转角度
    float rot_x, rot_y, rot_z;       // 旋转中心
    float r, g, b;         // 颜色
};

// 矩形体积参数
struct RectParams {
    float x, y, z;         // 角点
    float width, length, height; // 尺寸
    float angle_x, angle_y, angle_z; // 旋转角度
    float rot_x, rot_y, rot_z;       // 旋转中心
    float r, g, b;         // 颜色
};

// 切削工具参数
struct CutterParams {
    int type;               // 工具类型
    float x, y, z;          // 中心点
    float radius;           // 半径
    float length;           // 长度
    float r1, r2;           // 额外半径参数（用于Bull和Cone刀具）
    float angle_x, angle_z; // 旋转角度
    float r, g, b;          // 颜色
};

// 通用体积参数结构
struct VolumeParams {
    int type;               // 体积类型
    union {
        SphereParams sphere;
        CylinderParams cylinder;
        RectParams rect;
        CutterParams cutter;
    } params;
};

// CUDA差值计算函数接口
extern "C" void cuda_diff_volume(CudaNodeData* host_nodes, int numNodes, VolumeParams host_volume);

} // namespace cutsim

#endif // CUDA_DIFF_VOLUME_HPP
