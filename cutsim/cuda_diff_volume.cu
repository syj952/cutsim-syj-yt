// cuda_diff_volume.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

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

struct GLVertex {
    float x, y, z;
    // 可以添加其他需要的成员
};


struct BladeParams {
    float center_x, center_y, center_z;
    float dx, dy, dz;
    float cube_resolution;
    double* point_r_blade;  // 设备指针
    int point_r_count;      // 点的数量
    GLVertex* blade_points;  // 改为GLVertex数组指针
    int blade_points_count;  // 点的数量
    int device_id;
    // 这里可以根据需要添加更多参数
};


namespace cutsim {

    // 添加坐标旋转辅助函数
    __device__ float3 rotate_point(float3 p, float ax, float ay, float az) {
        // 绕X轴旋转
        float y1 = p.y * cosf(ax) - p.z * sinf(ax);
        float z1 = p.y * sinf(ax) + p.z * cosf(ax);

        // 绕Y轴旋转
        float x2 = p.x * cosf(ay) + z1 * sinf(ay);
        float z2 = -p.x * sinf(ay) + z1 * cosf(ay);

        // 绕Z轴旋转
        float x3 = x2 * cosf(az) - y1 * sinf(az);
        float y3 = x2 * sinf(az) + y1 * cosf(az);

        return make_float3(x3, y3, z2);
    }

    // 计算点到圆柱体的距离（完整实现）
    __device__ float cylinder_dist(float3 p, VolumeParams::CylinderParams* cyl) {
        // 检查圆柱体参数是否有效
        if (cyl->radius <= 0.0f || cyl->length <= 0.0f) {
            return -1000.0f; // 返回一个较大的负值，表示无效
        }

        float3 t;
        float d;

#ifdef MULTI_AXIS
        // 多轴模式：先将点p按刀具角度旋转
        // 注意：这里假设cyl.center是圆柱体中心点，cyl.angle_x和cyl.angle_z是旋转角度
        float3 rotated_p = rotate_point(make_float3(p.x - cyl->x, p.y - cyl->y, p.z - cyl->z),
            -cyl->angle_x, 0.0f, -cyl->angle_z);
        t = rotated_p;

        // 计算到圆柱轴线的距离（XY平面上的距离）
        d = sqrtf(rotated_p.x * rotated_p.x + rotated_p.y * rotated_p.y);
#else
        // 单轴模式：直接计算相对位置
        t = make_float3(p.x - cyl->x, p.y - cyl->y, p.z - cyl->z);

        // 计算到圆柱轴线的距离（XY平面上的距离）
        d = sqrtf(t.x * t.x + t.y * t.y);
#endif

        // 防止数值溢出
        if (isnan(d) || isinf(d) || d > 1e6f) {
            return -1000.0f;
        }

        // 根据CPU版本逻辑判断点的位置
        if (t.z >= 0.0f) {
            // 点在圆柱体上方或同高度
            return t.z > cyl->length ? cyl->holderradius - d : cyl->radius - d;  // 正值表示内部，负值表示外部
        }
        else {
            // 点在圆柱体下方
            if (d < cyl->radius) {
                // 在圆柱体内部
                return t.z;
            }
            else {
                // 在圆柱体底面外侧
                // 计算到底部边缘的距离
                float3 n = make_float3(t.x, t.y, 0.0f);

                // 防止除以零
                if (d < 1e-6f) {
                    return -1000.0f;
                }

                // 归一化并缩放到半径长度
                n.x = n.x * (cyl->radius / d);
                n.y = n.y * (cyl->radius / d);

                // 计算点到边缘的距离并取负值
                float3 diff = make_float3(t.x - n.x, t.y - n.y, t.z);
                float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

                // 检查计算结果是否有效
                if (isnan(dist) || isinf(dist)) {
                    return -1000.0f;
                }

                return -dist;
            }
        }
    }
    // 矩形体积距离计算函数
    __device__ float rectangle_dist(float3 p, VolumeParams::RectangleParams* rect) {
        // 计算到各面的距离
        float dx = fabsf(p.x - rect->center_x) - rect->length_x / 2;
        float dy = fabsf(p.y - rect->center_y) - rect->length_y / 2;
        float dz = fabsf(p.z - rect->center_z) - rect->length_z / 2;

        // 计算内部/外部符号距离
        float inside_dx = fmaxf(dx, 0.0f);
        float inside_dy = fmaxf(dy, 0.0f);
        float inside_dz = fmaxf(dz, 0.0f);
        float dist = -sqrtf(inside_dx * inside_dx + inside_dy * inside_dy + inside_dz * inside_dz);

        // 外部距离计算
        if (dx < 0 && dy < 0 && dz < 0) {
            return fminf(fminf(fabsf(dx), fabsf(dy)), fabsf(dz));
        }
        return dist;
    }
    // 根据体积类型计算距离
    __device__ float calculate_distance(float3 point, VolumeParams* volume) {
        if (volume->type == VolumeParams::CYLINDER_VOLUME) {
            return cylinder_dist(point, &(volume->params.cylinder));
        }
        else if (volume->type == VolumeParams::RECTANGLE_VOLUME) {
            return rectangle_dist(point, &(volume->params.rectangle));
        }
        // 可继续添加其他体积类型
        return 10000.0f; // 默认返回较大正值表示外部
    }

    // CUDA核函数，计算差值操作
    // 修改CUDA核函数，增加更多的边界检查和错误防护
    __global__ void diff_kernel(CudaNodeData* nodes, int numNodes, VolumeParams* volume) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int nodeIdx = idx / 8;
        int vertexIdx = idx % 8;

        if (nodeIdx >= numNodes) {
            return;
        }

        // 获取节点的顶点坐标
        float3 point;
        point.x = nodes[nodeIdx].x[vertexIdx];
        point.y = nodes[nodeIdx].y[vertexIdx];
        point.z = nodes[nodeIdx].z[vertexIdx];
        // 检查坐标是否有效
        if (isnan(point.x) || isnan(point.y) || isnan(point.z) ||
            isinf(point.x) || isinf(point.y) || isinf(point.z)) {
            printf("坐标检查false");
            return;
        }

        // 计算点到体积的距离
        float dist = calculate_distance(point, volume);
        // 检查计算结果是否有效
        if (isnan(dist) || isinf(dist)) {
            printf("结果检查false");
            return;
        }
        // 更新节点的距离值
        nodes[nodeIdx].f[vertexIdx] = dist;
    }

    __device__ float3 calculatePlaneNormal(float3 a, float3 b, float3 c) {
        float3 ab = { b.x - a.x, b.y - a.y, b.z - a.z };
        float3 ac = { c.x - a.x, c.y - a.y, c.z - a.z };

        // 计算叉乘得到法向量
        float3 normal = {
            ab.y * ac.z - ab.z * ac.y,
            ab.z * ac.x - ab.x * ac.z,
            ab.x * ac.y - ab.y * ac.x
        };

        // 归一化法向量
        float length = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (length < 1e-6f) return make_float3(0, 0, 0);
        return { normal.x / length, normal.y / length, normal.z / length };
    }
    // 计算向量长度
    __device__ float length(float3 v) {
        return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    // 向量归一化
    __device__ float3 normalize(float3 v) {
        float len = length(v);
        if (len < 1e-6f) return v;
        return make_float3(v.x / len, v.y / len, v.z / len);
    }

    __device__ float dot3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

    __device__ float3 cross3(float3 a, float3 b) {
        return make_float3(
            a.y*b.z - a.z*b.y,
            a.z*b.x - a.x*b.z,
            a.x*b.y - a.y*b.x
        );
    }
    // 添加辅助函数：点是否在三角面内
    __device__ bool pointInTriangle3D(float3 p, float3 a, float3 b, float3 c) {
        float3 v0 = { b.x - a.x, b.y - a.y, b.z - a.z };
        float3 v1 = { c.x - a.x, c.y - a.y, c.z - a.z };
        float3 v2 = { p.x - a.x, p.y - a.y, p.z - a.z };

        float d00 = dot3(v0, v0);
        float d01 = dot3(v0, v1);
        float d11 = dot3(v1, v1);
        float d20 = dot3(v2, v0);
        float d21 = dot3(v2, v1);

        float denom = d00 * d11 - d01 * d01;
        if (fabsf(denom) < 1e-6f) return false;

        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0f - v - w;

        return (u >= -1e-6f) && (v >= -1e-6f) && (w >= -1e-6f);
    }

    // 新增加函数：判断点是否在六面体（由两个相邻四边形构成）
    __device__ bool pointInHexahedron(float3 p, GLVertex* quad1, GLVertex* quad2) {
        // 将四边形分解为6个三角面（每个四边形分解为2个三角形）
        float3 faces[12][3] = {
            // 底面三角形1
            {make_float3(quad1[0].x, quad1[0].y, quad1[0].z),
             make_float3(quad1[1].x, quad1[1].y, quad1[1].z),
             make_float3(quad1[2].x, quad1[2].y, quad1[2].z)},
            // 底面三角形2
            {make_float3(quad1[0].x, quad1[0].y, quad1[0].z),
             make_float3(quad1[2].x, quad1[2].y, quad1[2].z),
             make_float3(quad1[3].x, quad1[3].y, quad1[3].z)},
            // 顶面三角形1
            {make_float3(quad2[0].x, quad2[0].y, quad2[0].z),
             make_float3(quad2[1].x, quad2[1].y, quad2[1].z),
             make_float3(quad2[2].x, quad2[2].y, quad2[2].z)},
            // 顶面三角形2
            {make_float3(quad2[0].x, quad2[0].y, quad2[0].z),
             make_float3(quad2[2].x, quad2[2].y, quad2[2].z),
             make_float3(quad2[3].x, quad2[3].y, quad2[3].z)},
            // 侧面四边形分解的三角形
            {make_float3(quad1[0].x, quad1[0].y, quad1[0].z),
             make_float3(quad1[1].x, quad1[1].y, quad1[1].z),
             make_float3(quad2[1].x, quad2[1].y, quad2[1].z)},
            {make_float3(quad1[0].x, quad1[0].y, quad1[0].z),
             make_float3(quad2[1].x, quad2[1].y, quad2[1].z),
             make_float3(quad2[0].x, quad2[0].y, quad2[0].z)},
            {make_float3(quad1[1].x, quad1[1].y, quad1[1].z),
             make_float3(quad1[2].x, quad1[2].y, quad1[2].z),
             make_float3(quad2[2].x, quad2[2].y, quad2[2].z)},
            {make_float3(quad1[1].x, quad1[1].y, quad1[1].z),
             make_float3(quad2[2].x, quad2[2].y, quad2[2].z),
             make_float3(quad2[1].x, quad2[1].y, quad2[1].z)},
            {make_float3(quad1[2].x, quad1[2].y, quad1[2].z),
             make_float3(quad1[3].x, quad1[3].y, quad1[3].z),
             make_float3(quad2[3].x, quad2[3].y, quad2[3].z)},
            {make_float3(quad1[2].x, quad1[2].y, quad1[2].z),
             make_float3(quad2[3].x, quad2[3].y, quad2[3].z),
             make_float3(quad2[2].x, quad2[2].y, quad2[2].z)},
            {make_float3(quad1[3].x, quad1[3].y, quad1[3].z),
             make_float3(quad1[0].x, quad1[0].y, quad1[0].z),
             make_float3(quad2[0].x, quad2[0].y, quad2[0].z)},
            {make_float3(quad1[3].x, quad1[3].y, quad1[3].z),
             make_float3(quad2[0].x, quad2[0].y, quad2[0].z),
             make_float3(quad2[3].x, quad2[3].y, quad2[3].z)}
        };

        // 检查点是否在顶面底面之间
        float3 normal_0 = calculatePlaneNormal(faces[0][0], faces[0][1], faces[0][2]);
        float3 vec_to_p_0 = make_float3(p.x - faces[0][0].x, p.y - faces[0][0].y, p.z - faces[0][0].z);
        float3 vec_to_p_2 = make_float3(p.x - faces[2][0].x, p.y - faces[2][0].y, p.z - faces[2][0].z);
        if (dot3(normal_0, vec_to_p_0)* dot3(normal_0, vec_to_p_2) > 0) {
                        //printf("不在顶面底面之间");
                        return false;
                    }
        // 检查点是否在所有侧面的同一侧
        float first_sign = 0.0f;
        for(int i = 4; i < 12; ++i) {
            float3 normal = calculatePlaneNormal(faces[i][0], faces[i][1], faces[i][2]);
            float3 vec_to_p = make_float3(p.x - faces[i][0].x, p.y - faces[i][0].y, p.z - faces[i][0].z);
            float dot_result = dot3(normal, vec_to_p);

            // 记录第一个面的符号
            if (i == 4) {
                first_sign = dot_result;
                continue;
            }
            // 后续面必须与第一个面符号相同
            if (dot_result * first_sign < 0) {
                //printf("不在侧面之间\n");
                return false;
            }
        }
        return true;
    }

    // 点到线段的最小距离（3D优化版）
    __device__ float pointToSegmentDist3D(float3 p, float3 a, float3 b) {
        float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
        float3 ap = make_float3(p.x - a.x, p.y - a.y, p.z - a.z);

        float t = dot3(ap, ab) / dot3(ab, ab);
        t = fmaxf(0.0f, fminf(1.0f, t));

        float3 projection = make_float3(a.x + t * ab.x, a.y + t * ab.y, a.z + t * ab.z);
        float3 delta = make_float3(p.x - projection.x, p.y - projection.y, p.z - projection.z);

        return sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    }

    // 新增加函数：计算点到六面体的最小距离
    __device__ float distanceToHexahedron(float3 p, GLVertex* quad1, GLVertex* quad2) {
        float min_dist = 1e12f;

        // 检查所有三角面
        for(int i = 0; i < 12; ++i) {
            float3 tri[3];
            switch(i) {
                case 0: // 底面三角形1
                    tri[0] = make_float3(quad1[0].x, quad1[0].y, quad1[0].z);
                    tri[1] = make_float3(quad1[1].x, quad1[1].y, quad1[1].z);
                    tri[2] = make_float3(quad1[2].x, quad1[2].y, quad1[2].z);
                    break;
                case 1: // 底面三角形2
                    tri[0] = make_float3(quad1[0].x, quad1[0].y, quad1[0].z);
                    tri[1] = make_float3(quad1[2].x, quad1[2].y, quad1[2].z);
                    tri[2] = make_float3(quad1[3].x, quad1[3].y, quad1[3].z);
                    break;
                case 2: // 顶面三角形1
                    tri[0] = make_float3(quad2[0].x, quad2[0].y, quad2[0].z);
                    tri[1] = make_float3(quad2[1].x, quad2[1].y, quad2[1].z);
                    tri[2] = make_float3(quad2[2].x, quad2[2].y, quad2[2].z);
                    break;
                case 3: // 顶面三角形2
                    tri[0] = make_float3(quad2[0].x, quad2[0].y, quad2[0].z);
                    tri[1] = make_float3(quad2[2].x, quad2[2].y, quad2[2].z);
                    tri[2] = make_float3(quad2[3].x, quad2[3].y, quad2[3].z);
                    break;
                case 4: // 侧面四边形1-三角形1
                    tri[0] = make_float3(quad1[0].x, quad1[0].y, quad1[0].z);
                    tri[1] = make_float3(quad1[1].x, quad1[1].y, quad1[1].z);
                    tri[2] = make_float3(quad2[1].x, quad2[1].y, quad2[1].z);
                    break;
                case 5: // 侧面四边形1-三角形2
                    tri[0] = make_float3(quad1[0].x, quad1[0].y, quad1[0].z);
                    tri[1] = make_float3(quad2[1].x, quad2[1].y, quad2[1].z);
                    tri[2] = make_float3(quad2[0].x, quad2[0].y, quad2[0].z);
                    break;
                case 6: // 侧面四边形2-三角形1
                    tri[0] = make_float3(quad1[1].x, quad1[1].y, quad1[1].z);
                    tri[1] = make_float3(quad1[2].x, quad1[2].y, quad1[2].z);
                    tri[2] = make_float3(quad2[2].x, quad2[2].y, quad2[2].z);
                    break;
                case 7: // 侧面四边形2-三角形2
                    tri[0] = make_float3(quad1[1].x, quad1[1].y, quad1[1].z);
                    tri[1] = make_float3(quad2[2].x, quad2[2].y, quad2[2].z);
                    tri[2] = make_float3(quad2[1].x, quad2[1].y, quad2[1].z);
                    break;
                case 8: // 侧面四边形3-三角形1
                    tri[0] = make_float3(quad1[2].x, quad1[2].y, quad1[2].z);
                    tri[1] = make_float3(quad1[3].x, quad1[3].y, quad1[3].z);
                    tri[2] = make_float3(quad2[3].x, quad2[3].y, quad2[3].z);
                    break;
                case 9: // 侧面四边形3-三角形2
                    tri[0] = make_float3(quad1[2].x, quad1[2].y, quad1[2].z);
                    tri[1] = make_float3(quad2[3].x, quad2[3].y, quad2[3].z);
                    tri[2] = make_float3(quad2[2].x, quad2[2].y, quad2[2].z);
                    break;
                case 10: // 侧面四边形4-三角形1
                    tri[0] = make_float3(quad1[3].x, quad1[3].y, quad1[3].z);
                    tri[1] = make_float3(quad1[0].x, quad1[0].y, quad1[0].z);
                    tri[2] = make_float3(quad2[0].x, quad2[0].y, quad2[0].z);
                    break;
                case 11: // 侧面四边形4-三角形2
                    tri[0] = make_float3(quad1[3].x, quad1[3].y, quad1[3].z);
                    tri[1] = make_float3(quad2[0].x, quad2[0].y, quad2[0].z);
                    tri[2] = make_float3(quad2[3].x, quad2[3].y, quad2[3].z);
                    break;
            }

            float3 normal = calculatePlaneNormal(tri[0], tri[1], tri[2]);
            bool in_triangle = pointInTriangle3D(p, tri[0], tri[1], tri[2]);
            float face_dist = fabsf(dot3(normal, make_float3(p.x - tri[0].x, p.y - tri[0].y, p.z - tri[0].z)));

            if (in_triangle) {
                // 点在三角面内部，直接使用面距离
                min_dist = fminf(min_dist, face_dist);
            } else {
                // 遍历三角形的三条边
                for(int j = 0; j < 3; ++j) {
                    int k = (j + 1) % 3;
                    float3 a = tri[j];
                    float3 b = tri[k];

                    // 计算线段参数t
                    float3 ab = make_float3(b.x-a.x, b.y-a.y, b.z-a.z);
                    float3 ap = make_float3(p.x-a.x, p.y-a.y, p.z-a.z);
                    float t = dot3(ap, ab) / dot3(ab, ab);

                    if (t >= 0.0f && t <= 1.0f) {
                        // 投影在线段内部
                        min_dist = fminf(min_dist, pointToSegmentDist3D(p, a, b));
                    } else {
                        // 投影在线段外部，计算到两个端点的距离
                        float dist_a = length(make_float3(p.x-a.x, p.y-a.y, p.z-a.z));
                        float dist_b = length(make_float3(p.x-b.x, p.y-b.y, p.z-b.z));
                        min_dist = fminf(min_dist, fminf(dist_a, dist_b));
                    }
                }
            }
        }
        return min_dist;
    }

    __device__ float distanceToCutEdge(float3 p, GLVertex* quad1, GLVertex* quad2) {
        float min_dist = 1e12f;

        // 检查所有三角面
        for(int i = 0; i < 2; ++i) {
            float3 tri[3];
            switch(i) {
                case 0: // 侧面四边形2-三角形1
                    tri[0] = make_float3(quad1[1].x, quad1[1].y, quad1[1].z);
                    tri[1] = make_float3(quad1[2].x, quad1[2].y, quad1[2].z);
                    tri[2] = make_float3(quad2[2].x, quad2[2].y, quad2[2].z);
                    break;
                case 1: // 侧面四边形2-三角形2
                    tri[0] = make_float3(quad1[1].x, quad1[1].y, quad1[1].z);
                    tri[1] = make_float3(quad2[2].x, quad2[2].y, quad2[2].z);
                    tri[2] = make_float3(quad2[1].x, quad2[1].y, quad2[1].z);
                    break;
            }

            float3 normal = calculatePlaneNormal(tri[0], tri[1], tri[2]);
            bool in_triangle = pointInTriangle3D(p, tri[0], tri[1], tri[2]);
            float face_dist = fabsf(dot3(normal, make_float3(p.x - tri[0].x, p.y - tri[0].y, p.z - tri[0].z)));

            if (in_triangle) {
                // 点在三角面内部，直接使用面距离
                min_dist = fminf(min_dist, face_dist);
            } else {
                // 遍历三角形的三条边
                for(int j = 0; j < 3; ++j) {
                    int k = (j + 1) % 3;
                    float3 a = tri[j];
                    float3 b = tri[k];

                    // 计算线段参数t
                    float3 ab = make_float3(b.x-a.x, b.y-a.y, b.z-a.z);
                    float3 ap = make_float3(p.x-a.x, p.y-a.y, p.z-a.z);
                    float t = dot3(ap, ab) / dot3(ab, ab);

                    if (t >= 0.0f && t <= 1.0f) {
                        // 投影在线段内部
                        min_dist = fminf(min_dist, pointToSegmentDist3D(p, a, b));
                    } else {
                        // 投影在线段外部，计算到两个端点的距离
                        float dist_a = length(make_float3(p.x-a.x, p.y-a.y, p.z-a.z));
                        float dist_b = length(make_float3(p.x-b.x, p.y-b.y, p.z-b.z));
                        min_dist = fminf(min_dist, fminf(dist_a, dist_b));
                    }
                }
            }
        }
        return min_dist;
    }


    // 修改后的blade_diff_kernel
    __global__ void blade_diff_kernel(
        CudaNodeData* nodes, int numNodes, BladeParams* blade,
        float* z_array, float* dmin_array, int* record_count, int max_records)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int nodeIdx = idx / 8;
        int vertexIdx = idx % 8;

        if (nodeIdx >= numNodes) return;

        float x = nodes[nodeIdx].x[vertexIdx];
        float y = nodes[nodeIdx].y[vertexIdx];
        float z = nodes[nodeIdx].z[vertexIdx];
        float3 p = {x, y, z};

        float min_dist = 1e12f;
        bool inside = false;
        int inside_index;

        // 遍历所有相邻四边形对
        for(int i = 0; i < blade->blade_points_count - 4; i += 4) {
            GLVertex* quad1 = &blade->blade_points[i];
            GLVertex* quad2 = &blade->blade_points[i+4];

            if(pointInHexahedron(p, quad1, quad2)) {
                min_dist =  distanceToHexahedron(p, quad1, quad2);
                if(min_dist<blade->cube_resolution){
                    min_dist = distanceToCutEdge(p, quad1, quad2);
                }
                inside_index=i;
                inside = true;
                break;
            }
            else{
                min_dist = fminf(min_dist, distanceToHexahedron(p, quad1, quad2));
            }
        }

        // 符号距离计算
        float signed_dist = inside ? min_dist : -min_dist;

        //printf("%f\n",signed_dist);

        // 记录结果
        if (inside) {
            //printf("inside\n");
            int rec_idx = atomicAdd(record_count, 1);
            if (rec_idx < max_records) {
                z_array[rec_idx] = inside_index;
                dmin_array[rec_idx] = signed_dist;
            }
        }
        else{
            //printf("outside\n");
        }

        // 更新节点距离值
        nodes[nodeIdx].f[vertexIdx] = signed_dist;
    }

    // 主接口函数，从C++代码调用
    extern "C" void cuda_diff_volume(CudaNodeData* host_nodes, int numNodes, VolumeParams host_volume) {
        // 添加调试输出
        //printf("CUDA函数开始执行，处理 %d 个节点\n", numNodes);
        //printf("host_nodes: %p\n", host_nodes);
        //printf("node_count: %d\n", numNodes);
        //fflush(stdout); // 强制刷新标准输出

        // 检查输入参数
        if (host_nodes == NULL || numNodes <= 0) {
            fprintf(stderr, "CUDA Error: 无效的输入参数\n");
            fflush(stderr);
            return;
        }

        // 检查节点数量是否过大
        size_t requiredMemory = numNodes * sizeof(CudaNodeData);
        size_t freeMemory, totalMemory;
        cudaMemGetInfo(&freeMemory, &totalMemory);
        //printf("需要的GPU内存: %zu 字节, 可用GPU内存: %zu 字节\n", requiredMemory, freeMemory);
        fflush(stdout);

        if (requiredMemory > freeMemory) {
            fprintf(stderr, "CUDA Error: GPU内存不足，需要 %zu 字节，但只有 %zu 字节可用\n",
                requiredMemory, freeMemory);
            fflush(stderr);
        }

        // 分配GPU内存
        CudaNodeData* dev_nodes = NULL;
        VolumeParams* dev_volume = NULL;

        cudaError_t err;

        // 分配设备内存
        err = cudaMalloc((void**)&dev_nodes, numNodes * sizeof(CudaNodeData));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error (node data alloc): %s\n", cudaGetErrorString(err));
            fflush(stderr);
            return;
        }
        //printf("成功分配设备内存 dev_nodes: %p\n", dev_nodes);
        //fflush(stdout);

        err = cudaMalloc((void**)&dev_volume, sizeof(VolumeParams));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error (volume params alloc): %s\n", cudaGetErrorString(err));
            fflush(stderr);
            cudaFree(dev_nodes);
            return;
        }
        //printf("成功分配设备内存 dev_volume: %p\n", dev_volume);
        //fflush(stdout);

        // 拷贝数据到GPU
        err = cudaMemcpy(dev_nodes, host_nodes, numNodes * sizeof(CudaNodeData), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error (node data copy): %s\n", cudaGetErrorString(err));
            fflush(stderr);
            cudaFree(dev_nodes);
            cudaFree(dev_volume);
            return;
        }
        //printf("成功拷贝节点数据到设备\n");
        //fflush(stdout);

        err = cudaMemcpy(dev_volume, &host_volume, sizeof(VolumeParams), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error (volume params copy): %s\n", cudaGetErrorString(err));
            fflush(stderr);
            cudaFree(dev_nodes);
            cudaFree(dev_volume);
            return;
        }
        //printf("成功拷贝体积参数到设备\n");
    //    //printf("体积类型: %d\n", host_volume.type);

        // 计算网格和块大小
        int threadsPerBlock = 256;
        int totalThreads = numNodes * 8;
        // CUDA 2080Ti gridDim.x 最大为 2,147,483,647，实际建议略小
        const int maxThreadsPerLaunch = 100'000'000; // 可根据实际情况调整
        int threadsProcessed = 0;

        while (threadsProcessed < totalThreads) {
            int threadsThisBatch = totalThreads - threadsProcessed;
            if (threadsThisBatch > maxThreadsPerLaunch) {
                threadsThisBatch = maxThreadsPerLaunch;
            }
            int grid = (threadsThisBatch + threadsPerBlock - 1) / threadsPerBlock;

            printf("启动CUDA核函数:本批次%d个线程，gpu块分配：%d blocks, 每block %d threads\n", threadsThisBatch, grid, threadsPerBlock);
            fflush(stdout);

            // kernel 内部索引加上偏移
            diff_kernel << <grid, threadsPerBlock >> > (
                reinterpret_cast<CudaNodeData*>((char*)dev_nodes + (threadsProcessed / 8) * sizeof(CudaNodeData)),
                threadsThisBatch / 8,
                dev_volume
                );

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error (kernel execution): %s\n", cudaGetErrorString(err));
                fflush(stderr);
                cudaFree(dev_nodes);
                cudaFree(dev_volume);
                return;
            }

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error (synchronize): %s\n", cudaGetErrorString(err));
                fflush(stderr);
                cudaFree(dev_nodes);
                cudaFree(dev_volume);
                return;
            }

            threadsProcessed += threadsThisBatch;
        }


        //   printf("核函数执行完成\n");
         //  fflush(stdout);

           // 拷贝结果回CPU
        err = cudaMemcpy(host_nodes, dev_nodes, numNodes * sizeof(CudaNodeData), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error (result copy): %s\n", cudaGetErrorString(err));
            fflush(stderr);
            cudaFree(dev_nodes);
            cudaFree(dev_volume);
            return;
        }
        //printf("成功拷贝结果回主机\n");
        fflush(stdout);

        // 释放GPU内存
        cudaFree(dev_nodes);
        cudaFree(dev_volume);

        //printf("CUDA函数执行完成\n");
        //printf("\n");
        fflush(stdout);
    }

    extern "C" void cuda_diff_volume_blade(CudaNodeData* host_nodes, int numNodes, BladeParams host_blade,
        float** host_z_array, float** host_dmin_array, int* host_record_count) {

        // 检查输入参数
        if (host_nodes == NULL || numNodes <= 0) {
            fprintf(stderr, "CUDA Error: 无效的输入参数\n");
            fflush(stderr);
            return;
        }

        // 检查节点数量是否过大
        size_t requiredMemory = numNodes * sizeof(CudaNodeData);
        size_t freeMemory, totalMemory;
        cudaMemGetInfo(&freeMemory, &totalMemory);
        //printf("需要的GPU内存: %zu 字节, 可用GPU内存: %zu 字节\n", requiredMemory, freeMemory);
        fflush(stdout);

        if (requiredMemory > freeMemory) {
            fprintf(stderr, "CUDA Error: GPU内存不足，需要 %zu 字节，但只有 %zu 字节可用\n",
                requiredMemory, freeMemory);
            fflush(stderr);
        }

        // 分配GPU内存
        CudaNodeData* dev_nodes = NULL;
        BladeParams* dev_volume = NULL;

        cudaError_t err;

        // 分配设备内存
        cudaMalloc((void**)&dev_nodes, numNodes * sizeof(CudaNodeData));

        // 处理blade_points数据
        GLVertex* dev_blade_points = nullptr;
        // 分配blade_points内存
        if (host_blade.blade_points_count > 0) {
            err = cudaMalloc((void**)&dev_blade_points, host_blade.blade_points_count * sizeof(GLVertex));
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error (blade points alloc): %s\n", cudaGetErrorString(err));
                cudaFree(dev_nodes);
                return;
            }
            // 拷贝数据到设备
            err = cudaMemcpy(dev_blade_points, host_blade.blade_points,
                host_blade.blade_points_count * sizeof(GLVertex), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error (blade points copy): %s\n", cudaGetErrorString(err));
                cudaFree(dev_nodes);
                cudaFree(dev_blade_points);
                return;
            }
            host_blade.blade_points = dev_blade_points; // 更新指针
        }

        // 4. 分配BladeParams内存
        err = cudaMalloc((void**)&dev_volume, sizeof(BladeParams));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error (blade params alloc): %s\n", cudaGetErrorString(err));
            cudaFree(dev_nodes);
            if (dev_blade_points) cudaFree(dev_blade_points);
            return;
        }

        // 5. 拷贝BladeParams到设备
        err = cudaMemcpy(dev_volume, &host_blade, sizeof(BladeParams), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error (blade params copy): %s\n", cudaGetErrorString(err));
            cudaFree(dev_nodes);
            if (dev_blade_points) cudaFree(dev_blade_points);
            cudaFree(dev_volume);
            return;
        }

        // 拷贝数据到GPU
        cudaMemcpy(dev_nodes, host_nodes, numNodes * sizeof(CudaNodeData), cudaMemcpyHostToDevice);


        // 计算网格和块大小
        int threadsPerBlock = 256;
        int totalThreads = numNodes * 8;
        const int maxThreadsPerLaunch = 100'000'000; // 可根据实际情况调整
        int threadsProcessed = 0;

        //分配z和d_min数组及计数器
        float* dev_z_array = nullptr;
        float* dev_dmin_array = nullptr;
        int* dev_record_count = nullptr;
        int max_records = numNodes * 8; // 或更大
        cudaMalloc(&dev_z_array, max_records * sizeof(float));
        cudaMalloc(&dev_dmin_array, max_records * sizeof(float));
        cudaMalloc(&dev_record_count, sizeof(int));
        cudaMemset(dev_record_count, 0, sizeof(int));

        while (threadsProcessed < totalThreads) {
            int threadsThisBatch = totalThreads - threadsProcessed;
            if (threadsThisBatch > maxThreadsPerLaunch) {
                threadsThisBatch = maxThreadsPerLaunch;
            }
            int grid = (threadsThisBatch + threadsPerBlock - 1) / threadsPerBlock;

            printf("启动CUDA核函数:本批次%d个线程，gpu块分配：%d blocks, 每block %d threads\n", threadsThisBatch, grid, threadsPerBlock);
            fflush(stdout);

            cudaSetDevice(host_blade.device_id);

            blade_diff_kernel << <grid, threadsPerBlock >> > (
                reinterpret_cast<CudaNodeData*>((char*)dev_nodes + (threadsProcessed / 8) * sizeof(CudaNodeData)),
                threadsThisBatch / 8,
                dev_volume,
                dev_z_array, dev_dmin_array, dev_record_count, max_records
                );


            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error (kernel execution): %s\n", cudaGetErrorString(err));
                fflush(stderr);
                cudaFree(dev_nodes);
                cudaFree(dev_volume);
                return;
            }

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error (synchronize): %s\n", cudaGetErrorString(err));
                fflush(stderr);
                cudaFree(dev_nodes);
                cudaFree(dev_volume);
                return;
            }

            threadsProcessed += threadsThisBatch;
        }

        // 拷贝记录数
        cudaMemcpy(host_record_count, dev_record_count, sizeof(int), cudaMemcpyDeviceToHost);

        // 分配主机内存
        *host_z_array = (float*)malloc((*host_record_count) * sizeof(float));
        *host_dmin_array = (float*)malloc((*host_record_count) * sizeof(float));

        // 拷贝z和d_min数组
        cudaMemcpy(*host_z_array, dev_z_array, (*host_record_count) * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(*host_dmin_array, dev_dmin_array, (*host_record_count) * sizeof(float), cudaMemcpyDeviceToHost);

        // 释放
        cudaFree(dev_z_array);
        cudaFree(dev_dmin_array);
        cudaFree(dev_record_count);

        err = cudaMemcpy(host_nodes, dev_nodes, numNodes * sizeof(CudaNodeData), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error (result copy): %s\n", cudaGetErrorString(err));
            fflush(stderr);
            cudaFree(dev_nodes);
            cudaFree(dev_volume);
            return;
        }
        //printf("成功拷贝结果回主机\n");
        fflush(stdout);

        // 释放GPU内存
        cudaFree(dev_nodes);
        cudaFree(dev_volume);
        cudaFree(dev_blade_points);

        //printf("CUDA函数执行完成\n");
        //printf("\n");
        fflush(stdout);

    }

}
