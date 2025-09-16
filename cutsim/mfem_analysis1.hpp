#ifndef MFEM_ANALYSIS1_HPP
#define MFEM_ANALYSIS1_HPP
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include "../cutsim/volume.hpp"

using namespace std;
using namespace mfem;

// 结构体用于存储模态参与因子计算结果
struct ModalParticipationFactors {
    double mass_factor_x, mass_factor_y, mass_factor_z;
    double stiff_factor_x, stiff_factor_y, stiff_factor_z;
    double mass_form, stiff_form;
    double node_x_value, node_y_value, node_z_value;
};

// 主要分析函数
void runMFEMAnalysis(const std::string& meshFile, std::vector<int>& nodeIds,std::vector<cutsim::AptCutterVolume::ForceData> collected_force_data);


ModalParticipationFactors calculateModalParticipationFactors(
    mfem::HypreLOBPCG* lobpcg,
    mfem::HypreParMatrix* A,
    mfem::HypreParMatrix* M,
    int mode_index,
    int node_id
);

void printModalParticipationFactors(
    mfem::HypreLOBPCG* lobpcg,
    mfem::HypreParMatrix* A,
    mfem::HypreParMatrix* M,
    int mode_index,
    int node_id
);

void saveModalParticipationFactorsToFile(
    mfem::HypreLOBPCG* lobpcg,
    mfem::HypreParMatrix* A,
    mfem::HypreParMatrix* M,
    int mode_index,
    const mfem::Array<int>& node_ids,
    const char* output_filename
);
#endif // MFEM_ANALYSIS1_HPP
