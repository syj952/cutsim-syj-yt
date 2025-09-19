#include "mfem_analysis1.hpp"

using namespace std;
using namespace mfem;


// 计算指定模态和节点的局部质量参与因子和局部刚度参与因子
ModalParticipationFactors calculateModalParticipationFactors(
    HypreLOBPCG* lobpcg,           // LOBPCG求解器
    HypreParMatrix* A,             // 刚度矩阵
    HypreParMatrix* M,             // 质量矩阵
    int mode_index,                // 模态索引（从0开始）
    int node_id                    // 节点编号
) {
    ModalParticipationFactors result = {0};

    // 获取指定模态的特征向量
    HypreParVector x = lobpcg->GetEigenvector(mode_index);

    // 创建工作向量
    HypreParVector y_vec(MPI_COMM_WORLD, A->GetGlobalNumRows(), A->GetRowStarts());
    HypreParVector x_vec(MPI_COMM_WORLD, A->GetGlobalNumCols(), A->GetColStarts());
    HypreParVector z_vec(MPI_COMM_WORLD, M->GetGlobalNumRows(), M->GetRowStarts());

    // 转换格式：从ParGridFunction到HypreParVector
    for(int j = 0; j < A->GetGlobalNumCols(); j++) {
        double* x_vec_data = x_vec.GetData();
        const double* x_data = x.GetData();
        x_vec_data[j] = x_data[j];
    }

    // 获取节点在X、Y、Z方向的特征向量值
    const double* x_data = x.GetData();
    result.node_x_value = x_data[node_id * 3];     // X方向
    result.node_y_value = x_data[node_id * 3 + 1]; // Y方向
    result.node_z_value = x_data[node_id * 3 + 2]; // Z方向

    // 计算刚度矩阵的二次型：x^T * K * x
    A->Mult(x_vec, y_vec);
    result.stiff_form = 0;
    for(int j = 0; j < A->GetGlobalNumCols(); j++) {
        const double* y_data = y_vec.GetData();
        const double* x_data = x_vec.GetData();
        result.stiff_form += y_data[j] * x_data[j];
    }

    // 计算质量矩阵的二次型：x^T * M * x
    M->Mult(x_vec, z_vec);
    result.mass_form = 0;
    for(int j = 0; j < x_vec.Size(); j++) {
        const double* z_data = z_vec.GetData();
        const double* x_data = x_vec.GetData();
        result.mass_form += z_data[j] * x_data[j];
    }

    // 计算局部刚度参与因子（X、Y、Z方向）
    if (result.node_x_value != 0) {
        result.stiff_factor_x = result.stiff_form / (result.node_x_value * result.node_x_value);
    }
    if (result.node_y_value != 0) {
        result.stiff_factor_y = result.stiff_form / (result.node_y_value * result.node_y_value);
    }
    if (result.node_z_value != 0) {
        result.stiff_factor_z = result.stiff_form / (result.node_z_value * result.node_z_value);
    }

    // 计算局部质量参与因子（X、Y、Z方向）
    if (result.node_x_value != 0) {
        result.mass_factor_x = result.mass_form / (result.node_x_value * result.node_x_value);
    }
    if (result.node_y_value != 0) {
        result.mass_factor_y = result.mass_form / (result.node_y_value * result.node_y_value);
    }
    if (result.node_z_value != 0) {
        result.mass_factor_z = result.mass_form / (result.node_z_value * result.node_z_value);
    }

    return result;
}

void printModalParticipationFactors(
    HypreLOBPCG* lobpcg,
    HypreParMatrix* A,
    HypreParMatrix* M,
    int mode_index,
    int node_id
) {
    ModalParticipationFactors factors = calculateModalParticipationFactors(lobpcg, A, M, mode_index, node_id);


    cout << "=== 模态 " << mode_index << " 节点 " << node_id << " 的参与因子 ===" << endl;
    cout << "节点特征向量值:" << endl;
    cout << "  X方向: " << factors.node_x_value << endl;
    cout << "  Y方向: " << factors.node_y_value << endl;
    cout << "  Z方向: " << factors.node_z_value << endl;

    cout << "二次型值:" << endl;
    cout << "  刚度二次型 (x^T * K * x): " << factors.stiff_form << endl;
    cout << "  质量二次型 (x^T * M * x): " << factors.mass_form << endl;

    cout << "局部刚度参与因子:" << endl;
    cout << "  X方向: " << factors.stiff_factor_x << endl;
    cout << "  Y方向: " << factors.stiff_factor_y << endl;
    cout << "  Z方向: " << factors.stiff_factor_z << endl;

    cout << "局部质量参与因子:" << endl;
    cout << "  X方向: " << factors.mass_factor_x << endl;
    cout << "  Y方向: " << factors.mass_factor_y << endl;
    cout << "  Z方向: " << factors.mass_factor_z << endl;
    cout << "========================================" << endl;
}

void saveModalParticipationFactorsToFile(
    HypreLOBPCG* lobpcg,
    HypreParMatrix* A,
    HypreParMatrix* M,
    int mode_index,
    const Array<int>& node_ids,
    const char* output_filename
) {
    ofstream output_file(output_filename);
    if (!output_file.is_open()) {
        cerr << "无法打开输出文件: " << output_filename << endl;
        return;
    }

    // 写入CSV文件头
    output_file << "node_id,mass_factor_x,mass_factor_y,mass_factor_z,stiff_factor_x,stiff_factor_y,stiff_factor_z" << endl;

    for (int i = 0; i < node_ids.Size(); i++) {
        int node_id = node_ids[i];
        ModalParticipationFactors factors = calculateModalParticipationFactors(
            lobpcg, A, M, mode_index, node_id
        );

        output_file << node_id << ","
                   << factors.mass_factor_x << ","
                   << factors.mass_factor_y << ","
                   << factors.mass_factor_z << ","
                   << factors.stiff_factor_x << ","
                   << factors.stiff_factor_y << ","
                   << factors.stiff_factor_z << endl;
    }

    output_file.close();
    cout << "模态参与因子数据已保存到文件: " << output_filename << endl;
}


void runMFEMAnalysis(const std::string& meshFile, std::vector<int>& eigenvalues,std::vector<cutsim::AptCutterVolume::ForceData> collected_force_data)

{        int argc = 1;
         char prog_name[] = "mfem_analysis";
         char* argv_array[] = {prog_name, nullptr};
         char** argv = argv_array;  // 创建一个 char** 类型的变量
    // 1. Initialize MPI and HYPRE
    Mpi::Init(argc, argv);
    int num_procs = Mpi::WorldSize();
    cout<<"num of procs: " << num_procs << endl;
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 2. Parse command-line options
    const char *mesh_file = meshFile.c_str(); //"/home/emstan/Downloads/mfem-master/data/beam-hex.mesh";
    int order = 1;
    int nev = 3;
    int seed = 4;
    Array<int> node_id;
    for (int i = 0; i < eigenvalues.size(); i++)
    {node_id.Append(eigenvalues[i]);}
    bool static_cond = false;
    bool visualization = true;
    bool amg_elast = false;
    bool perform_modal = true;
    bool perform_static = true;
    bool apply_force_for_static = true;
    bool reorder_space = false;
    const char *device_config = "cpu";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&order, "-o", "--order", "Finite element order (polynomial degree).");
    args.AddOption(&nev, "-n", "--num-eigs", "Number of desired eigenmodes.");
    args.AddOption(&seed, "-s", "--seed", "Random seed used to initialize LOBPCG.");
    //args.AddOption(&node_id, "-nid", "--node-id", "Node ID for modal participation factor calculation.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&amg_elast, "-elast", "--amg-for-elasticity", "-sys",
                   "--amg-for-systems",
                   "Use the special AMG elasticity solver (GM/LN approaches), "
                   "or standard AMG for systems (unknown approach).");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization", "Enable or disable GLVis visualization.");
    args.AddOption(&perform_modal, "-modal", "--perform-modal", "-no-modal",
                   "--no-perform-modal", "Perform modal analysis.");
    args.AddOption(&perform_static, "-static", "--perform-static", "-no-static",
                   "--no-perform-static", "Perform static analysis.");
    args.AddOption(&apply_force_for_static, "-sf", "--apply-force-for-static", "-no-sf",
                   "--no-apply-force-for-static", "Apply force boundary conditions for static analysis.");
    args.AddOption(&reorder_space, "-nodes", "--by-nodes", "-vdim", "--by-vdim",
                   "Use byNODES ordering of vector space instead of byVDIM");
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }

    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    // 3. Enable hardware devices
    Device device(device_config);
    if (myid == 0) { device.Print(); }

    // 4. Read the mesh
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    if ( mesh->bdr_attributes.Max() < 2)
    {
        if (myid == 0)
            cerr << "\nInput mesh should have at least two materials and "
                 << "two boundary attributes! (See schematic in ex2.cpp)\n"
                 << endl;

    }
  // 7. Create parallel mesh
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    //    // 5. Handle NURBS meshes
    //    if (mesh->NURBSext)
    //    {
    //        mesh->DegreeElevate(order, order);
    //        cout<< "numbrs mesh" <<endl;
    //    }

    //    // 6. Refine the mesh
    //    {
    //        int ref_levels = (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
    //        for (int l = 0; l < ref_levels; l++)
    //        {
    //            mesh->UniformRefinement();
    //        }
    //    }


    //    {
    //        int par_ref_levels = 0;
    //        for (int l = 0; l < par_ref_levels; l++)
    //        {
    //            pmesh->UniformRefinement();
    //        }
    //    }

    // 8. Define finite element space
    FiniteElementCollection *fec;
    ParFiniteElementSpace *fespace;
    const bool use_nodal_fespace = pmesh->NURBSext && !amg_elast;
    if (use_nodal_fespace)
    {
        fec = NULL;
        fespace = (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace();
    }
    else
    {
        fec = new H1_FECollection(order, dim);
        if (reorder_space)
        {
            fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byNODES);
        }
        else
        {
            fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
        }
    }
    HYPRE_BigInt size = fespace->GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of unknowns: " << size << endl
             << "Assembling: " << flush;
    }

    // 9. Set up material parameters of steel
    double E = 200.0e3;  // Young's modulus
    double nu = 0.3;     // Poisson's ratio
    double density = 7850.0;  // Density

    Vector lambda(pmesh->attributes.Max());
    lambda = E*nu/(1.0+nu)/(1.0-2.0*nu);
    PWConstCoefficient lambda_func(lambda);

    Vector mu(pmesh->attributes.Max());
    mu = E/2.0/(1.0+nu);
    PWConstCoefficient mu_func(mu);

    if (myid == 0)
    {
        cout << "lambda = " << lambda[0] << ", mu = " << mu[0] << endl;
    }

    // 10. Set up boundary conditions
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;  // Fix boundary attribute 1

    // 11. Set up bilinear forms (stiffness and mass matrices)
    ParBilinearForm *a = new ParBilinearForm(fespace);
    a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));
    if (static_cond) { a->EnableStaticCondensation(); }
    if (myid == 0) { cout << "stiffness matrix ... " << flush; }
    a->Assemble();
    a->EliminateEssentialBCDiag(ess_bdr, 1.0);
    a->Finalize();

    ParBilinearForm *m = new ParBilinearForm(fespace);
    Coefficient *density_coeff = new ConstantCoefficient(density);
    m->AddDomainIntegrator(new VectorMassIntegrator(*density_coeff));
    m->Assemble();
    m->EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
    m->Finalize();
    if (myid == 0) { cout << "mass matrix ... done." << endl; }

    HypreParMatrix *A = a->ParallelAssemble();
    HypreParMatrix *M = m->ParallelAssemble();

    // 12. Perform Modal Analysis
    if (perform_modal)
    {
        if (myid == 0)
        {
            cout << "\n=== 开始模态分析 ===" << endl;
        }

        // Set up LOBPCG eigensolver
        HypreBoomerAMG *amg = new HypreBoomerAMG(*A);
        amg->SetPrintLevel(0);
        if (amg_elast)
        {
            amg->SetElasticityOptions(fespace);
        }
        else
        {
            amg->SetSystemsOptions(dim);
        }

        HypreLOBPCG *lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);
        lobpcg->SetNumModes(nev);
        lobpcg->SetRandomSeed(seed);
        lobpcg->SetPreconditioner(*amg);
        lobpcg->SetMaxIter(30);
        lobpcg->SetTol(1e-1);
        lobpcg->SetPrecondUsageMode(1);
        lobpcg->SetPrintLevel(1);
        lobpcg->SetMassMatrix(*M);
        lobpcg->SetOperator(*A);

        // Solve eigenvalue problem
        Array<double> eigenvalues;
        lobpcg->Solve();
        lobpcg->GetEigenvalues(eigenvalues);
        ParGridFunction x_modal(fespace);

        // Print eigenvalues and modal participation factors
        if (myid == 0)
        {
            cout << "solve finished" << endl;
            cout << "\n特征值和频率:" << endl;
            for (int i = 0; i < nev; i++)
            {
                double freq = sqrt(eigenvalues[i]) / (2.0 * M_PI);
                cout << "模态 " << i << ": 特征值 = " << eigenvalues[i]
                        << ", 频率 = " << freq << " Hz" << endl;
            }
        }

        // Calculate modal participation factors for first mode
        if (nev > 0)
        {
           saveModalParticipationFactorsToFile(lobpcg, A, M,0,node_id,"modal_participation_factors.csv");
        }

        //        // Save modal results
        //        if (!use_nodal_fespace)
        //        {
        //            pmesh->SetNodalFESpace(fespace);
        //        }

        //        for (int i = 0; i < min(nev, 3); i++)  // Save first 3 modes
        //        {
        //            x_modal = lobpcg->GetEigenvector(i);
        //            ostringstream mode_name;
        //            mode_name << "modal_mode_" << setfill('0') << setw(2) << i << "."
        //                      << setfill('0') << setw(6) << myid;
        //            ofstream mode_ofs(mode_name.str().c_str());
        //            mode_ofs.precision(8);
        //            x_modal.Save(mode_ofs);
        //        }


    }

    // 13. Perform Static Analysis
    if (perform_static)
    {
        if (myid == 0)
        {
            cout << "static solve" << endl;
            cout << "\n=== 开始静力分析 ===" << endl;
        }

        // Set up load vector
        ParLinearForm *b = new ParLinearForm(fespace);
        VectorArrayCoefficient f(dim);

        if (apply_force_for_static)
        {
            // Apply boundary force (similar to ex2p)

            for (int i = 0; i < dim-1; i++)
            {
                f.Set(i, new ConstantCoefficient(0.0));
            }
            //x方向
            {
                Vector pull_force_x(pmesh->bdr_attributes.Max());
                pull_force_x = 0.0;
                for(int i=2;i<=pull_force_x.Size();i++)
                {
                pull_force_x(i) = collected_force_data[i-2].force_value.x;  // Apply downward force on boundary attribute 2
                }
                f.Set(0, new PWConstCoefficient(pull_force_x));
            }
            //y方向
            {
                Vector pull_force_y(pmesh->bdr_attributes.Max());
                pull_force_y = 0.0;
                for(int i=2;i<=pull_force_y.Size();i++)
                {
                pull_force_y(i) = collected_force_data[i-2].force_value.y;  // Apply downward force on boundary attribute 2
                }
                f.Set(1, new PWConstCoefficient(pull_force_y));
            }
            //z方向
            {
                Vector pull_force_z(pmesh->bdr_attributes.Max());
                pull_force_z = 0.0;
                for(int i=2;i<=pull_force_z.Size();i++)
                {
                pull_force_z(i) = collected_force_data[i-2].force_value.z;  // Apply downward force on boundary attribute 2
                }
                f.Set(2, new PWConstCoefficient(pull_force_z));
            }
            b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
        }

        if (myid == 0) { cout << "right-hand side ... " << flush; }
        b->Assemble();
        cout<<"b assembled" <<endl;
        // Set up solution vector
        ParGridFunction x_static(fespace);
        x_static = 0.0;

        // Get essential DOFs
        Array<int> ess_tdof_list;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        // Form linear system
        HypreParMatrix A_static;
        Vector B, X;
        const int copy_interior = 1;
        a->FormLinearSystem(ess_tdof_list, x_static, *b, A_static, X, B, copy_interior);

        if (myid == 0)
        {
            cout << "done." << endl;
            cout << "Size of linear system: " << A_static.GetGlobalNumRows() << endl;
        }

        // Set up solver
        HypreBoomerAMG *amg_static = new HypreBoomerAMG(A_static);
        if (amg_elast && !a->StaticCondensationIsEnabled())
        {
            amg_static->SetElasticityOptions(fespace);
        }
        else
        {
            amg_static->SetSystemsOptions(dim, reorder_space);
        }

        HyprePCG *pcg = new HyprePCG(A_static);
        pcg->SetTol(1e-1);
        pcg->SetMaxIter(20);
        pcg->SetPrintLevel(2);
        pcg->SetPreconditioner(*amg_static);
        pcg->Mult(B, X);
        cout << "static solution done." << endl;
        // Recover solution
        a->RecoverFEMSolution(X, *b, x_static);
        cout << "x static got." << endl;
        // Save static results
        // 14. For non-NURBS meshes, make the mesh curved based on the finite element
        //     space. This means that we define the mesh elements through a fespace
        //     based transformation of the reference element.  This allows us to save
        //     the displaced mesh as a curved mesh when using high-order finite
        //     element displacement field. We assume that the initial mesh (read from
        //     the file) is not higher order curved mesh compared to the chosen FE
        //     space.
        if (!use_nodal_fespace)
        {
            pmesh->SetNodalFESpace(fespace);
        }

        {
            GridFunction *nodes = pmesh->GetNodes();
            *nodes += x_static;
            x_static *= -1;

            ostringstream mesh_name, sol_name;
            mesh_name << "static_mesh." << setfill('0') << setw(6) << myid;
            sol_name << "static_sol." << setfill('0') << setw(6) << myid;

            ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            pmesh->Print(mesh_ofs);

            ofstream sol_ofs(sol_name.str().c_str());
            sol_ofs.precision(8);
            x_static.Save(sol_ofs);
        }

        // Visualization
        if (visualization)
        {
            char vishost[] = "localhost";
            int visport = 19916;
            socketstream sol_sock(vishost, visport);
            sol_sock << "parallel " << num_procs << " " << myid << "\n";
            sol_sock.precision(8);
            sol_sock << "solution\n" << *pmesh << x_static << flush;
        }

        delete pcg;
        delete amg_static;
        delete b;
    }

    // 14. Clean up
    delete A;
    delete M;
    delete a;
    delete m;
    delete density_coeff;
    //delete lobpcg;
    //delete amg;
    //    if (fec)
    //    {
    //        delete fespace;
    //        delete fec;
    //    }
    //    delete pmesh;

}









