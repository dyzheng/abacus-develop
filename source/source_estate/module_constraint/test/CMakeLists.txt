remove_definitions(-D__MPI)
remove_definitions(-D__EXX)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)
remove_definitions(-D__MLALGO)

AddTest(
  TARGET MODULE_ESTATE_constraint_weight_grid
  LIBS parameter ${math_libs} base device planewave_serial cell_info
  SOURCES weight_grid_test.cpp
    ../weight_grid.cpp
    ../../../source_io/module_output/output.cpp
)

AddTest(
  TARGET MODULE_ESTATE_constraint_observe
  LIBS parameter ${math_libs} base device planewave_serial cell_info
  SOURCES constraint_observe_test.cpp
    ../constraint_observe.cpp
    ../weight_grid.cpp
    ../../../source_base/module_grid/radial.cpp
    ../../../source_base/module_grid/delley.cpp
    ../../../source_io/module_output/output.cpp
)

AddTest(
  TARGET MODULE_ESTATE_constraint_mu_solver
  LIBS parameter ${math_libs} base device
  SOURCES mu_solver_test.cpp ../mu_solver.cpp
)

AddTest(
  TARGET MODULE_ESTATE_constraint_io
  LIBS parameter ${math_libs} base device
  SOURCES constraint_io_test.cpp ../constraint_io.cpp
)

AddTest(
  TARGET MODULE_ESTATE_constraint_inject_pw
  LIBS parameter ${math_libs} base device planewave_serial cell_info
  SOURCES constraint_inject_pw_test.cpp
    ../constraint_inject_pw.cpp
    ../constraint_observe.cpp
    ../weight_grid.cpp
    ../../../source_base/module_grid/radial.cpp
    ../../../source_base/module_grid/delley.cpp
    ../../../source_io/module_output/output.cpp
)

AddTest(
  TARGET MODULE_ESTATE_constraint_accounting
  LIBS parameter ${math_libs} base device planewave_serial cell_info
  SOURCES constraint_accounting_test.cpp
    ../constraint_accounting.cpp
    ../weight_grid.cpp
    ../../../source_io/module_output/output.cpp
)

AddTest(
  TARGET MODULE_ESTATE_constraint_loop
  LIBS parameter ${math_libs} base device planewave_serial cell_info
  SOURCES constraint_loop_test.cpp
    ../constraint_loop.cpp
    ../constraint_accounting.cpp
    ../constraint_io.cpp
    ../constraint_inject_pw.cpp
    ../constraint_observe.cpp
    ../mu_solver.cpp
    ../weight_grid.cpp
    ../../../source_base/module_grid/radial.cpp
    ../../../source_base/module_grid/delley.cpp
    ../../../source_io/module_output/output.cpp
)

AddTest(
  TARGET MODULE_ESTATE_constraint_inject_lcao
  LIBS parameter ${math_libs} base device planewave_serial cell_info neighbor container
  SOURCES constraint_inject_lcao_test.cpp
    ../constraint_inject_lcao.cpp
    ../../../source_lcao/module_gint/biggrid_info.cpp
    ../../../source_lcao/module_gint/big_grid.cpp
    ../../../source_lcao/module_gint/divide_info.cpp
    ../../../source_lcao/module_gint/gint_atom.cpp
    ../../../source_lcao/module_gint/gint_info.cpp
    ../../../source_lcao/module_gint/gint.cpp
    ../../../source_lcao/module_gint/gint_vl.cpp
    ../../../source_lcao/module_gint/gint_vl_metagga.cpp
    ../../../source_lcao/module_gint/gint_vl_nspin4.cpp
    ../../../source_lcao/module_gint/gint_vl_metagga_nspin4.cpp
    ../../../source_lcao/module_gint/gint_rho.cpp
    ../../../source_lcao/module_gint/gint_tau.cpp
    ../../../source_lcao/module_gint/gint_fvl.cpp
    ../../../source_lcao/module_gint/gint_fvl_meta.cpp
    ../../../source_lcao/module_gint/gint_env_gamma.cpp
    ../../../source_lcao/module_gint/gint_env_k.cpp
    ../../../source_lcao/module_gint/gint_dvlocal.cpp
    ../../../source_lcao/module_gint/localcell_info.cpp
    ../../../source_lcao/module_gint/phi_operator.cpp
    ../../../source_lcao/module_gint/set_ddphi.cpp
    ../../../source_lcao/module_gint/unitcell_info.cpp
    ../../../source_lcao/module_gint/gint_common.cpp
    ../../../source_lcao/module_gint/gint_interface.cpp
    ../../../source_lcao/module_hcontainer/base_matrix.cpp
    ../../../source_lcao/module_hcontainer/atom_pair.cpp
    ../../../source_lcao/module_hcontainer/hcontainer.cpp
    ../../../source_lcao/module_hcontainer/func_transfer.cpp
    ../../../source_lcao/module_hcontainer/transfer.cpp
    ../../../source_basis/module_ao/ORB_atomic.cpp
    ../../../source_basis/module_ao/ORB_atomic_lm.cpp
    ../../../source_basis/module_ao/parallel_orbitals.cpp
    ../../../source_io/module_output/output.cpp
)
