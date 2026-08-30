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
    ../../../source_base/module_grid/partition.cpp
    ../../../source_io/module_output/output.cpp
)

AddTest(
  TARGET MODULE_ESTATE_constraint_observe
  LIBS parameter ${math_libs} base device planewave_serial cell_info
  SOURCES constraint_observe_test.cpp
    ../constraint_observe.cpp
    ../weight_grid.cpp
    ../../../source_base/module_grid/partition.cpp
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
    ../../../source_base/module_grid/partition.cpp
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
    ../../../source_base/module_grid/partition.cpp
    ../../../source_io/module_output/output.cpp
)
