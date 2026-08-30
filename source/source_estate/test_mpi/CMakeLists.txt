remove_definitions(-D__EXX)
remove_definitions(-D__CUDA)
remove_definitions(-D__UT_USE_CUDA)
remove_definitions(-D__UT_USE_ROCM)
remove_definitions(-D__ROCM)
remove_definitions(-D__MLALGO)
remove_definitions(-D_OPENMP)

AddTest(
  TARGET MODULE_ESTATE_charge_mpi_test
  LIBS parameter ${math_libs} psi base device planewave 
  SOURCES charge_mpi_test.cpp ../module_charge/charge_mpi.cpp
)

add_test(NAME MODULE_ESTATE_charge_mpi_test_4np
      COMMAND mpirun -np 4 ./MODULE_ESTATE_charge_mpi_test;
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

AddTest(
  TARGET MODULE_ESTATE_constraint_weight_grid_mpi
  LIBS parameter ${math_libs} base device planewave cell_info
  SOURCES weight_grid_mpi_test.cpp
    ../module_constraint/weight_grid.cpp
    ../module_constraint/constraint_observe.cpp
    ../../source_base/module_grid/partition.cpp
    ../../source_io/module_output/output.cpp
)

add_test(NAME MODULE_ESTATE_constraint_weight_grid_mpi_4np
      COMMAND mpirun -np 4 ./MODULE_ESTATE_constraint_weight_grid_mpi;
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
