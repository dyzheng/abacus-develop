remove_definitions(-D__EXX)
remove_definitions(-D__CUDA)
remove_definitions(-D__UT_USE_CUDA)
remove_definitions(-D__UT_USE_ROCM)
remove_definitions(-D__ROCM)
remove_definitions(-D__MLALGO)
remove_definitions(-D_OPENMP)

# CPU-only builds need the device stubs; CUDA/ROCm get the real symbols from
# the device object library (adding the stubs there would duplicate them).
if(NOT USE_CUDA AND NOT USE_ROCM)
  set(MBIS_MPI_STUB_SOURCES
      ../module_mbis_alm/mbis_alm_gpu_stub.cpp
      ../module_constraint/constraint_mbis_gpu_stub.cpp)
else()
  set(MBIS_MPI_STUB_SOURCES "")
endif()

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
    ../module_mbis_alm/mbis_alm.cpp
    ${MBIS_MPI_STUB_SOURCES}
    ../module_constraint/weight_grid.cpp
    ../module_constraint/constraint_observe.cpp
    ../module_constraint/constraint_dipole.cpp
    ../module_constraint/constraint_io.cpp
    ../../source_io/module_output/output.cpp
)

add_test(NAME MODULE_ESTATE_constraint_weight_grid_mpi_4np
      COMMAND mpirun -np 4 ./MODULE_ESTATE_constraint_weight_grid_mpi;
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

AddTest(
  TARGET MODULE_ESTATE_constraint_alloc_pw_mpi
  LIBS parameter ${math_libs} base device
  SOURCES alloc_pw_mpi_test.cpp
    ../module_constraint/alloc_pw.cpp
    ../module_constraint/alloc_linalg.cpp
)

add_test(NAME MODULE_ESTATE_constraint_alloc_pw_mpi_2np
      COMMAND mpirun -np 2 ./MODULE_ESTATE_constraint_alloc_pw_mpi;
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME MODULE_ESTATE_constraint_alloc_pw_mpi_4np
      COMMAND mpirun -np 4 ./MODULE_ESTATE_constraint_alloc_pw_mpi;
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
