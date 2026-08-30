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
