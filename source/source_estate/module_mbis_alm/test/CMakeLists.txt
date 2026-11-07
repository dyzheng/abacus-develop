remove_definitions(-D__MPI)
remove_definitions(-D__EXX)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)
remove_definitions(-D__MLALGO)

AddTest(
  TARGET MODULE_ESTATE_mbis_alm
  LIBS ${math_libs}
  SOURCES mbis_alm_test.cpp
    ../mbis_alm.cpp
)

AddTest(
  TARGET MODULE_ESTATE_mbis_alm_adapter
  LIBS ${math_libs}
  SOURCES mbis_alm_adapter_test.cpp
    ../mbis_alm.cpp
    ../mbis_alm_adapter.cpp
)
