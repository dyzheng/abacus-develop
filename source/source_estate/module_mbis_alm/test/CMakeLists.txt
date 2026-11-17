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
    ../mbis_alm_gpu_stub.cpp
)

AddTest(
  TARGET MODULE_ESTATE_mbis_alm_adapter
  LIBS ${math_libs}
  SOURCES mbis_alm_adapter_test.cpp
    ../mbis_alm.cpp
    ../mbis_alm_adapter.cpp
    ../mbis_alm_gpu_stub.cpp
)

# Device-backend test: only built when CUDA is enabled.  The directory above
# removes -D__CUDA for the CPU tests, so re-add it for this target and restore
# the CPU-only state afterwards.
if(USE_CUDA)
  add_definitions(-D__CUDA)
  AddTest(
    TARGET MODULE_ESTATE_mbis_alm_gpu
    LIBS ${math_libs} cudart
    SOURCES mbis_alm_gpu_test.cpp
      ../mbis_alm.cpp
      ../cuda/mbis_alm_gpu.cu
  )
  remove_definitions(-D__CUDA)
endif()
