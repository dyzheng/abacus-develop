list(APPEND objects
    fft.cpp
    pw_basis.cpp
    pw_basis_k.cpp
    pw_basis_sup.cpp
    pw_distributeg.cpp
    pw_distributeg_method1.cpp
    pw_distributeg_method2.cpp
    pw_distributer.cpp
    pw_init.cpp
    pw_transform.cpp
    pw_transform_k.cpp
)

add_library(
    planewave
    OBJECT
    ${objects}
)

if(USE_CUFFTDX)
  add_library(planewave_fused_fft OBJECT kernels/cuda/fused_veff.cu)
  target_link_libraries(planewave_fused_fft PRIVATE mathdx::cufftdx)
  set_target_properties(planewave_fused_fft PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 70)
  target_sources(${ABACUS_BIN_NAME} PRIVATE $<TARGET_OBJECTS:planewave_fused_fft>)
endif()

if(ENABLE_COVERAGE)
  add_coverage(planewave)
endif()

if(BUILD_TESTING)
  if(ENABLE_MPI)
    add_subdirectory(test)
    add_subdirectory(test_serial)
    add_subdirectory(kernels/test)
  endif()
endif()
