add_library(
    xc_
    OBJECT
    xc_functional.cpp
    xc_functional_vxc.cpp
    xc_functional_gradcorr.cpp
    xc_functional_wrapper_xc.cpp
    xc_functional_wrapper_gcxc.cpp
    xc_funct_exch_lda.cpp
    xc_funct_corr_lda.cpp
    xc_funct_exch_gga.cpp
    xc_funct_corr_gga.cpp
    xc_funct_hcth.cpp
    xc_functional_libxc.cpp
    xc_functional_libxc_tools.cpp
    xc_functional_libxc_vxc.cpp
    xc_functional_libxc_wrapper_xc.cpp
    xc_functional_libxc_wrapper_gcxc.cpp
    xc_functional_libxc_wrapper_tauxc.cpp
)

if(USE_CUDA)
    target_sources(xc_ PRIVATE
        xc_functional_gpu_launcher.cu
        xc_functional_gradcorr_gpu.cu
        xc_functional_vxc_gpu.cpp
    )
endif()

if(ENABLE_COVERAGE)
  add_coverage(xc_)
endif()

if(BUILD_TESTING)
if(ENABLE_MPI)
if(ENABLE_LIBXC)
  add_subdirectory(test)
  add_subdirectory(kernels/test)
endif()
endif()
endif()
