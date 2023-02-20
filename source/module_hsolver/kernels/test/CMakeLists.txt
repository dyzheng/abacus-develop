remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)

if(USE_CUDA OR USE_ROCM)
    AddTest(
        TARGET Hsolver_Kernels_UTs
        LIBS ${math_libs} base device
        SOURCES math_kernel_test.cpp math_dngvd_test.cpp
        ../../../module_base/parallel_global.cpp ../../../module_base/parallel_common.cpp
        ../../../module_base/parallel_reduce.cpp
    )
elseif()
    AddTest(
        TARGET Hsolver_Kernels_UTs
        LIBS ${math_libs} base device
        SOURCES math_kernel_test.cpp ../../../module_base/parallel_global.cpp
        ../../../module_base/parallel_common.cpp  ../../../module_base/parallel_reduce.cpp
    )
endif()