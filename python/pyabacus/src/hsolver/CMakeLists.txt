# add diago shared library
list(APPEND _diago
    ${HSOLVER_PATH}/diago_dav_subspace.cpp
    ${HSOLVER_PATH}/diago_david.cpp
    ${HSOLVER_PATH}/diago_cg.cpp
    ${HSOLVER_PATH}/diag_const_nums.cpp
    ${HSOLVER_PATH}/diago_iter_assist.cpp
    ${HSOLVER_PATH}/diag_hs_para.cpp
    ${HSOLVER_PATH}/diago_pxxxgvx.cpp


    ${HSOLVER_PATH}/kernels/hegvd_op.cpp
    ${HSOLVER_PATH}/kernels/bpcg_kernel_op.cpp
    # dependency
    ${BASE_PATH}/kernels/math_kernel_op.cpp
    ${BASE_PATH}/kernels/math_kernel_op_vec.cpp
    ${BASE_PATH}/kernels/math_ylm_op.cpp
    ${BASE_PATH}/module_device/device.cpp
    ${BASE_PATH}/module_device/memory_op.cpp

    ${PSI_PATH}/psi.cpp
  )
add_library(diagopack SHARED
    ${_diago}
    )

# Link against libabacus_core in DYNAMIC mode
if(TARGET abacus::abacus_core)
    target_link_libraries(diagopack
        abacus::abacus_core
        ${math_libs}
    )
elseif(ABACUS_CORE_LIB)
    target_link_libraries(diagopack
        ${ABACUS_CORE_LIB}
        ${math_libs}
    )
else()
    message(FATAL_ERROR "libabacus_core not found for hsolver module")
endif()

list(APPEND pymodule_hsolver
    ${PROJECT_SOURCE_DIR}/src/hsolver/py_hsolver.cpp
    )

pybind11_add_module(_hsolver_pack MODULE ${pymodule_hsolver})
target_link_libraries(_hsolver_pack PRIVATE pybind11::headers diagopack)
target_compile_definitions(_hsolver_pack PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Set RPATH to find libabacus_core at runtime
if(ABACUS_INSTALL_DIR)
    set_target_properties(diagopack PROPERTIES
        INSTALL_RPATH "$ORIGIN;${ABACUS_INSTALL_DIR}/lib;${ABACUS_INSTALL_DIR}/lib64"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    set_target_properties(_hsolver_pack PROPERTIES
        INSTALL_RPATH "$ORIGIN;${ABACUS_INSTALL_DIR}/lib;${ABACUS_INSTALL_DIR}/lib64"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
else()
    set_target_properties(diagopack PROPERTIES INSTALL_RPATH "$ORIGIN")
    set_target_properties(_hsolver_pack PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS _hsolver_pack diagopack DESTINATION ${TARGET_PACK}/hsolver)