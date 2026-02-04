list(APPEND pymodule_base
    ${PROJECT_SOURCE_DIR}/src/ModuleBase/py_base_math.cpp
    ${BASE_PATH}/kernels/math_ylm_op.cpp
    ${BASE_PATH}/kernels/math_kernel_op.cpp
    ${BASE_PATH}/kernels/math_kernel_op_vec.cpp
    ${BASE_PATH}/module_device/memory_op.cpp
    ${BASE_PATH}/module_device/device.cpp
    )

pybind11_add_module(_base_pack MODULE ${pymodule_base})

# Link against libabacus_core in DYNAMIC mode
if(TARGET abacus::abacus_core)
    target_link_libraries(_base_pack PRIVATE
        pybind11::headers
        abacus::abacus_core
        ${math_libs}
    )
elseif(ABACUS_CORE_LIB)
    target_link_libraries(_base_pack PRIVATE
        pybind11::headers
        ${ABACUS_CORE_LIB}
        ${math_libs}
    )
else()
    message(FATAL_ERROR "libabacus_core not found for ModuleBase")
endif()

target_compile_definitions(_base_pack PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Set RPATH to find libabacus_core at runtime
if(ABACUS_INSTALL_DIR)
    set_target_properties(_base_pack PROPERTIES
        INSTALL_RPATH "$ORIGIN;${ABACUS_INSTALL_DIR}/lib;${ABACUS_INSTALL_DIR}/lib64"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
else()
    set_target_properties(_base_pack PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS _base_pack DESTINATION ${TARGET_PACK}/ModuleBase)