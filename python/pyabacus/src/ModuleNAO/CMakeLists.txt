# add nao shared library
list(APPEND _naos
    ${NAO_PATH}/atomic_radials.cpp
    ${NAO_PATH}/beta_radials.cpp
    ${NAO_PATH}/hydrogen_radials.cpp
    ${NAO_PATH}/numerical_radial.cpp
    ${NAO_PATH}/pswfc_radials.cpp
    ${NAO_PATH}/radial_collection.cpp
    ${NAO_PATH}/radial_set.cpp
    ${NAO_PATH}/real_gaunt_table.cpp
    ${NAO_PATH}/sphbes_radials.cpp
    ${NAO_PATH}/two_center_bundle.cpp
    ${NAO_PATH}/two_center_integrator.cpp
    ${NAO_PATH}/two_center_table.cpp
    # dependency
    ${ABACUS_SOURCE_DIR}/source_base/kernels/math_ylm_op.cpp
    ${ABACUS_SOURCE_DIR}/source_base/kernels/math_kernel_op.cpp
    ${ABACUS_SOURCE_DIR}/source_base/kernels/math_kernel_op_vec.cpp
    # ${ABACUS_SOURCE_DIR}/source_psi/kernels/psi_memory_op.cpp
    ${ABACUS_SOURCE_DIR}/source_base/module_device/memory_op.cpp
    ${ABACUS_SOURCE_DIR}/source_base/module_device/device.cpp
)
add_library(naopack SHARED
    ${_naos}
)

# Link against libabacus_core in DYNAMIC mode
if(TARGET abacus::abacus_core)
    target_link_libraries(naopack
        abacus::abacus_core
        ${math_libs}
    )
elseif(ABACUS_CORE_LIB)
    target_link_libraries(naopack
        ${ABACUS_CORE_LIB}
        ${math_libs}
    )
else()
    message(FATAL_ERROR "libabacus_core not found for ModuleNAO")
endif()

list(APPEND pymodule_nao
    ${PROJECT_SOURCE_DIR}/src/ModuleNAO/py_m_nao.cpp
)

pybind11_add_module(_nao_pack MODULE ${pymodule_nao})
target_link_libraries(_nao_pack PRIVATE pybind11::headers naopack)
target_compile_definitions(_nao_pack PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Set RPATH to find libabacus_core at runtime
if(ABACUS_INSTALL_DIR)
    set_target_properties(naopack PROPERTIES
        INSTALL_RPATH "$ORIGIN;${ABACUS_INSTALL_DIR}/lib;${ABACUS_INSTALL_DIR}/lib64"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    set_target_properties(_nao_pack PROPERTIES
        INSTALL_RPATH "$ORIGIN;${ABACUS_INSTALL_DIR}/lib;${ABACUS_INSTALL_DIR}/lib64"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
else()
    set_target_properties(naopack PROPERTIES INSTALL_RPATH "$ORIGIN")
    set_target_properties(_nao_pack PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS _nao_pack naopack DESTINATION ${TARGET_PACK}/ModuleNAO)