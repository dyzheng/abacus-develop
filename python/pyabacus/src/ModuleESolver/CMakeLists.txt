# CMakeLists.txt for ModuleESolver Python bindings
# This module provides Python bindings for ESolver_KS_LCAO
#
# Build mode: DYNAMIC only - links against pre-built libabacus_core.so

# Set paths for ESolver related sources
set(ESOLVER_PATH "${ABACUS_SOURCE_DIR}/source_esolver")
set(ESTATE_PATH "${ABACUS_SOURCE_DIR}/source_estate")
set(LCAO_PATH "${ABACUS_SOURCE_DIR}/source_lcao")
set(HAMILT_PATH "${ABACUS_SOURCE_DIR}/source_hamilt")
set(CELL_PATH "${ABACUS_SOURCE_DIR}/source_cell")
set(PSI_PATH "${ABACUS_SOURCE_DIR}/source_psi")
set(BASIS_PATH "${ABACUS_SOURCE_DIR}/source_basis")
set(IO_PATH "${ABACUS_SOURCE_DIR}/source_io")

# Accessor implementation sources
set(ACCESSOR_SOURCES
    ${PROJECT_SOURCE_DIR}/src/ModuleESolver/accessors/py_accessors_impl.cpp
)

# ESolver implementation and binding sources
set(ESOLVER_SOURCES
    ${PROJECT_SOURCE_DIR}/src/ModuleESolver/py_esolver_lcao_impl.cpp
    ${PROJECT_SOURCE_DIR}/src/ModuleESolver/py_esolver_bindings.cpp
)

# Python module source files
list(APPEND pymodule_esolver
    ${ACCESSOR_SOURCES}
    ${ESOLVER_SOURCES}
)

# Common include directories (for source headers)
set(ESOLVER_INCLUDE_DIRS_SOURCE
    ${ABACUS_SOURCE_DIR}
    ${ESOLVER_PATH}
    ${ESTATE_PATH}
    ${ESTATE_PATH}/module_charge
    ${ESTATE_PATH}/module_dm
    ${ESTATE_PATH}/potentials
    ${LCAO_PATH}
    ${LCAO_PATH}/module_hcontainer
    ${HAMILT_PATH}
    ${CELL_PATH}
    ${PSI_PATH}
    ${BASIS_PATH}
    ${BASIS_PATH}/module_ao
    ${BASIS_PATH}/module_nao
    ${IO_PATH}
    ${IO_PATH}/module_parameter
    ${PROJECT_SOURCE_DIR}/src/ModuleESolver
    ${PROJECT_SOURCE_DIR}/src/ModuleESolver/accessors
)

# =============================================================================
# DYNAMIC build mode - link against pre-built libabacus_core.so
# =============================================================================
if(PYABACUS_BUILD_MODE STREQUAL "DYNAMIC")
    message(STATUS "PyABACUS ESolver: Using DYNAMIC build mode (fast incremental builds)")

    # Create pybind11 module
    pybind11_add_module(_esolver_pack MODULE ${pymodule_esolver})

    # Include directories - prefer installed headers if available
    if(ABACUS_CORE_INCLUDE_DIRS)
        target_include_directories(_esolver_pack PRIVATE
            ${ABACUS_CORE_INCLUDE_DIRS}
            ${ABACUS_CORE_INCLUDE_DIRS}/source_esolver
            ${ABACUS_CORE_INCLUDE_DIRS}/source_estate
            ${ABACUS_CORE_INCLUDE_DIRS}/source_estate/module_charge
            ${ABACUS_CORE_INCLUDE_DIRS}/source_estate/module_dm
            ${ABACUS_CORE_INCLUDE_DIRS}/source_estate/potentials
            ${ABACUS_CORE_INCLUDE_DIRS}/source_lcao
            ${ABACUS_CORE_INCLUDE_DIRS}/source_lcao/module_hcontainer
            ${ABACUS_CORE_INCLUDE_DIRS}/source_hamilt
            ${ABACUS_CORE_INCLUDE_DIRS}/source_cell
            ${ABACUS_CORE_INCLUDE_DIRS}/source_psi
            ${ABACUS_CORE_INCLUDE_DIRS}/source_basis
            ${ABACUS_CORE_INCLUDE_DIRS}/source_basis/module_ao
            ${ABACUS_CORE_INCLUDE_DIRS}/source_basis/module_nao
            ${ABACUS_CORE_INCLUDE_DIRS}/source_io
            ${ABACUS_CORE_INCLUDE_DIRS}/source_io/module_parameter
            ${PROJECT_SOURCE_DIR}/src/ModuleESolver
            ${PROJECT_SOURCE_DIR}/src/ModuleESolver/accessors
        )
    else()
        target_include_directories(_esolver_pack PRIVATE ${ESOLVER_INCLUDE_DIRS_SOURCE})
    endif()

    # Link against ABACUS core library
    if(TARGET abacus::abacus_core)
        # Found via find_package
        target_link_libraries(_esolver_pack PRIVATE
            pybind11::headers
            abacus::abacus_core
        )
    elseif(ABACUS_CORE_LIB)
        # Found directly
        target_link_libraries(_esolver_pack PRIVATE
            pybind11::headers
            ${ABACUS_CORE_LIB}
        )
    endif()

    # Define standard ABACUS macros
    target_compile_definitions(_esolver_pack PRIVATE
        __LCAO
        VERSION_INFO=${PROJECT_VERSION}
    )

    # Add MPI if available (required by ABACUS)
    find_package(MPI QUIET)
    if(MPI_FOUND)
        target_link_libraries(_esolver_pack PRIVATE MPI::MPI_CXX)
        target_compile_definitions(_esolver_pack PRIVATE __MPI)
    endif()

    # Add OpenMP if available
    find_package(OpenMP QUIET)
    if(OpenMP_FOUND)
        target_link_libraries(_esolver_pack PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Set RPATH to find libabacus_core at runtime
    if(ABACUS_INSTALL_DIR)
        set_target_properties(_esolver_pack PROPERTIES
            INSTALL_RPATH "$ORIGIN;${ABACUS_INSTALL_DIR}/lib;${ABACUS_INSTALL_DIR}/lib64"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    else()
        set_target_properties(_esolver_pack PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()

    message(STATUS "PyABACUS ESolver module will be built with dynamic linking")

    # Install targets
    install(TARGETS _esolver_pack DESTINATION ${TARGET_PACK}/esolver)
endif()
