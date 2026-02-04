cmake_minimum_required(VERSION 3.15...3.26)

# project settings
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# find python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# set source path
set(ABACUS_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../source")
set(BASE_PATH "${ABACUS_SOURCE_DIR}/source_base")
set(NAO_PATH "${ABACUS_SOURCE_DIR}/source_basis/module_nao")
set(HSOLVER_PATH "${ABACUS_SOURCE_DIR}/source_hsolver")
set(PSI_PATH "${ABACUS_SOURCE_DIR}/source_psi")
set(ENABLE_LCAO ON)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../../cmake")

# =============================================================================
# Try to find pre-built libabacus_core for fast incremental builds
# =============================================================================
if(DEFINED ENV{ABACUS_INSTALL_DIR})
    set(ABACUS_INSTALL_DIR "$ENV{ABACUS_INSTALL_DIR}")
endif()

# Try find_package first (preferred method)
if(ABACUS_INSTALL_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${ABACUS_INSTALL_DIR}")
endif()

find_package(abacus_core QUIET CONFIG)

if(abacus_core_FOUND)
    set(PYABACUS_BUILD_MODE "DYNAMIC")
    message(STATUS "")
    message(STATUS "========================================================================")
    message(STATUS "Found pre-built libabacus_core - using DYNAMIC build mode")
    message(STATUS "========================================================================")
    message(STATUS "Library: ${ABACUS_CORE_LIBRARIES}")
    message(STATUS "Include: ${ABACUS_CORE_INCLUDE_DIRS}")
    message(STATUS "")
    message(STATUS "This enables fast incremental builds!")
    message(STATUS "Only binding code will be recompiled on subsequent installs.")
    message(STATUS "========================================================================")
    message(STATUS "")
else()
    # Fallback: try to find library directly
    find_library(ABACUS_CORE_LIB
        NAMES abacus_core
        HINTS
            ${ABACUS_INSTALL_DIR}/lib
            ${ABACUS_INSTALL_DIR}/lib64
            ${CMAKE_INSTALL_PREFIX}/lib
            ${CMAKE_INSTALL_PREFIX}/lib64
        PATH_SUFFIXES lib lib64
    )

    if(ABACUS_CORE_LIB)
        set(PYABACUS_BUILD_MODE "DYNAMIC")
        # Set include dirs manually
        if(ABACUS_INSTALL_DIR)
            set(ABACUS_CORE_INCLUDE_DIRS "${ABACUS_INSTALL_DIR}/include/abacus")
        endif()
        message(STATUS "")
        message(STATUS "========================================================================")
        message(STATUS "Found libabacus_core directly - using DYNAMIC build mode")
        message(STATUS "========================================================================")
        message(STATUS "Library: ${ABACUS_CORE_LIB}")
        message(STATUS "Include: ${ABACUS_CORE_INCLUDE_DIRS}")
        message(STATUS "========================================================================")
        message(STATUS "")
    else()
        message(FATAL_ERROR "
========================================================================
libabacus_core not found!

PyABACUS requires pre-built libabacus_core. Please build ABACUS first:

  cd /path/to/abacus-develop
  cmake -B build -DENABLE_LCAO=ON
  cmake --build build -j\$(nproc)
  cmake --install build --prefix \$HOME/.local/abacus

Then set environment and rebuild pyabacus:
  export ABACUS_INSTALL_DIR=\$HOME/.local/abacus
  pip install -e . --no-build-isolation
========================================================================
")
    endif()
endif()

# Export build mode for submodules
set(PYABACUS_BUILD_MODE ${PYABACUS_BUILD_MODE} CACHE STRING "PyABACUS build mode (DYNAMIC only)")

# =============================================================================
# Setup math libraries (needed for both build modes)
# =============================================================================
if(DEFINED ENV{MKLROOT} AND NOT DEFINED MKLROOT)
    set(MKLROOT "$ENV{MKLROOT}")
endif()
if(MKLROOT)
  set(MKL_INTERFACE lp64)
  set(ENABLE_MPI ON)
  if (ENABLE_MPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_CXX_INCLUDE_PATH})
  endif()

  set(USE_OPENMP ON)
  if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_link_options(${OpenMP_CXX_LIBRARIES})
  endif()
  find_package(MKL REQUIRED)
  add_definitions(-D__MKL)
  include_directories(${MKL_INCLUDE} ${MKL_INCLUDE}/fftw)

  if(NOT ENABLE_MLALGO)
    list(APPEND math_libs IntelMKL::MKL)
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
    list(APPEND math_libs -lifcore)
  endif()
else()
    find_package(FFTW3 REQUIRED)
    add_compile_definitions(__FFTW3)
    find_package(LAPACK REQUIRED)
    include_directories(${FFTW3_INCLUDE_DIRS})
    message(STATUS "FFTW3_INCLUDE_DIRS: ${FFTW3_INCLUDE_DIRS}")
    list(APPEND math_libs FFTW3::FFTW3 LAPACK::LAPACK)

  if(ENABLE_LCAO)
    find_package(ScaLAPACK REQUIRED)
    list(APPEND math_libs ScaLAPACK::ScaLAPACK)
  endif()
endif()

# =============================================================================
# Include directories setup
# =============================================================================
if(PYABACUS_BUILD_MODE STREQUAL "DYNAMIC" AND ABACUS_CORE_INCLUDE_DIRS)
    # Use installed headers
    include_directories(${ABACUS_CORE_INCLUDE_DIRS})
    include_directories(${ABACUS_CORE_INCLUDE_DIRS}/source_base/module_container)
else()
    # Use source headers
    include_directories(
        ${BASE_PATH}
        ${ABACUS_SOURCE_DIR}
        ${ABACUS_SOURCE_DIR}/source_base/module_container
    )
endif()

# Add LCAO compile definition globally for all ABACUS modules
if(ENABLE_LCAO)
    add_compile_definitions(__LCAO)
endif()

# =============================================================================
# Add basic libraries
# =============================================================================
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# set RPATH
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# set package name to pyabacus
set(TARGET_PACK pyabacus)
set(CMAKE_INSTALL_RPATH "${PYTHON_SITE_PACKAGES}/${TARGET_PACK}")

# Option to build the driver module (requires full ABACUS library)
option(BUILD_PYABACUS_DRIVER "Build PyABACUS driver module" OFF)

# add subdirectories for submodules
add_subdirectory(${PROJECT_SOURCE_DIR}/src/hsolver)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/ModuleBase)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/ModuleNAO)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/ModuleESolver)

# Conditionally add driver module
if(BUILD_PYABACUS_DRIVER)
    message(STATUS "Building PyABACUS driver module")
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/ModuleDriver)
endif()
