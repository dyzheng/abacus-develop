add_subdirectory(ks_pw)

list(APPEND objects
    hamilt_pw.cpp
    ks_pw/ekinetic_pw.cpp
    ks_pw/veff_pw.cpp
    ks_pw/nonlocal_pw.cpp
    ks_pw/meta_pw.cpp
    ks_pw/velocity_pw.cpp
    ks_pw/operator_pw.cpp
)

add_library(
    hamilt_pwdft
    OBJECT
    ${objects}
)

if(ENABLE_COVERAGE)
  add_coverage(hamilt_pwdft)
endif()

if (BUILD_TESTING)
    add_subdirectory(kernels/test)
endif()
