add_subdirectory(operator_pw)

list(APPEND objects
    hamilt_pw.cpp
    operator_pw/ekinetic_pw.cpp
    operator_pw/veff_pw.cpp
    operator_pw/nonlocal_pw.cpp
    operator_pw/meta_pw.cpp
    operator_pw/velocity_pw.cpp
    operator_pw/operator_pw.cpp
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
