list(APPEND objects
    H_Ewald_pw.cpp
    VL_in_pw.cpp
    VNL_in_pw.cpp
    VNL_grad_pw.cpp
    energy.cpp
    global.cpp
    magnetism.cpp
    occupy.cpp
    structure_factor.cpp
    pw_complement.cpp
    soc.cpp
    of_stress_pw.cpp
    symmetry_rho.cpp
    threshold_elec.cpp
    wavefunc.cpp
    wf_atomic.cpp
    wf_igk.cpp
    xc_3.cpp
)

add_library(
    pw
    OBJECT
    ${objects}
)

if(ENABLE_COVERAGE)
  add_coverage(pw)
endif()

if(BUILD_TESTING)
  add_subdirectory(kernels/test)
endif()

