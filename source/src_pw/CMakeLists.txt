list(APPEND objects
    H_Ewald_pw.cpp
    energy.cpp
    global.cpp
    magnetism.cpp
    occupy.cpp
    structure_factor.cpp
    pw_complement.cpp
    soc.cpp
    symmetry_rho.cpp
    threshold_elec.cpp
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

