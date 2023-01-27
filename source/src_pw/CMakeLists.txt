list(APPEND objects
    global.cpp
    structure_factor.cpp
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

