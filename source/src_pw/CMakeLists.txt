list(APPEND objects
    global.cpp
    symmetry_rho.cpp
    threshold_elec.cpp
)

add_library(
    pw
    OBJECT
    ${objects}
)

if(ENABLE_COVERAGE)
  add_coverage(pw)
endif()

