list(APPEND objects
    global.cpp
    symmetry_rho.cpp
)

add_library(
    pw
    OBJECT
    ${objects}
)

if(ENABLE_COVERAGE)
  add_coverage(pw)
endif()

