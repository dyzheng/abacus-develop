list(APPEND objects
    operator.cpp
    module_ewald/H_Ewald_pw.cpp
)

add_library(
    hamilt_general
    OBJECT
    ${objects}
)

if(ENABLE_COVERAGE)
  add_coverage(hamilt_general)
endif()
