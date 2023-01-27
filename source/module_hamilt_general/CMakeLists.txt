list(APPEND objects
    operator.cpp
)

add_library(
    hamilt_general
    OBJECT
    ${objects}
)

if(ENABLE_COVERAGE)
  add_coverage(hamilt_general)
endif()
