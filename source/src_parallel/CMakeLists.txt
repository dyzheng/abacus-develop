list(APPEND objects
    parallel_kpoints.cpp
)

add_library(
    parallel
    OBJECT
    ${objects}
)

if(ENABLE_COVERAGE)
  add_coverage(parallel)
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
