list(APPEND objects
    deltap.cpp
    deltap_overlap.cpp
    deltap_berry.cpp
    deltap_gauge.cpp
    deltap_io.cpp
)

add_library(
    deltap
    OBJECT
    ${objects}
)

if(ENABLE_COVERAGE)
  add_coverage(deltap)
endif()

if(BUILD_TESTING)
  if(ENABLE_MPI)
    add_subdirectory(test)
  endif()
endif()
