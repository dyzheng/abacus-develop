list(APPEND objects
    upf_valence_parser.cpp
    deltaqs_projector.cpp
)

add_library(
    deltaqs
    OBJECT
    ${objects}
)

if(ENABLE_COVERAGE)
  add_coverage(deltaqs)
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
