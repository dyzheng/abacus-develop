if(ENABLE_LCAO)

AddTest(
  TARGET MODULE_LCAO_deltaqs_upf_valence_parser_test
  LIBS ${math_libs} base device parameter cell
  SOURCES test_upf_valence_parser.cpp
    ../upf_valence_parser.cpp
)

endif()
