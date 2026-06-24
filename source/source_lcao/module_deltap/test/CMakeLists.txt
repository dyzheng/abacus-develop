if(ENABLE_LCAO)

AddTest(
  TARGET MODULE_LCAO_deltap_math_test
  LIBS ${math_libs} base device parameter
  SOURCES deltap_math_test.cpp
)

AddTest(
  TARGET MODULE_LCAO_deltap_gauge_test
  LIBS ${math_libs} base device parameter
  SOURCES deltap_gauge_test.cpp
)

endif()
