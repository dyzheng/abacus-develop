list(APPEND objects
    spin_constrain.cpp
    init_sc.cpp
    cal_mw.cpp
    cal_mw_from_lambda.cpp
    cal_mw_helper.cpp
    cal_h_lambda.cpp
    sc_parse_json.cpp
    basic_funcs.cpp
    lambda_loop_helper.cpp
    lambda_loop.cpp
    template_helpers.cpp
    lambda_update_strategies.cpp
    lambda_strategy_integration.cpp
    deltaspin_lcao.cpp
)

add_library(
    deltaspin
    OBJECT
    ${objects}
)

if(ENABLE_COVERAGE)
  add_coverage(deltaspin)
endif()

if(BUILD_TESTING)
  if(ENABLE_MPI)
    add_subdirectory(test)
  endif()
endif()
