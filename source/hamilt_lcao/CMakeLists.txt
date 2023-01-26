add_subdirectory(operator_lcao)

if(ENABLE_LCAO)
    list(APPEND objects
        hamilt_lcao.cpp
        operator_lcao/operator_lcao.cpp
        operator_lcao/ekinetic_lcao.cpp
        operator_lcao/nonlocal_lcao.cpp
        operator_lcao/overlap_lcao.cpp
        operator_lcao/veff_lcao.cpp
        operator_lcao/meta_lcao.cpp
        operator_lcao/op_dftu_lcao.cpp
        operator_lcao/deepks_lcao.cpp
        operator_lcao/op_exx_lcao.cpp
    )

    add_library(
        hamilt_lcao
        OBJECT
        ${objects}
    )

    if(ENABLE_COVERAGE)
    add_coverage(hamilt_lcao)
    endif()

endif()