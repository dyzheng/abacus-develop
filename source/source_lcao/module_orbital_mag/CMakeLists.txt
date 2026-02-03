if(ENABLE_LCAO)
    add_library(orbital_mag OBJECT
        orbital_mag.cpp
    )
    target_link_libraries(orbital_mag
        parameter
        ${math_libs}
        psi
        base
        device
        container
    )

    if(BUILD_TESTING)
        add_subdirectory(test)
    endif()
endif()
