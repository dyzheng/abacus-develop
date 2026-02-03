if(ENABLE_LCAO)
    AddTest(
        TARGET orbital_mag_test
        LIBS parameter ${math_libs} psi base device container
        SOURCES orbital_mag_test.cpp
                ../orbital_mag.cpp
                ../../module_hcontainer/func_folding.cpp
                ../../module_hcontainer/base_matrix.cpp
                ../../module_hcontainer/hcontainer.cpp
                ../../module_hcontainer/atom_pair.cpp
                ../../module_rt/td_folding.cpp
                ../../../source_basis/module_ao/parallel_orbitals.cpp
                tmp_mocks.cpp
    )
endif()
