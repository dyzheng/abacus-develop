if(ENABLE_LCAO)
  list(APPEND objects
      DM_gamma.cpp
      DM_k.cpp
      LCAO_gen_fixedH.cpp
      LCAO_hamilt.cpp
      LCAO_matrix.cpp
      LCAO_nnr.cpp
      build_st_pw.cpp
      center2_orb-orb11.cpp
      center2_orb-orb21.cpp
      center2_orb-orb22.cpp
      global_fp.cpp
      local_orbital_charge.cpp
      local_orbital_wfc.cpp
      record_adj.cpp
      dm_2d.cpp
      wavefunc_in_pw.cpp
  )

  add_library(
      lcao
      OBJECT
      ${objects}
  )

  if(ENABLE_COVERAGE)
    add_coverage(lcao)
  endif()
endif()
