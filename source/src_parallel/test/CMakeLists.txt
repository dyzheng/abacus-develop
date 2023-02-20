AddTest(
  TARGET ParaKpoints
  LIBS MPI::MPI_CXX
  SOURCES parallel_kpoints_test.cpp ../../module_base/global_variable.cpp ../../module_base/parallel_global.cpp ../../module_base/parallel_common.cpp ../parallel_kpoints.cpp
)

install(FILES parallel_kpoints_test.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

find_program(BASH bash)

add_test(NAME parallel_kpoints_test
      COMMAND ${BASH} parallel_kpoints_test.sh
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
