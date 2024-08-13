remove_definitions(-D__DEEPKS)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)
remove_definitions(-D__EXX)

AddTest(
  TARGET pwdft_soc
  LIBS ${math_libs}
  SOURCES soc_test.cpp ../soc.cpp
  	../../../module_base/global_variable.cpp
  	../../../module_base/global_function.cpp
	../../../module_base/global_file.cpp
	../../../module_base/tool_quit.cpp
	../../../module_base/global_variable.cpp
	../../../module_base/global_file.cpp
	../../../module_base/memory.cpp
	../../../module_base/timer.cpp
	../../../module_base/parallel_global.cpp
	../../../module_base/parallel_comm.cpp
	../../../module_base/parallel_common.cpp
	../../../module_base/parallel_reduce.cpp
)

AddTest(
	TARGET radial_proj_test
	LIBS base device ${math_libs}
	SOURCES radial_proj_test.cpp ../radial_proj.cpp
)

AddTest(
	TARGET becp_pw_test
	LIBS base device ${math_libs} psi
	SOURCES ../becp_pw.cpp becp_pw_test.cpp
	../radial_proj.cpp 
	../structure_factor.cpp
	../structure_factor_k.cpp
	../../../module_basis/module_pw/pw_basis.cpp
	../../../module_basis/module_pw/pw_basis_k.cpp
	../../../module_basis/module_pw/pw_init.cpp
	../../../module_basis/module_pw/fft.cpp
	../../../module_basis/module_pw/pw_distributer.cpp
	../../../module_basis/module_pw/pw_distributeg.cpp
	../../../module_basis/module_pw/pw_transform.cpp
	../../../module_basis/module_nao/projgen.cpp
	../global.cpp
)

add_test(NAME becp_pw_test_parallel
	COMMAND mpirun -np 4 ./becp_pw_test
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)