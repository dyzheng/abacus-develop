
AddTest(
  TARGET test_hcontainer
  LIBS ${math_libs} psi base device
  SOURCES test_hcontainer.cpp ../hcontainer.cpp ../atom_pair.cpp ../base_matrix.cpp
)
