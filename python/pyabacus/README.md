# pyabacus: a Python interface for the ABACUS package

`pyabacus` is a Python interface for the ABACUS package, which provides a high-level Python API for interacting with the `ABACUS` library.

This project is built using [pybind11](http://github.com/pybind/pybind11) and [scikit-build-core](https://scikit-build-core.readthedocs.io/), so you can easily build the project and use it in your Python environment.

Now, `pyabacus` provides the following modules:
- `io`: a module for input/output in pyabacus.
- `Cell`: a module for the cell structure to bridge `ModuleNAO` in python module with users' input.
- `ModuleBase`: a module for basic math functions.
- `ModuleNAO`: a module for numerical atomic orbitals (NAO).
- `hsolver`: a module for solving the Hamiltonian.
- `esolver`: a module for ESolver bindings (requires `libabacus_core`).
- `ase`: ASE Calculator integration for geometry optimization and MD.

<!-- toc -->

- [Installation](#installation)
- [Building with ESolver Module](#building-with-esolver-module)
- [CI Examples](#ci-examples)
- [License](#license)
- [Test call](#test-call)

<!-- tocstop -->

## Installation

- Create and activate a new conda env, e.g. `conda create -n myenv python=3.8 & conda activate myenv`.
- Clone ABACUS main repository and `cd abacus-develop/python/pyabacus`.
- Build pyabacus by `pip install -v .` or install test dependencies & build  pyabacus by `pip install .[test]`. (Use `pip install -v .[test] -i https://pypi.tuna.tsinghua.edu.cn/simple` to accelerate installation process.)

## Building with ESolver Module

The ESolver module provides Python bindings for ABACUS's electronic structure solver, enabling:
- Python-controlled SCF workflows with breakpoint support
- ASE (Atomic Simulation Environment) Calculator integration
- Direct access to Hamiltonian, density matrix, and other internal data

**The ESolver module requires `libabacus_core.so`** from a compiled ABACUS installation.

### Step 1: Build and install ABACUS

```bash
cd /path/to/abacus-develop
cmake -B build -DENABLE_LCAO=ON
cmake --build build -j8
cmake --install build --prefix /path/to/install
```

### Step 2: Set environment variable and build pyabacus

```bash
export ABACUS_INSTALL_DIR=/path/to/install
cd python/pyabacus
pip install -e . --no-build-isolation
```

### Build behavior

- **With `libabacus_core`**: ESolver module is built with full ABACUS functionality
- **Without `libabacus_core`**: ESolver module is skipped (other modules still work)

### Verify ESolver module

```python
# Check if ESolver module is available
try:
    from pyabacus.esolver._esolver_pack import ESolverLCAO_gamma
    print("ESolver module is available")
except ImportError:
    print("ESolver module not built - libabacus_core not found during build")
```

## CI Examples

There are examples for CI in `.github/workflows`. A simple way to produces
binary "wheels" for all platforms is illustrated in the "wheels.yml" file,
using .

Use `pytest -v` to run all the unit tests for pyabacus in the local machine.

```shell
$ cd tests/
$ pytest -v
```

Run `python vis_nao.py` to visualize the numerical orbital.

```shell
$ cd examples/
$ python vis_nao.py
```

Run `python ex_s_rotate.py` in `examples` to check the S matrix.

```shell
$ cd examples/
$ python ex_s_rotate.py
norm(S_e3 - S_numer) =  3.341208104032616e-15
```

Run `python diago_matrix.py` in `examples` to check the diagonalization of a matrix.

```shell
$ cd examples/
$ python diago_matrix.py

====== Calculating eigenvalues using davidson method... ======
eigenvalues calculated by pyabacus-davidson is:
 [-0.38440611  0.24221155  0.31593272  0.53144616  0.85155108  1.06950154
  1.11142053  1.12462153]
eigenvalues calculated by scipy is:
 [-0.38440611  0.24221155  0.31593272  0.53144616  0.85155108  1.06950154
  1.11142051  1.12462151]
eigenvalues difference:
 [4.47258897e-12 5.67104697e-12 8.48299209e-12 1.08900666e-11
 1.87927451e-12 3.15688586e-10 2.11438165e-08 2.68884972e-08]

====== Calculating eigenvalues using dav_subspace method... ======
enter diag... is_subspace = 0, ntry = 0
eigenvalues calculated by pyabacus-dav_subspace is:
 [-0.38440611  0.24221155  0.31593272  0.53144616  0.85155108  1.06950154
  1.11142051  1.12462153]
eigenvalues calculated by scipy is:
 [-0.38440611  0.24221155  0.31593272  0.53144616  0.85155108  1.06950154
  1.11142051  1.12462151]
eigenvalues difference:
 [ 4.64694949e-12  2.14706031e-12  1.09236509e-11  4.66293670e-13
 -8.94295749e-12  4.71351846e-11  5.39378986e-10  1.97244101e-08]
```

## License

pybind11 is provided under a BSD-style license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.

## Test call

```python
import pyabacus as m
s = m.ModuleBase.Sphbes()
s.sphbesj(1, 0.0)
0.0
```

[`cibuildwheel`]: https://cibuildwheel.readthedocs.io
