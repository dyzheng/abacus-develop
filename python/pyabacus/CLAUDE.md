# CLAUDE.md - PyABACUS Development Guide

This file provides guidance for developing and extending the pyabacus Python bindings for ABACUS.

## Project Overview

PyABACUS provides Python bindings for ABACUS (Atomic-orbital Based Ab-initio Computation at UStc), enabling:
- Python-controlled SCF workflows with breakpoint support
- ASE (Atomic Simulation Environment) Calculator integration
- Direct access to Hamiltonian, density matrix, and other internal data

## Working Style

- NEVER spend more than 2 minutes exploring/planning before starting implementation. No plan files unless explicitly asked.
- When asked to "continue", resume coding immediately — do not re-explore the codebase.
- Only make changes the user explicitly asked for. Ask before expanding scope.
- All source and test files must be kept under 500 lines. Split proactively if approaching this limit.
- After any C++ pybind11 changes, always run `cmake --build build` before considering done.
- After any Python changes, run `pytest` immediately.
- Fix compilation errors in the same session — never leave broken builds. Fix the first error, rebuild, repeat.
- Do not batch many edits before validating — compile/test incrementally.

## Directory Structure

```
python/pyabacus/
├── src/
│   ├── pyabacus/                 # Python package
│   │   ├── __init__.py           # Main package init
│   │   ├── ase/                  # ASE Calculator integration
│   │   │   ├── __init__.py
│   │   │   └── calculator.py     # AbacusCalculator class
│   │   ├── esolver/              # ESolver Python interface
│   │   │   ├── __init__.py
│   │   │   ├── data_types.py     # Data containers (ForceData, StressData, etc.)
│   │   │   └── workflow.py       # LCAOWorkflow high-level interface
│   │   ├── cell/                 # Unit cell handling
│   │   ├── driver/               # ABACUS driver interface
│   │   ├── hsolver/              # Hamiltonian solver
│   │   └── io/                   # Input/output utilities
│   └── ModuleESolver/            # C++ pybind11 bindings (requires libabacus_core)
│       ├── py_esolver_lcao.hpp   # ESolver C++ header
│       ├── py_esolver_lcao_impl.cpp  # ESolver C++ implementation
│       ├── py_esolver_bindings.cpp   # pybind11 bindings
│       └── accessors/            # Data accessor implementations
│           └── py_accessors_impl.cpp
├── tests/                        # Test files
│   ├── test_ase_calculator.py    # ASE Calculator tests
│   └── ...
├── CMakeLists.txt                # CMake build configuration
└── pyproject.toml                # Python package configuration
```

## Build System

PyABACUS uses scikit-build-core with CMake for building C++ extensions.

### Building from Source

```bash
cd python/pyabacus

# Standard build (editable install) - basic modules only
pip install -e . --no-build-isolation

# Clean rebuild
pip uninstall pyabacus -y
pip install -e . --no-build-isolation
```

### Building with ESolver Module (Full Functionality)

The ESolver module requires `libabacus_core.so` from a compiled ABACUS installation.

```bash
# Step 1: Build and install ABACUS
cd /path/to/abacus-develop
cmake -B build -DENABLE_LCAO=ON
cmake --build build -j8
cmake --install build --prefix /path/to/install

# Step 2: Set environment variable and build pyabacus
export ABACUS_INSTALL_DIR=/path/to/install
cd python/pyabacus
pip install -e . --no-build-isolation
```

**Build behavior:**
- **With `libabacus_core`**: ESolver module is built with full ABACUS functionality
- **Without `libabacus_core`**: ESolver module is skipped (other modules still work)

### Build Dependencies

- Python >= 3.8
- pybind11 >= 2.10.0
- scikit-build-core >= 0.3.3
- CMake >= 3.15
- C++11 compiler (C++14 for some features)
- **For ESolver module**: libabacus_core.so (from ABACUS build)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ase_calculator.py -v

# Skip integration tests (require full ABACUS)
pytest tests/ -v -k 'not Integration'

# Run with coverage
pytest tests/ --cov=pyabacus --cov-report=html
```

### Test Markers

- `@pytest.mark.integration` - Tests requiring full ABACUS installation

## Key Components

### 1. ESolver Bindings (`ModuleESolver/`)

The C++ bindings expose ABACUS ESolver functionality to Python. **This module requires `libabacus_core.so`** - if not found during build, the module is skipped.

**Key Classes:**
- `PyESolverLCAO<TK, TR>` - Main ESolver wrapper (template)
- `PyChargeAccessor` - Access charge density
- `PyEnergyAccessor` - Access energy components
- `PyForceAccessor` - Access atomic forces
- `PyStressAccessor` - Access stress tensor
- `PyHamiltonianAccessor` - Access H(k), S(k) matrices
- `PyDensityMatrixAccessor` - Access density matrix

**File Structure:**
- `py_esolver_lcao.hpp` - Header with class declarations
- `py_esolver_lcao_impl.cpp` - Main ESolver implementation
- `py_esolver_bindings.cpp` - pybind11 module bindings
- `accessors/py_accessors_impl.cpp` - Accessor class implementations

**Adding New Bindings:**

1. Add accessor class in `py_esolver_lcao.hpp`:
```cpp
class PyNewAccessor {
public:
    PyNewAccessor() = default;
    void set_from_data(const double* ptr, int size);
    py::array_t<double> get_data() const;
    bool is_valid() const { return valid_; }
private:
    std::vector<double> data_;
    bool valid_ = false;
};
```

2. Implement in `py_accessors_impl.cpp`:
```cpp
void PyNewAccessor::set_from_data(const double* ptr, int size) {
    data_.resize(size);
    std::copy(ptr, ptr + size, data_.begin());
    valid_ = true;
}

py::array_t<double> PyNewAccessor::get_data() const {
    if (!is_valid()) throw std::runtime_error("Data not available");
    // Return numpy array...
}
```

3. Add pybind11 binding:
```cpp
void bind_new_accessor(py::module& m) {
    py::class_<py_esolver::PyNewAccessor>(m, "NewAccessor")
        .def(py::init<>())
        .def("get_data", &py_esolver::PyNewAccessor::get_data)
        .def("is_valid", &py_esolver::PyNewAccessor::is_valid);
}
```

4. Register in `PYBIND11_MODULE`:
```cpp
bind_new_accessor(m);
```

### 2. Data Types (`esolver/data_types.py`)

Python dataclasses for structured data with unit conversion methods.

**Key Classes:**
- `ForceData` - Forces with `to_eV_Ang()` conversion
- `StressData` - Stress tensor with `to_voigt()` and `to_eV_Ang3()`
- `EnergyData` - Energy components with `to_eV()` conversion
- `ChargeData` - Charge density data
- `HamiltonianData` - H(k), S(k), H(R), S(R) matrices
- `DensityMatrixData` - DM(k), DM(R) matrices
- `SCFResult` - SCF calculation results

**Unit Conversion Constants:**
```python
RY_TO_EV = 13.605693122994        # 1 Ry = 13.6057 eV
BOHR_TO_ANG = 0.529177249         # 1 Bohr = 0.529177 Å
RY_BOHR_TO_EV_ANG = 25.7112       # Force: Ry/Bohr → eV/Å
KBAR_TO_EV_ANG3 = 1/1602.1766208  # Stress: kbar → eV/Å³
```

### 3. LCAOWorkflow (`esolver/workflow.py`)

High-level Python interface for LCAO calculations with callback support.

**Key Methods:**
- `initialize()` - Initialize calculation
- `run_scf()` - Run SCF with callbacks
- `cal_force()` / `cal_stress()` - Calculate forces/stress
- `update_positions()` / `update_cell()` - Update geometry

**Callback Events:**
- `before_scf` - After `before_scf()` call
- `after_iter` - After each SCF iteration
- `before_after_scf` - Before `after_scf()` (main breakpoint)
- `after_scf` - After `after_scf()` call

### 4. ASE Calculator (`ase/calculator.py`)

ASE-compatible Calculator using pyabacus ESolver directly.

**Features:**
- Implements `energy`, `forces`, `stress` properties
- Automatic unit conversion (ABACUS → ASE units)
- Position/cell synchronization
- Compatible with ASE optimizers (BFGS, FIRE, etc.)

**Usage:**
```python
from ase import Atoms
from ase.optimize import BFGS
from pyabacus.ase import AbacusCalculator

calc = AbacusCalculator(input_dir='./Si_scf/', gamma_only=True)
atoms = Atoms('Si2', positions=[[0,0,0], [1.35,1.35,1.35]],
              cell=[5.43,5.43,5.43], pbc=True)
atoms.calc = calc

# Geometry optimization
opt = BFGS(atoms)
opt.run(fmax=0.01)

# Cell relaxation
from ase.constraints import UnitCellFilter
ucf = UnitCellFilter(atoms)
opt = BFGS(ucf)
opt.run(fmax=0.01)
```

## Development Workflow

### Adding a New Feature

1. **Write tests first** (TDD approach):
```python
# tests/test_new_feature.py
def test_new_feature():
    # Test expected behavior
    pass
```

2. **Add Python data types** if needed (`esolver/data_types.py`):
```python
@dataclass
class NewData:
    value: np.ndarray

    def to_converted_units(self) -> np.ndarray:
        return self.value * CONVERSION_FACTOR
```

3. **Add C++ bindings** if accessing ABACUS internals:
   - Header: `ModuleESolver/py_esolver_lcao.hpp`
   - Implementation: `ModuleESolver/py_esolver_lcao_impl.cpp` or `ModuleESolver/accessors/py_accessors_impl.cpp`

4. **Add workflow methods** (`esolver/workflow.py`):
```python
def new_method(self) -> NewData:
    accessor = self._esolver.get_new_data()
    return NewData(value=accessor.get_value())
```

5. **Update exports** (`esolver/__init__.py`, `__init__.py`)

6. **Rebuild and test**:
```bash
pip install -e . --no-build-isolation
pytest tests/test_new_feature.py -v
```

### Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Document public APIs with docstrings (NumPy style)
- C++ code follows ABACUS style (see main CLAUDE.md)

## Common Tasks

### Exposing New ABACUS Data to Python

1. Identify the data source in ABACUS C++ code
2. Create accessor class in `py_esolver_lcao.hpp`
3. Implement data extraction in `py_esolver_lcao_impl.cpp` or `accessors/py_accessors_impl.cpp`
4. Add pybind11 bindings in `py_esolver_bindings.cpp`
5. Create Python data container in `data_types.py`
6. Add workflow method in `workflow.py`
7. Export in `__init__.py`

### Adding Unit Conversion

1. Define conversion constant in `data_types.py`:
```python
NEW_UNIT_CONVERSION = 1.234  # old_unit → new_unit
```

2. Add conversion method to data class:
```python
def to_new_units(self) -> np.ndarray:
    return self.data * NEW_UNIT_CONVERSION
```

### Debugging C++ Bindings

```bash
# Build with debug symbols
CMAKE_BUILD_TYPE=Debug pip install -e . --no-build-isolation

# Check if bindings are available
python -c "from pyabacus.esolver._esolver_pack import ESolverLCAO_gamma; print(dir(ESolverLCAO_gamma()))"
```

## Important Notes

- **Template instantiation**: `PyESolverLCAO` is templated for gamma-only (`double`) and multi-k (`complex<double>`)
- **Memory management**: Use pybind11's return value policies correctly
- **Thread safety**: ABACUS uses MPI; be careful with parallel access
- **Unit consistency**: Always document units in docstrings
- **ESolver module dependency**: The ESolver module requires `libabacus_core.so`. Without it, the module is not built (no placeholder mode).

## Resources

- ABACUS Documentation: https://abacus.deepmodeling.com/
- pybind11 Documentation: https://pybind11.readthedocs.io/
- ASE Documentation: https://wiki.fysik.dtu.dk/ase/
- NumPy Style Docstrings: https://numpydoc.readthedocs.io/
