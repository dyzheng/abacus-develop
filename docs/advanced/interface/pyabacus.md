# PyABACUS

PyABACUS provides Python bindings for ABACUS, enabling Python-controlled DFT calculations with support for:
- Running ABACUS calculations from Python
- Python-controlled SCF workflows with breakpoint support
- ASE (Atomic Simulation Environment) Calculator integration
- Direct access to Hamiltonian, density matrix, and other internal data

## Installation

### From Source (Basic)

For basic functionality without ESolver module:

```bash
cd python/pyabacus
pip install -e .
```

### With ESolver Module (Recommended)

The ESolver module provides direct Python bindings to ABACUS internals, enabling memory-persistent calculations and access to internal data structures. This requires building ABACUS first.

**Step 1: Build and install ABACUS**

```bash
cd /path/to/abacus-develop
cmake -B build -DENABLE_LCAO=ON
cmake --build build -j8
cmake --install build --prefix /path/to/install
```

**Step 2: Set environment variable and build pyabacus**

```bash
export ABACUS_INSTALL_DIR=/path/to/install
cd python/pyabacus
pip install -e . --no-build-isolation
```

**Build behavior:**
- **With `libabacus_core`**: ESolver module is built with full ABACUS functionality
- **Without `libabacus_core`**: ESolver module is skipped (other modules still work)

**Verify ESolver module is available:**

```python
try:
    from pyabacus.esolver import LCAOWorkflow
    print("ESolver module available")
except ImportError:
    print("ESolver module not available")
```

### With C++ Driver Support

For the `abacus()` function that runs ABACUS as a subprocess:

```bash
# Ensure ABACUS is installed and in PATH
which abacus  # Should return path to abacus executable

# Install pyabacus
cd python/pyabacus
pip install -e .
```

## Quick Start

### Using the Driver Function

The simplest way to run ABACUS from Python:

```python
from pyabacus import abacus

# Run calculation
result = abacus(
    input_dir="./Si_scf/",
    calculate_force=True,
    calculate_stress=True
)

# Access results
print(f"Total energy: {result.etot} eV")
print(f"Forces shape: {result.forces.shape}")
print(f"Converged: {result.converged}")
```

### Using LCAOWorkflow (ESolver Mode)

For more control and efficiency in sequential calculations:

```python
from pyabacus.esolver import LCAOWorkflow

# Create workflow
workflow = LCAOWorkflow("./Si_scf/", gamma_only=True)
workflow.initialize()

# Run SCF
result = workflow.run_scf(max_iter=100)
print(f"Converged: {result.converged}")
print(f"Energy: {result.energy.etot} Ry")

# Calculate forces and stress
workflow.cal_force()
workflow.cal_stress()
print(f"Forces: {workflow.force.to_eV_Ang()}")
print(f"Stress: {workflow.stress.to_voigt()}")

# Cleanup when done
workflow.cleanup()
```

## API Reference

### `abacus()` Function

```python
def abacus(
    input_dir: str = ".",
    calculate_force: bool = False,
    calculate_stress: bool = False,
    nprocs: int = 1,
    nthreads: int = 1,
    abacus_path: Optional[str] = None,
) -> CalculationResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | str | "." | Directory containing INPUT, STRU, KPT files |
| `calculate_force` | bool | False | Calculate atomic forces |
| `calculate_stress` | bool | False | Calculate stress tensor |
| `nprocs` | int | 1 | Number of MPI processes |
| `nthreads` | int | 1 | Number of OpenMP threads |
| `abacus_path` | str | None | Path to ABACUS executable (auto-detected if None) |

**Returns:** `CalculationResult` object

### `CalculationResult` Class

| Attribute | Type | Description |
|-----------|------|-------------|
| `etot` | float | Total energy in eV |
| `forces` | np.ndarray | Forces (nat, 3) in eV/Å (if calculated) |
| `stress` | np.ndarray | Stress tensor (3, 3) in kbar (if calculated) |
| `converged` | bool | SCF convergence status |
| `niter` | int | Number of SCF iterations |
| `output_dir` | str | Path to output directory |

## Output File Tracking

The `abacus()` function automatically tracks output files:

```python
result = abacus("./Si_scf/")

# Access tracked output files
print(result.output_files)  # Dict of output file paths
```

## Convenience Functions

```python
from pyabacus import get_version, find_abacus

# Get pyabacus version
print(get_version())

# Find ABACUS executable
abacus_path = find_abacus()
print(f"ABACUS found at: {abacus_path}")
```

## Examples

### Single-Point Calculation

```python
from pyabacus import abacus

result = abacus("./Si_scf/")
print(f"Energy: {result.etot:.6f} eV")
```

### Force Calculation

```python
from pyabacus import abacus

result = abacus("./Si_scf/", calculate_force=True)
for i, force in enumerate(result.forces):
    print(f"Atom {i}: {force}")
```

### Parallel Calculation

```python
from pyabacus import abacus

result = abacus(
    "./Si_scf/",
    nprocs=4,
    nthreads=2
)
```

## Advanced Features

### ESolver Module

The ESolver module provides direct Python bindings to ABACUS internals, enabling:
- Memory-persistent calculations for efficient sequential runs
- Direct access to Hamiltonian, density matrix, and other internal data
- Callback system for SCF breakpoints and custom workflows

**Requirements:** `libabacus_core.so` from ABACUS installation

### LCAOWorkflow Class

High-level Python interface for LCAO calculations with callback support.
Inherits from `_BaseWorkflow`, which provides shared methods for both LCAO and PW.

**Constructor:**

```python
LCAOWorkflow(input_dir: str, gamma_only: bool = True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | str | - | Directory containing INPUT, STRU, KPT files |
| `gamma_only` | bool | True | Use gamma-only calculation (True) or multi-k (False) |

**Core Methods:**

| Method | Description |
|--------|-------------|
| `initialize()` | Initialize the calculation (must be called first) |
| `run_scf(max_iter=100, istep=0, callback=None)` | Run SCF calculation, returns `SCFResult` |
| `cleanup()` | Release resources and reset state |

**Manual SCF Control:**

| Method | Description |
|--------|-------------|
| `before_scf(istep=0)` | Prepare for SCF calculation |
| `run_scf_step(iter_num)` | Run a single SCF iteration |
| `after_scf(istep=0)` | Finalize SCF calculation |

**Force and Stress:**

| Method | Description |
|--------|-------------|
| `cal_force()` | Calculate forces (call after SCF convergence) |
| `cal_stress()` | Calculate stress tensor (call after SCF convergence) |

**Geometry Update:**

| Method | Description |
|--------|-------------|
| `update_positions(positions)` | Update atomic positions (Angstrom, Cartesian) |
| `update_cell(cell)` | Update cell vectors (Angstrom) |
| `get_positions()` | Get atomic positions (nat, 3) in Angstrom |
| `get_cell()` | Get cell vectors (3, 3) in Angstrom |

**Data Access Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `charge` | ChargeData | Charge density data |
| `energy` | EnergyData | Energy components |
| `hamiltonian` | HamiltonianData | H(k), S(k), H(R), S(R) matrices |
| `density_matrix` | DensityMatrixData | DM(k), DM(R) matrices |
| `force` | ForceData | Forces (after `cal_force()`) |
| `stress` | StressData | Stress tensor (after `cal_stress()`) |
| `is_converged` | bool | SCF convergence status |
| `niter` | int | Current iteration number |
| `drho` | float | Charge density difference |
| `nks` | int | Number of k-points |
| `nbasis` | int | Number of basis functions |
| `nbands` | int | Number of bands |
| `nspin` | int | Number of spin channels |
| `nat` | int | Number of atoms |

**Data Access Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_psi(ik)` | np.ndarray | Wave function coefficients (nbands, nbasis) |
| `get_eigenvalues(ik)` | np.ndarray | Eigenvalues (nbands,) |
| `get_occupations(ik)` | np.ndarray | Occupation numbers (nbands,) |
| `get_kvec(ik)` | np.ndarray | K-vector in direct coordinates (3,) |
| `get_kweights()` | np.ndarray | K-point weights (nks,) |

**Example:**

```python
from pyabacus.esolver import LCAOWorkflow
import numpy as np

workflow = LCAOWorkflow("./Si_scf/", gamma_only=True)
workflow.initialize()

# Run SCF
result = workflow.run_scf(max_iter=100)

# Access internal data
print(f"Number of k-points: {workflow.nks}")
print(f"Number of bands: {workflow.nbands}")

# Get eigenvalues for first k-point
eigenvalues = workflow.get_eigenvalues(0)
print(f"Eigenvalues: {eigenvalues}")

# Get Hamiltonian matrices
ham = workflow.hamiltonian
print(f"H(k) shape: {ham.Hk[0].shape}")

# Cleanup
workflow.cleanup()
```

### PWWorkflow Class

High-level Python interface for plane wave calculations, also inheriting from
`_BaseWorkflow`. All shared methods (force, stress, position/cell update, cleanup,
data access properties) are available identically to `LCAOWorkflow`.

**Constructor:**

```python
PWWorkflow(input_dir: str, precision: str = "double")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | str | - | Directory containing INPUT, STRU, KPT files |
| `precision` | str | "double" | "single" (complex\<float\>) or "double" (complex\<double\>) |

**PW-specific Properties and Methods:**

| Member | Type | Description |
|--------|------|-------------|
| `npwx` | int (property) | Maximum number of plane waves |
| `get_npw(ik)` | int | Number of plane waves for k-point ik |

**Example:**

```python
from pyabacus.esolver import PWWorkflow

workflow = PWWorkflow("./Si_scf/", precision="double")
workflow.initialize()

# Run SCF (Python-side loop with convergence check)
result = workflow.run_scf(max_iter=100)
print(f"Converged: {result.converged}")
print(f"Energy: {result.energy.etot} Ry")

# PW-specific: get plane wave counts
print(f"npwx: {workflow.npwx}")
print(f"npw(k=0): {workflow.get_npw(0)}")

workflow.cleanup()
```

### Callback System

The callback system allows you to register functions that are called at specific points during the SCF calculation.

**Callback Events:**

| Event | When Called | Callback Signature |
|-------|-------------|-------------------|
| `before_scf` | After `before_scf()` call | `callback(workflow)` |
| `after_iter` | After each SCF iteration | `callback(workflow, iter_num)` |
| `before_after_scf` | Before `after_scf()` - main breakpoint | `callback(workflow)` |
| `after_scf` | After `after_scf()` call | `callback(workflow)` |

**Methods:**

```python
# Register a callback
workflow.register_callback(event: str, callback: Callable)

# Unregister a callback
workflow.unregister_callback(event: str, callback: Callable) -> bool

# Clear callbacks
workflow.clear_callbacks(event: str = None)  # None clears all
```

**Example - SCF Monitoring:**

```python
from pyabacus.esolver import LCAOWorkflow

def monitor_scf(workflow, iter_num):
    """Called after each SCF iteration."""
    print(f"Iter {iter_num}: drho = {workflow.drho:.2e}")

def save_data(workflow):
    """Called before after_scf - main breakpoint."""
    import numpy as np
    np.save("charge.npy", workflow.charge.rho)
    np.save("energy.npy", workflow.energy.etot)
    print(f"Final energy: {workflow.energy.etot} Ry")

workflow = LCAOWorkflow("./Si_scf/")
workflow.initialize()

# Register callbacks
workflow.register_callback('after_iter', monitor_scf)
workflow.register_callback('before_after_scf', save_data)

# Run SCF - callbacks will be called automatically
result = workflow.run_scf()

workflow.cleanup()
```

**Example - Custom SCF Loop:**

```python
from pyabacus.esolver import LCAOWorkflow

workflow = LCAOWorkflow("./Si_scf/")
workflow.initialize()

# Manual SCF control
workflow.before_scf(istep=0)

for iter_num in range(1, 101):
    workflow.run_scf_step(iter_num)

    # Custom logic
    if workflow.drho < 1e-8:
        print(f"Converged at iteration {iter_num}")
        break

    # Access data during SCF
    if iter_num % 10 == 0:
        print(f"Iter {iter_num}: E = {workflow.energy.etot:.6f} Ry")

workflow.after_scf(istep=0)
workflow.cleanup()
```

### Data Type Classes

PyABACUS provides dataclass containers for structured data with unit conversion methods.

#### ChargeData

```python
@dataclass
class ChargeData:
    rho: np.ndarray      # Charge density on real-space grid
    nspin: int           # Number of spin channels
    nrxx: int            # Number of real-space grid points
    rhog: np.ndarray     # Charge density in G-space (optional)
    ngmc: int            # Number of G-vectors (optional)
```

**Methods:**
- `total_charge()` - Get total charge
- `spin_density()` - Get spin density (nspin=2 only)

#### EnergyData

```python
@dataclass
class EnergyData:
    etot: float           # Total energy (Ry)
    eband: float          # Band energy (Ry)
    hartree_energy: float # Hartree energy (Ry)
    etxc: float           # Exchange-correlation energy (Ry)
    ewald_energy: float   # Ewald energy (Ry)
    demet: float          # -TS term for metals (Ry)
    exx: float            # Exact exchange energy (Ry)
    evdw: float           # van der Waals energy (Ry)
```

**Methods:**
- `to_dict()` - Convert to dictionary
- `to_eV()` - Get total energy in eV

#### ForceData

```python
@dataclass
class ForceData:
    forces: np.ndarray  # Forces (nat, 3) in Ry/Bohr
    nat: int            # Number of atoms
```

**Methods:**
- `to_eV_Ang()` - Convert forces to eV/Å
- `to_dict()` - Convert to dictionary

#### StressData

```python
@dataclass
class StressData:
    stress: np.ndarray  # Stress tensor (3, 3) in kbar
```

**Methods:**
- `to_voigt()` - Convert to Voigt notation (6,): xx, yy, zz, yz, xz, xy
- `to_eV_Ang3()` - Convert to eV/Å³ in Voigt notation
- `to_dict()` - Convert to dictionary

#### HamiltonianData

```python
@dataclass
class HamiltonianData:
    Hk: List[np.ndarray]  # H(k) matrices for each k-point
    Sk: List[np.ndarray]  # S(k) matrices for each k-point
    HR: np.ndarray        # H(R) in real space
    SR: np.ndarray        # S(R) in real space
    nbasis: int           # Number of basis functions
    nks: int              # Number of k-points
```

**Methods:**
- `get_Hk(ik)` - Get H(k) for k-point ik
- `get_Sk(ik)` - Get S(k) for k-point ik

#### DensityMatrixData

```python
@dataclass
class DensityMatrixData:
    DMK: List[np.ndarray]  # DM(k) for each k-point
    DMR: np.ndarray        # DM(R) in real space
    nks: int               # Number of k-points
    nrow: int              # Number of rows
    ncol: int              # Number of columns
```

**Methods:**
- `get_DMK(ik)` - Get DM(k) for k-point ik
- `trace(ik)` - Get trace of DM(k)

#### SCFResult

```python
@dataclass
class SCFResult:
    converged: bool       # SCF convergence status
    niter: int            # Number of iterations
    drho: float           # Final charge density difference
    energy: EnergyData    # Energy data
    charge: ChargeData    # Charge data
```

**Methods:**
- `summary()` - Get formatted summary string

### Unit Conversion Constants

PyABACUS uses atomic units internally. All conversion constants are defined in
`pyabacus.constants` (single source of truth):

```python
from pyabacus.constants import (
    RY_TO_EV,
    BOHR_TO_ANG,
    ANG_TO_BOHR,
    RY_BOHR_TO_EV_ANG,
    KBAR_TO_EV_ANG3,
    ENERGY_FIELDS,
)

# Constants
RY_TO_EV = 13.605698              # 1 Ry = 13.605698 eV
BOHR_TO_ANG = 0.529177249         # 1 Bohr = 0.529177 Å
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG  # 1 Å in Bohr
RY_BOHR_TO_EV_ANG = RY_TO_EV / BOHR_TO_ANG  # ~25.7112 Force: Ry/Bohr → eV/Å
KBAR_TO_EV_ANG3 = 1/1602.1766208 # Stress: kbar → eV/Å³
ENERGY_FIELDS = ['etot', 'eband', 'hartree_energy', 'etxc',
                 'ewald_energy', 'demet', 'exx', 'evdw']
```

**Example:**

```python
from pyabacus.esolver import LCAOWorkflow
from pyabacus.constants import RY_TO_EV

workflow = LCAOWorkflow("./Si_scf/")
workflow.initialize()
result = workflow.run_scf()

# Convert energy from Ry to eV
energy_eV = result.energy.etot * RY_TO_EV
print(f"Energy: {energy_eV:.6f} eV")

# Or use the convenience method
energy_eV = result.energy.to_eV()

workflow.cleanup()
```

## ASE Calculator Integration

PyABACUS provides an ASE-compatible Calculator for seamless integration with ASE's optimization and molecular dynamics tools.

### Overview

The `AbacusCalculator` class implements the ASE Calculator interface, supporting:
- Energy, forces, and stress calculations
- Geometry optimization with ASE optimizers (BFGS, FIRE, etc.)
- Cell relaxation with `UnitCellFilter`
- Molecular dynamics simulations

### Calculator Modes

| Mode | Description | Memory | Efficiency | Use Case |
|------|-------------|--------|------------|----------|
| `ESOLVER` | Direct ESolver with memory persistence | Persistent | High | Geometry optimization, MD |
| `DRIVER` | Independent calculations via subprocess | None | Lower | Single-point, isolation needed |

### AbacusCalculator Class

```python
from pyabacus.ase import AbacusCalculator

calc = AbacusCalculator(
    input_dir: str = '.',
    gamma_only: bool = True,
    mode: Union[CalculatorMode, str] = 'esolver',
    nprocs: int = 1,
    nthreads: int = 1,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | str | '.' | Directory with INPUT, STRU, KPT |
| `gamma_only` | bool | True | Gamma-only calculation (ESolver mode) |
| `mode` | str/CalculatorMode | 'esolver' | 'esolver' or 'driver' |
| `nprocs` | int | 1 | MPI processes (Driver mode only) |
| `nthreads` | int | 1 | OpenMP threads (Driver mode only) |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `workflow` | LCAOWorkflow | Underlying workflow (ESolver mode only) |

**Methods:**

| Method | Description |
|--------|-------------|
| `cleanup()` | Release resources (ESolver mode) |
| `get_energy_components()` | Get detailed energy breakdown (ESolver mode) |

### Basic Usage

**ESolver Mode (Recommended for optimization):**

```python
from ase import Atoms
from pyabacus.ase import AbacusCalculator

# Use context manager for automatic cleanup
with AbacusCalculator(input_dir='./Si_scf/', gamma_only=True) as calc:
    atoms = Atoms('Si2',
                  positions=[[0, 0, 0], [1.35, 1.35, 1.35]],
                  cell=[5.43, 5.43, 5.43],
                  pbc=True)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    print(f"Energy: {energy:.6f} eV")
    print(f"Max force: {np.max(np.abs(forces)):.6f} eV/Å")
```

**Driver Mode:**

```python
from pyabacus.ase import AbacusCalculator

calc = AbacusCalculator(
    input_dir='./Si_scf/',
    mode='driver',
    nprocs=4
)
atoms.calc = calc
energy = atoms.get_potential_energy()
```

### Geometry Optimization

```python
from ase import Atoms
from ase.optimize import BFGS
from pyabacus.ase import AbacusCalculator

with AbacusCalculator(input_dir='./Si_scf/') as calc:
    atoms = Atoms('Si2',
                  positions=[[0, 0, 0], [1.35, 1.35, 1.35]],
                  cell=[5.43, 5.43, 5.43],
                  pbc=True)
    atoms.calc = calc

    # Run BFGS optimization
    opt = BFGS(atoms, trajectory='opt.traj')
    opt.run(fmax=0.01)  # Converge when max force < 0.01 eV/Å

    print(f"Optimized energy: {atoms.get_potential_energy():.6f} eV")
```

### Cell Relaxation

```python
from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from pyabacus.ase import AbacusCalculator

with AbacusCalculator(input_dir='./Si_scf/') as calc:
    atoms = Atoms('Si2',
                  positions=[[0, 0, 0], [1.35, 1.35, 1.35]],
                  cell=[5.43, 5.43, 5.43],
                  pbc=True)
    atoms.calc = calc

    # Wrap atoms with UnitCellFilter for cell optimization
    ucf = UnitCellFilter(atoms)
    opt = BFGS(ucf, trajectory='cell_opt.traj')
    opt.run(fmax=0.01)

    print(f"Optimized cell:\n{atoms.get_cell()}")
```

### Energy Components

```python
from pyabacus.ase import AbacusCalculator

with AbacusCalculator(input_dir='./Si_scf/') as calc:
    atoms.calc = calc
    atoms.get_potential_energy()  # Run calculation first

    # Get detailed energy breakdown (ESolver mode only)
    components = calc.get_energy_components()
    for name, value in components.items():
        print(f"{name}: {value:.6f} eV")
```

### Mode Comparison

| Feature | ESolver Mode | Driver Mode |
|---------|--------------|-------------|
| Memory persistence | Yes | No |
| Calculation efficiency | High | Lower |
| Energy components access | Yes | No |
| Cleanup required | Yes | No |
| Parallel support | Via ABACUS build | nprocs/nthreads |
| Best for | Optimization, MD | Single-point |

## Troubleshooting

### ABACUS executable not found

```
FileNotFoundError: ABACUS executable not found
```

**Solution:** Ensure ABACUS is installed and in your PATH:
```bash
export PATH=/path/to/abacus/bin:$PATH
```

Or specify the path explicitly:
```python
result = abacus("./", abacus_path="/path/to/abacus")
```

### MPI errors

```
mpirun: command not found
```

**Solution:** Install MPI and ensure it's in PATH:
```bash
# Ubuntu/Debian
sudo apt install openmpi-bin

# macOS
brew install open-mpi
```

### ESolver module not available

```
ImportError: Could not import ESolver bindings
```

**Solution:** Build pyabacus with ESolver support using static build (recommended):
```bash
cd python/pyabacus
pip install -e . --no-build-isolation
```

Or use dynamic build if you have ABACUS installed:
```bash
export ABACUS_INSTALL_DIR=/path/to/install
cd python/pyabacus
pip install -e . --no-build-isolation -C cmake.define.PYABACUS_ESOLVER_DYNAMIC=ON
```

### LCAOWorkflow initialization fails

```
RuntimeError: Workflow not initialized. Call initialize() first.
```

**Solution:** Always call `initialize()` before using the workflow:
```python
workflow = LCAOWorkflow("./Si_scf/")
workflow.initialize()  # Don't forget this!
result = workflow.run_scf()
```

### ASE Calculator not working

```
ImportError: ASE is required for AbacusCalculator
```

**Solution:** Install ASE:
```bash
pip install ase
```

### Memory issues with ESolver mode

If you encounter memory issues with ESolver mode during long optimization runs:

**Solution:** Use the context manager or call `cleanup()` explicitly:
```python
# Option 1: Context manager (recommended)
with AbacusCalculator(input_dir='./') as calc:
    # ... calculations ...
# Resources automatically released

# Option 2: Manual cleanup
calc = AbacusCalculator(input_dir='./')
try:
    # ... calculations ...
finally:
    calc.cleanup()
```

### Convergence issues

If SCF doesn't converge:

1. Check input parameters (ecutwfc, mixing parameters)
2. Use callbacks to monitor convergence:
```python
def monitor(wf, iter_num):
    print(f"Iter {iter_num}: drho = {wf.drho:.2e}")

workflow.register_callback('after_iter', monitor)
result = workflow.run_scf(max_iter=200)
```

### Unit conversion errors

Remember that internal ABACUS units differ from ASE units:
- Energy: Rydberg (ABACUS) vs eV (ASE)
- Length: Bohr (ABACUS) vs Angstrom (ASE)
- Force: Ry/Bohr (ABACUS) vs eV/Å (ASE)

Use the provided conversion methods or import constants from `pyabacus.constants`:
```python
from pyabacus.constants import RY_TO_EV

# Forces
forces_eV_Ang = workflow.force.to_eV_Ang()

# Stress
stress_voigt = workflow.stress.to_eV_Ang3()

# Energy
energy_eV = workflow.energy.to_eV()

# Or manual conversion
energy_eV_manual = workflow.energy.etot * RY_TO_EV
```
