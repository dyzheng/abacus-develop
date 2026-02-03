# ASE

## Introduction

[ASE](https://wiki.fysik.dtu.dk/ase/) (Atomic Simulation Environment) provides a set of Python tools for setting, running, and analysing atomic simulations. ABACUS provides two ways to integrate with ASE:

1. **ase-abacus** (subprocess-based): An external calculator that runs ABACUS as a subprocess. Suitable for simple calculations and when you need full isolation between calculations.

2. **pyabacus.ase** (direct integration): A built-in calculator that uses pyabacus directly, offering two flexible modes:
   - **ESolver Mode**: Memory-persistent calculations for efficient geometry optimization
   - **Driver Mode**: Independent calculations via file I/O for maximum flexibility

## Method 1: ase-abacus (Subprocess-based)

### Installation

```bash
git clone https://gitlab.com/1041176461/ase-abacus.git
cd ase-abacus
pip install .
```

Another direct way:
```bash
pip install git+https://gitlab.com/1041176461/ase-abacus.git
```

### Environment variables

[ABACUS](http://abacus.ustc.edu.cn) supports two types of basis sets: PW, LCAO. The path of pseudopotential and numerical orbital files can be set throught the environment variables `ABACUS_PP_PATH` and `ABACUS_ORBITAL_PATH`, respectively, e.g.:

```bash
  PP=${HOME}/pseudopotentials
  ORB=${HOME}/orbitals
  export ABACUS_PP_PATH=${PP}
  export ABACUS_ORBITAL_PATH=${ORB}
```

For PW calculations, only `ABACUS_PP_PATH` is needed. For LCAO calculations, both `ABACUS_PP_PATH` and `ABACUS_ORBITAL_PATH` should be set.

Also, one can manally set the paths of PP and ORB when using ABACUS calculator in ASE.

### ABACUS Calculator

The default initialization command for the ABACUS calculator is

```python
from ase.calculators.abacus import Abacus
```

In order to run a calculation, you have to ensure that at least the following parameters are specified, either in the initialization or as environment variables:

|keyword         |description
|:---------------|:----------------------------------------------------------
|`pp`            |dict of pseudopotentials for involved elememts, <br> such as `pp={'Al':'Al_ONCV_PBE-1.0.upf',...}`.
|`pseudo_dir`    |directory where the pseudopotential are located, <br> Can also be specified with the `ABACUS_PP_PATH` <br> environment variable. Default: `pseudo_dir=./`.
|`basis`         |dict of orbital files for involved elememts, such as <br> `basis={'Al':'Al_gga_10au_100Ry_4s4p1d.orb'}`.<br> It must be set if you want to do LCAO <br> calculations. But for pw calculations, it can be omitted.
|`basis_dir`     |directory where the orbital files are located, <br> Can also be specified with the `ABACUS_ORBITAL_PATH`<br> environment variable. Default: `basis_dir=./`.
|`xc`            |which exchange-correlation functional is used.<br> An alternative way to set this parameter is via <br> seting `dft_functional` which is an ABACUS <br> parameter used to specify exchange-correlation <br> functional
|`kpts`          |a tuple (or list) of 3 integers `kpts=(int, int, int)`, <br>it is interpreted as the dimensions of a Monkhorst-Pack <br>  grid, when `kmode` is `Gamma` or `MP`. It is <br>  interpreted as k-points, when `kmode` is `Direct`,<br>  `Cartesian` or `Line`, and `knumber` should also<br>  be set in these modes to denote the number of k-points.<br>  Some other parameters for k-grid settings:<br>  including `koffset` and `kspacing`.

For more information on pseudopotentials and numerical orbitals, please visit [ABACUS]. The elaboration of input parameters can be found [here](../input_files/input-main.md).


The input parameters can be set like::
```python
  # for ABACUS calculator
  calc = Abacus(profile=profile,
                ecutwfc=100,
                scf_nmax=100,
                smearing_method='gaussian',
                smearing_sigma=0.01,
                basis_type='pw',
                ks_solver='dav',
                calculation='scf',
                pp=pp,
                basis=basis,
                kpts=kpts)
```

The command to run jobs can be set by specifying `AbacusProfile`::

```python
  from ase.calculators.abacus import AbacusProfile
  # for OpenMP setting inside python env
  import os
  os.environ("OMP_NUM_THREADS") = 1
  # for MPI setting used in abacus
  mpi_num = 4
  # for ABACUS Profile
  abacus = '/usr/local/bin/abacus' # specify abacus exec
  profile = AbacusProfile(command=f'mpirun -n {mpi_num} {abacus}')  # directly the command for running ABACUS
```

in which `abacus` sets the absolute path of the `abacus` executable.

---

## Method 2: pyabacus.ase (Direct Integration)

PyABACUS provides a built-in ASE calculator with **dual-mode** support, offering flexibility for different use cases.

### Installation

```bash
cd /path/to/abacus-develop/python/pyabacus
pip install -e .
```

### Two Calculation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **ESolver Mode** (default) | Memory-persistent, reuses workflow between calculations | Geometry optimization, MD, sequential calculations |
| **Driver Mode** | Independent calculations via file I/O | Single-point calculations, batch processing, isolation needed |

### Quick Start

```python
from ase import Atoms
from pyabacus.ase import AbacusCalculator, CalculatorMode

# Create atoms object
atoms = Atoms('Si2',
              positions=[[0, 0, 0], [1.35, 1.35, 1.35]],
              cell=[5.43, 5.43, 5.43],
              pbc=True)

# ESolver Mode (default) - efficient for sequential calculations
calc = AbacusCalculator(input_dir='./Si_scf/', gamma_only=True)
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

# Driver Mode - independent calculations
calc = AbacusCalculator(
    input_dir='./Si_scf/',
    mode=CalculatorMode.DRIVER,  # or mode='driver'
    nprocs=4,
    nthreads=2,
)
atoms.calc = calc
energy = atoms.get_potential_energy()
```

### ESolver Mode: Memory-Persistent Calculations

ESolver mode keeps the calculation state in memory, making it highly efficient for:
- Geometry optimization (BFGS, FIRE, etc.)
- Molecular dynamics
- Any workflow requiring multiple sequential calculations

**Key Features:**
- Workflow is initialized once and reused
- Position/cell updates don't require re-initialization
- Requires explicit cleanup or context manager usage

```python
from ase.optimize import BFGS
from pyabacus.ase import AbacusCalculator

# Using context manager (recommended) - auto cleanup
with AbacusCalculator(input_dir='./Si_relax/') as calc:
    atoms.calc = calc

    # Run geometry optimization
    opt = BFGS(atoms, trajectory='opt.traj')
    opt.run(fmax=0.01)

    print(f"Optimized energy: {atoms.get_potential_energy():.4f} eV")
# Resources automatically released here

# Manual cleanup (alternative)
calc = AbacusCalculator(input_dir='./Si_relax/')
atoms.calc = calc
opt = BFGS(atoms)
opt.run(fmax=0.01)
calc.cleanup()  # Explicitly release resources
```

**Cell Relaxation with UnitCellFilter:**

```python
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from pyabacus.ase import AbacusCalculator

with AbacusCalculator(input_dir='./Si_relax/') as calc:
    atoms.calc = calc

    # Relax both cell and atomic positions
    ucf = UnitCellFilter(atoms)
    opt = BFGS(ucf)
    opt.run(fmax=0.01)

    print(f"Final cell:\n{atoms.cell[:]}")
```

### Driver Mode: Independent Calculations

Driver mode runs each calculation independently via the `abacus()` function, suitable for:
- Single-point calculations
- Batch processing with full isolation
- When you need fresh calculations each time

**Key Features:**
- Each calculation is completely independent
- No memory persistence between calculations
- Supports MPI parallelization via `nprocs` and `nthreads`

```python
from pyabacus.ase import AbacusCalculator, CalculatorMode

# Driver mode with parallel execution
calc = AbacusCalculator(
    input_dir='./Si_scf/',
    mode='driver',  # String also accepted
    nprocs=4,       # MPI processes
    nthreads=2,     # OpenMP threads
)
atoms.calc = calc

# Each call runs a fresh ABACUS calculation
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

**Batch Processing Example:**

```python
from pyabacus.ase import AbacusCalculator, CalculatorMode
import numpy as np

# Calculate energy vs. lattice constant
lattice_constants = np.linspace(5.0, 5.5, 11)
energies = []

for a in lattice_constants:
    atoms = Atoms('Si2',
                  positions=[[0, 0, 0], [a/4, a/4, a/4]],
                  cell=[a, a, a],
                  pbc=True)

    # Each calculation is independent
    calc = AbacusCalculator(
        input_dir='./Si_eos/',
        mode=CalculatorMode.DRIVER,
    )
    atoms.calc = calc
    energies.append(atoms.get_potential_energy())

# Find equilibrium lattice constant
min_idx = np.argmin(energies)
print(f"Equilibrium lattice constant: {lattice_constants[min_idx]:.3f} Ang")
```

### Mode Comparison

| Feature | ESolver Mode | Driver Mode |
|---------|--------------|-------------|
| Memory persistence | Yes | No |
| Initialization overhead | Once | Every calculation |
| Best for | Sequential calculations | Independent calculations |
| Cleanup required | Yes (context manager or manual) | No |
| MPI support | Via ABACUS build | Via `nprocs` parameter |
| Access to workflow | `calc.workflow` | None |

### API Reference

#### AbacusCalculator

```python
AbacusCalculator(
    input_dir: str = '.',
    gamma_only: bool = True,
    mode: Union[CalculatorMode, str] = CalculatorMode.ESOLVER,
    nprocs: int = 1,
    nthreads: int = 1,
    **kwargs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | str | `'.'` | Directory containing INPUT, STRU, KPT files |
| `gamma_only` | bool | True | Use gamma-only calculation (ESolver mode) |
| `mode` | CalculatorMode/str | `ESOLVER` | Calculation mode: `'esolver'` or `'driver'` |
| `nprocs` | int | 1 | MPI processes (Driver mode only) |
| `nthreads` | int | 1 | OpenMP threads (Driver mode only) |

**Methods:**

| Method | Description |
|--------|-------------|
| `cleanup()` | Release resources (ESolver mode) |
| `get_potential_energy()` | Get total energy in eV |
| `get_forces()` | Get forces in eV/Å |
| `get_stress()` | Get stress in Voigt notation (eV/Å³) |
| `get_energy_components()` | Get detailed energy breakdown (ESolver mode) |

**Properties:**

| Property | Description |
|----------|-------------|
| `workflow` | Underlying LCAOWorkflow (ESolver mode) or None |
| `mode` | Current calculation mode |

### Advanced: Accessing Internal Data (ESolver Mode)

In ESolver mode, you can access internal ABACUS data through the workflow:

```python
from pyabacus.ase import AbacusCalculator

with AbacusCalculator(input_dir='./Si_scf/') as calc:
    atoms.calc = calc
    energy = atoms.get_potential_energy()

    # Access workflow for internal data
    wf = calc.workflow

    # Get Hamiltonian matrices
    H_data = wf.hamiltonian
    print(f"H(k) shape: {H_data.Hk[0].shape}")

    # Get density matrix
    DM_data = wf.density_matrix
    print(f"DM(k) shape: {DM_data.DMK[0].shape}")

    # Get detailed energy components
    components = calc.get_energy_components()
    for name, value in components.items():
        print(f"  {name}: {value:.6f} eV")
```

---

## MD Analysis
After molecular dynamics calculations, the log file `running_md.log` can be read. If the 'STRU_MD_*' files are not continuous (e.g. 'STRU_MD_0', 'STRU_MD_5', 'STRU_MD_10'...), the index parameter of read should be as a slice object. For example, when using the command `read('running_md.log', index=slice(0, 15, 5), format='abacus-out')` to parse 'running_md.log', 'STRU_MD_0', 'STRU_MD_5' and 'STRU_MD_10' will be read.

The `MD_dump` file is also supported to be read-in by `read('MD_dump', format='abacus-md')`


## SPAP Analysis

[SPAP](https://github.com/chuanxun/StructurePrototypeAnalysisPackage) (Structure Prototype Analysis Package) is written by Dr. Chuanxun Su to analyze symmetry and compare similarity of large amount of atomic structures. The coordination characterization function (CCF) is used to
measure structural similarity. An unique and advanced clustering method is developed to automatically classify structures into groups.


If you use this program and method in your research, please read and cite the publication:

`Su C, Lv J, Li Q, Wang H, Zhang L, Wang Y, Ma Y. Construction of crystal structure prototype database: methods and applications. J Phys Condens Matter. 2017 Apr 26;29(16):165901.`

and you should install it first with command `pip install spap`.
