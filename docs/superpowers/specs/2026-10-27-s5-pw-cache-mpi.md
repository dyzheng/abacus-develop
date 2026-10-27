# S5.1 PW allocation cache and MPI validation

Date: 2026-10-27

Plan: `docs/superpowers/plans/2026-10-21-unified-allocation-execution.md`,
Task S5.1.

This round closes the immediate S5.1 blocker: the PW allocation operator had
been wired and could complete an SCF at `mpirun -np 2`, but it failed at higher
MPI rank counts and still repeated band-independent FFT/overlap work on every
Hamiltonian application.  The changes below make the target/product cache
persistent, batch the pool reductions, and populate the onsite projector
workspace before the allocation target rows are read.

## 1. Test plan

1. Verify that the prepared orthogonal target context reproduces the legacy
   one-band kernel.
2. Rebuild `abacus_basic_para` and run the PW allocation unit tests only in
   MPI mode, with one OpenMP thread per rank.
3. Run the converged H2O/O-d fixed-mu SCF at 2 and 4 MPI ranks and compare the
   constraint audit, total moment, total charge and final energy.
4. Run a 5-step `kpar=2` case on 4 MPI ranks to exercise both intra-pool and
   cross-pool reductions without a long production run.

Per the current testing rule, PW tests use `OMP_NUM_THREADS=1` and MPI ranks;
serial PW executables are not used for acceptance in this round.

## 2. Test setup

- Branch: `feat/unified-local-moment-allocation`
- Build: `build/abacus_basic_para`, CPU, OpenMPI
- H2O, O-d target (`constraint_alloc_target_l=2`), Becke weights
- `nspin=2`, `nbands=8`, `ecutwfc=20 Ry`, `ecutrho=80 Ry`
- `fixed_mu=0.01 Ry`, `scf_thr=1e-7`, 15 Angstrom cubic cell
- MPI environment:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

### 2.1 Code changes under test

- `AllocPwCache` keys the target context by `(ik, weight_grid, target_l)` and
  the products by `(ik, spin, target_generation, lambda)`.  The cache is owned
  through `std::unique_ptr` and mutated only inside the const `act()` path.
- `build_pw_alloc_orthogonal_context()` prepares `phi G`, `V phi G` and
  `G^T m_raw G` once per target/product frame.  The per-band path then only
  forms the new band overlaps and vector combinations.
- Projector overlaps, raw target overlaps and the complete `R x R` target
  matrix are reduced in one pool collective instead of one collective per
  element.  The triangular projector overlap is mirrored after reduction.
- `build_target_context()` now calls `overlap_proj_psi()` before reading
  `tab_atomic_`.  `tab_atomic_` is the OnsiteProjector VKB workspace populated
  by `cal_becp()`, so an allocation-only run must not assume that another
  onsite consumer has already filled it.

## 3. Results

### 3.1 Build and MPI unit tests

```text
cmake --build build --target abacus_basic_para -j 14                 PASS
ctest -R '^MODULE_ESTATE_constraint_alloc_pw_mpi(_2np|_4np)?$'      PASS (3/3)
```

The three MPI tests were `alloc_pw_mpi`, `alloc_pw_mpi_2np` and
`alloc_pw_mpi_4np`; all ran with `OMP_NUM_THREADS=1`.

The prepared-kernel unit test reports:

```text
alloc_pw apply: max_diff=2.082e-17
alloc_pw prepared apply: max_diff=0.000e+00
```

### 3.2 Converged SCF at 2 and 4 ranks

Commands:

```bash
OMP_NUM_THREADS=1 mpirun -np 2 ../../../build/abacus_basic_para
OMP_NUM_THREADS=1 mpirun -np 4 ../../../build/abacus_basic_para
```

Both runs converged in 17 CG iterations.  The previous `np=4` failure,
`PW allocation target projector has zero numerical rank ... max_ST=0.000000`,
no longer occurs.

| Quantity | `np=2` | `np=4` | Difference |
|---|---:|---:|---:|
| Final energy (eV) | -442.15878835310752 | -442.15878835305523 | 5.2e-11 |
| Constraint charge `q` (e) | -0.009999076424 | -0.009999076422 | 2.0e-12 |
| Constraint energy `e_con` (Ry) | -0.005099990764 | -0.005099990764 | 0 |
| `sum_n_T` (e) | 0.124616325 | 0.124616325 | < 1e-12 |
| `sum_m_T` (mu_B) | 6.59859993e-04 | 6.59859993e-04 | < 1e-12 |
| `sum_n` (e) | 8.00000000 | 8.00000000 | 0 |
| `dsum_n` (e) | -8.65e-13 | -2.17e-13 | -6.48e-13 |
| `sum_m` (mu_B) | 3.12e-15 | 8.70e-16 | 2.25e-15 |

Both ranks reported `#SCF IS CONVERGED#`, rank/cond `5/1.00007`, and the same
fixed-mu status.  The 4-rank run took 51 s total; the 2-rank run took 82 s.
Hamiltonian-application step times were 2.1--8.2 s at `np=2` and 1.5--3.8 s at
`np=4`, compared with the earlier uncached WIP range of 12--35 s per step.

### 3.3 `kpar=2` cross-pool smoke test

Input: two k-points (`K_POINTS 1 1 2 0 0 0`), `kpar=2`, 4 MPI ranks, 5 SCF
steps.  The run completed without deadlock:

```text
ALLOC-PW audit sum_n_T=1.24286548e-01 sum_m_T=6.26336060e-04
              sum_n=8.00000000e+00 nelec=8.00000000e+00
              dsum_n=-7.39e-13
              sum_m=-9.98780202e-03 tot_mag=-9.98780202e-03
              dsum_m=-3.09e-16
```

The finite `sum_m` is the physical total magnetization of this non-zero-mu
frame, not a reduction residual: it agrees with `tot_mag` at the printed
precision and `dsum_m` is at machine noise.

## 4. Analysis

1. The 4-rank zero-rank failure was caused by reading `tab_atomic_` before
   `cal_becp()` had populated the VKB workspace.  The 2-rank path happened to
   receive a populated table from an earlier onsite consumer; rank-count
   dependent operator ordering exposed the latent initialization bug.  Calling
   `overlap_proj_psi()` explicitly makes target construction independent of
   the consumer order.
2. The 2/4-rank final energies and constraint observables now agree to
   floating-point reduction noise.  The small `dsum_n`/`dsum_m` differences are
   expected from different global reduction trees, not physical rank effects.
3. Batching the reductions removes the repeated per-target collectives.  The
   persistent cache removes repeated target FFTs and `R x R` algebra across
   bands and Hamiltonian applications.  The remaining per-band work is the
   unavoidable `V|psi>` FFT, raw overlap and vector accumulation.
4. The `kpar=2` smoke run confirms that the pool correction and the global
   all-pool sum use the correct pool count `KPAR * bndpar`; a full convergence
   run is still desirable for the regression matrix, but the reduction path is
   exercised.

## 5. Next steps

1. Add the PW frozen-target position force path and compare the analytic O-z
   force with a three-point finite difference (`< 0.0129 eV/A`).
2. Run the full `kpar=2` regression once the force path no longer needs the
   current 5-step smoke input.
3. Extend cache invalidation with an explicit geometry/target generation key
   before enabling PW allocation relax/MD.
4. Continue S5.2 with the same-state Becke/Hirshfeld comparison table and the
   Hirshfeld strain/stress derivative.
