# Validation report for `lts-backport-accel-2026Q4`

Date: 2026-10-22
Base: `LTS = 725bf475a`
Branch head at validation time: `a675e976b` (docs only after this)
Hardware: 20-core x86 CPU, NVIDIA RTX 3070 Ti, CUDA 12.x

## 1. Builds

| Build | Options | Result |
|---|---|---|
| `build_nspin4` | CPU, LCAO, ELPA, tests ON | pass |
| fresh CPU archive | `ENABLE_LIBXC=ON`, `ENABLE_LIBRI=ON`, LCAO+ELPA | pass |
| `build_gpu` | CUDA, LCAO, ELPA, `ENABLE_LIBXC=ON` | pass |
| fresh CUDA archive | + `ENABLE_FLOAT_FFTW=ON` | pass |

Notes:
- Single precision PW requires `ENABLE_FLOAT_FFTW=ON`. Without it even the
  unmodified LTS baseline fails at `START CHARGE` with
  `cudaErrorInvalidValue`.
- The CUDA build needs the system `nvtx3` header; it is present in
  `/usr/include/nvtx3` and was made visible to the conda nvcc for the
  build only (symlink removed afterwards).
- ROCm was not built (no ROCm hardware/toolchain).

## 2. Unit tests

Run from `build_nspin4` (CPU) unless noted:

| Test | Result |
|---|---|
| `HSolver_dav` | pass |
| `HSolver_dav_float` | pass |
| `HSolver_LCAO` (includes new genelpa upper-triangle tests) | pass |
| `hcontainer_test` | pass |
| `hcontainer_complex_test` | pass |
| `hcontainer_output_test` | pass |
| `hcontainer_readCSR_test` | pass |
| `pw_test` | pass |
| `basis_pw_serial` | pass |
| `basis_pw_k_serial` | pass |
| `XCTest_PBE` | pass |
| `XCTest_HSE` | pass |
| `XCTest_PZ_SPN` | pass |
| `XCTest_GRADCORR` | pass |
| `XCTest_SCAN` | pass |
| `XCTest_VXC` | pass |
| `XC_Functional_UTs` | pass |

The first run exposed two build defects that were fixed in this branch:
- `XCTest_GRADCORR` could not link the out-of-line
  `get_gcar_data<double>()` accessors (fixed by making them inline).
- The device-only accessor cache was not valid in the stress path for
  single-precision GPU SCAN (fixed by uploading Real-precision host
  copies).

## 3. CPU integration tests

Harness: `tests/integrate/Autotest.sh`, 4 MPI x 1 OMP.

| Case | Result |
|---|---|
| `101_PW_upf201_uspp_Fe` | pass |
| `102_PW_DA_davidson` | pass |
| `102_PW_DS_davsubspace` | pass |
| `106_PW_PU_chargemixing` | pass |
| `140_PW_15_SO` | pass |
| `160_PW_DJ_PK_PU_SO` | pass |
| `201_NO_OU_pseudopots` | pass |
| `206_NO_PU_chargemixing` | pass |
| `207_NO_KP_OHS_SPIN4` | pass |
| `240_NO_KP_15_SO` | pass |
| `260_NO_DJ_PK_PU_SO` | pass after refreshing `result.ref` (intended numeric change) |
| `281_NO_KP_HSE` | pass (LibXC+LibRI build) |
| `282_NO_KP_HSE_complex` | pass (LibXC+LibRI build) |
| `381_NO_GO_S1_HSE` | pass (LibXC+LibRI build) |
| `382_NO_GO_S2_HSE` | pass (LibXC+LibRI build) |
| `383_NO_GO_SO_HSE` | pass (LibXC+LibRI build) |

`260_NO_DJ_PK_PU_SO` (LCAO DFT+U SOC, nspin=4):
- old ref: E = -6789.28175035 eV, F = 11.329526, stress = 4894.902744
- new:     E = -6789.28164063 eV, F = 11.327294, stress = 4699.487919
- the new stress tensor (1384.6/-143.3/1139.0/-231.7 kbar) matches the
  accel reference tensor (1385.3/-143.4/1139.9/-231.3 kbar).

## 4. GPU integration tests

Harness: `tests/integrate/Autotest.sh -f CASES_GPU.txt`, 4 MPI x 1 OMP.

| Case | Result |
|---|---|
| `102_PW_CG_GPU` | pass |
| `102_PW_BPCG_GPU` | pass |
| `931_NO_H20_GPU` | pass |
| `934_NO_Si2_dzp_GPU` | pass |
| `934_NO_Si2_dzp_ns2_GPU` | pass |
| `935_NO_Si2_tzdp_ns2_k_GPU` | pass |

`102_PW_DA_davidson_GPU` segfaults inside cuBLAS. The **unmodified LTS
baseline** segfaults in the same place, so this is a pre-existing
environment issue, not a backport regression.

## 5. Single-precision GPU vs double CPU

Custom PW cases derived from `102_PW_CG_GPU` (AsGa, 2 atoms, 8 k-points,
`cal_force=1`, `cal_stress=1`), 1 MPI x 1 OMP. CPU = double; GPU =
`precision single`.

### PBE

| Quantity | CPU double | GPU single | abs error |
|---|---|---|---|
| E (eV) | -4869.7470519303 | -4869.7467581525 | 2.94e-4 (0.147 meV/atom) |
| max force (eV/A) | - | - | 0.0058 |
| max stress (kbar) | - | - | 0.24 |

Baseline LTS GPU-single errors: 0.150 meV/atom, 0.48 kbar. The backport
improves PBE energy and stress.

### SCAN (nspin=1, stress on)

| Quantity | CPU double | GPU single | abs error |
|---|---|---|---|
| E (eV) | -4932.1083705398 | -4932.1080997986 | 2.71e-4 (0.135 meV/atom) |
| max force (eV/A) | - | - | 0.0097 |
| max stress (kbar) | - | - | 0.59 |

Baseline LTS GPU-single errors: 0.139 meV/atom, 0.22 kbar. Energy improves
slightly; the maximum stress error grows from 0.22 to 0.59 kbar
(~0.005% of the ~12,500 kbar stress). This is the known trade-off of the
double-precision k-point tau accumulation (WP3-2).

## 6. Regressions found and fixed during validation

1. **GPU force regression (`102_PW_CG_GPU`/`BPCG_GPU`)**
   - The WP4-1 gcar device-to-host copy received a host pointer in LTS and
     raised `cudaErrorInvalidValue` after SCF.
   - Reverted to `std::vector::assign` in `transfer_gcar()`.
2. **LibXC unit-test link failure**
   - `get_gcar_data`/`get_kvec_c_data` were out-of-line.
   - Moved the trivial specializations inline in `pw_basis_k.h`.
3. **Single-precision GPU SCAN illegal access**
   - Using the device-only `get_gcar_data<Real>()` cache in `grad_wfc`
     was invalid in the stress path.
   - Build small Real-precision host gcar/kvec buffers and upload them.

## 7. Known limitations

- ROCm not tested.
- LTS aborts `sc_mag_switch` in `ReadInput::item_others()` (pre-existing
  guard). A scratch build with that guard disabled ran the FePS3
  nspin=4 SOC+DFT+U DeltaSpin case into the DeltaSpin inner loop and
  produced finite per-atom magnetizations, but was stopped after 50
  minutes without outer-SCF convergence. The guard was not changed in the
  backport.
- WP3-5 (single-pass CUDA float QR) is not backported because LTS has no
  QR kernel.
- WP2-4 (split GGA/mGGA masks) is not backported, so HSE references stay
  unchanged.
- Most WP6 performance items are deferred.

## 8. Non-collinear (nspin=4) coverage, excluding DeltaSpin

The non-DeltaSpin non-collinear fixes are complete for double precision:

- `5968babab`: G=0 magnetization contribution in nspin=4 mixing.
- `359f73459`: Pauli-to-spinor Hamiltonian H(k), spinor->Pauli DMR,
  DFT+U `transfer_vu`, and the H(-R)=H(R)^dagger fill. These apply to
  ordinary LCAO non-collinear/DFT+U calculations; the DeltaSpin lambda
  part is inactive when `sc_mag_switch=0`.
- `5ea2c233c`: genelpa upper-triangle contract.
- `0fc1675c7`, `8efdacb13`, `cb4c38e93`: Davidson MPI datatype,
  near-zero residual handling, Psi offset arithmetic.

Validated on CPU double:
- `140_PW_15_SO` (PW SOC, nspin=4): modified copy with `cg`, converges,
  finite forces/stress.
- `160_PW_DJ_PK_PU_SO` (PW DFT+U SOC): pass.
- `207_NO_KP_OHS_SPIN4`: pass.
- `240_NO_KP_15_SO` (LCAO SOC): pass.
- `260_NO_DJ_PK_PU_SO` (LCAO DFT+U SOC): pass after reference update.
- `383_NO_GO_SO_HSE` (LCAO SOC HSE): pass.

Validated on GPU double:
- A copy of `140_PW_15_SO` with `dav_subspace` gives bit-identical
  energy/force/stress to the unmodified LTS baseline. (CPU and GPU
  `dav_subspace` already differ in the baseline; that is pre-existing.)

Single-precision non-collinear (fixed in `1cb7358ba`):
- The LTS baseline segfaulted after SCF in `cal_force_nl_op<double>`
  because `z_deeq_nc`/`z_qq_so` were null for `precision single`. The
  backport keeps those double buffers valid (host for CPU; device double
  buffers for GPU single).
- CPU and GPU single now complete SCF + force + stress for nspin=4 SOC
  and non-SOC with `cg`, `dav` and `dav_subspace`.
- `dav_subspace`: CPU single vs CPU double energy difference is 0.02 meV
  (`-1678.3522966` vs `-1678.3523159` eV); GPU single gives
  `-1678.3522564` eV.
- `cg`: single precision gives `-1678.36554` (CPU) / `-1678.36549` (GPU)
  eV, about 13 meV from double `cg` (`-1678.35229`). Accel single `cg`
  gives `-1678.36513`, so this is a solver convergence characteristic,
  not a crash or a backport regression.
- Accel's non-collinear `gga_grad` (mapped-density gradient) is a new
  input/feature and is intentionally not backported to LTS.

## 9. LTS vs accel: precision and efficiency

Measured on the same machine, CPU, Release builds, LibXC enabled:

| Case | LTS | accel (255a78048) | Difference |
|---|---|---|---|
| PW PBE energy | -4869.7470519303 eV | -4869.7470519303 eV | < 1e-11 eV |
| PW PBE stress | -12196.09715526 kbar | -12196.09715525 kbar | ~1e-8 kbar |
| PW SCAN energy | -4932.1083705398 eV | -4932.1083705398 eV | < 1e-11 eV |
| LCAO 201 energy | -212.4543800202 eV | -212.4543800202 eV | ~1e-13 eV |
| LCAO 201 force | -0.0284320540 ... | -0.0284320540 ... | identical |
| LCAO DFT+U SOC 260 energy | -6789.28164063 eV | -6789.27589018 eV | 5.9 meV |
| LCAO DFT+U SOC 260 force | 11.327294 | 13.133034 (net vector) | different |
| LCAO DFT+U SOC 260 stress tensor | 1384.6/-143.3/1139.0 | 1385.3/-143.4/1139.9 | ~1 kbar |
| nspin=4 single CPU cg energy | -1678.36554 eV | -1678.36513 eV | 0.4 meV |

Interpretation:
- Common PW (PBE/SCAN) and ordinary LCAO double-precision results are
  numerically identical between this LTS branch and accel.
- LCAO nspin=4 DFT+U/SOC is **not** identical: accel contains a later
  DFT+U refactor and further sign/weight changes that are outside the
  migration plan. The stress tensor already agrees, but the total force
  and energy differ.
- Single-precision non-collinear now works on both; `dav_subspace`
  matches LTS double to 0.02 meV.
- Efficiency on small cases is comparable (LCAO 201: 0.70 s both;
  PW PBE: 2.71 s LTS vs 2.63 s accel for 1 MPI). This is not a
  large-system benchmark. LTS intentionally keeps most WP6 performance
  changes out, while accel contains additional gint/FFT/bspline
  optimizations, so large-system throughput is not guaranteed to match.

## 10. Precision-error analysis and tests/performance comparison

### 10.1 Performance cases

Small cases from `tests/performance`, CPU Release builds, 4 MPI x 1 OMP,
3 repetitions, median wall time (includes startup/IO):

| Case | LTS (s) | accel (s) | LTS/accel |
|---|---|---|---|
| P010_si2_pw | 55.39 | 56.43 | 0.98 |
| P000_si16_pw | 39.15 | 39.29 | 1.00 |
| P109_H2O_lcao | 44.09 | 41.18 | 1.07 |
| P100_si16_lcao | 16.29 | 15.95 | 1.02 |

The PW cases are indistinguishable within run-to-run noise. The two LCAO
cases are 2-7% faster on accel, consistent with the additional gint/LCAO
optimizations that were intentionally left out of LTS. This is not a
large-system benchmark.

### 10.2 Precision on those cases

`etotref` differences (LTS - accel):

| Case | difference |
|---|---|
| P010_si2_pw | 5.7e-14 eV |
| P000_si16_pw | 2.3e-13 eV |
| P109_H2O_lcao | 1.6e-12 eV |
| P100_si16_lcao | 2.7e-12 eV |

Forces (`totalforceref`) are identical in `result.out`; for P100 the
stress tensor is 42.4949158515 kbar (LTS) vs 42.4949158516 kbar (accel),
i.e. ~1e-10 kbar. These are double-precision rounding differences from
different loop/orderings, not physical differences.

### 10.3 LCAO nspin=4 DFT+U/SOC (`260_NO_DJ_PK_PU_SO`)

| Quantity | LTS branch | accel 255a | LTS base ref |
|---|---|---|---|
| E (eV) | -6789.28164063 | -6789.27589018 | -6789.28175035 |
| total force | 11.327294 | 13.133034 (net vector) | 11.329526 |
| stress tensor (kbar) | 1384.6/-143.3/1139.0/-231.7 | 1385.3/-143.4/1139.9/-231.3 | 4894.9 total metric |

Analysis:
- The backport changes this case by only 0.11 meV in energy and 0.002 in
  the force metric; the stress changes substantially because the old
  reference was produced by the pre-fix Pauli/genelpa code.
- The remaining difference to accel is **not** a defect of the backported
  commits: accel contains later work outside this migration plan, notably
  the DFT+U refactor (`fa14bddd5`) and further nspin=4 LCAO/DeltaSpin sign
  and force changes. Those were deliberately not ported.
- The stress tensor already agrees with accel to ~1 kbar, which confirms
  that the Pauli-to-spinor/genelpa fixes captured the main stress issue.

### 10.4 Single-precision nspin=4

After `1cb7358ba` the calculation completes. The remaining difference is
solver/state-selection dependent:

- `dav_subspace`: single vs double energy difference is 0.02 meV
  (CPU: -1678.3522966 vs -1678.3523159 eV); both converge to the same
  low-moment state.
- `cg`: single precision converges to a different magnetic local minimum
  (|M_z| ~ 0.74 uB) than double `cg` (~0.01 uB), giving a 13 meV energy
  difference. Accel single-precision `cg` gives the same higher-moment
  state (-1678.36513 vs LTS -1678.36554 eV), so this is a CG convergence
  characteristic, not an arithmetic error. Use `dav_subspace` when tight
  single-precision non-collinear energies are required.
