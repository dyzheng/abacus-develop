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
