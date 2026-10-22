# accel -> LTS backport sources

Branch: `lts-backport-accel-2026Q4`
LTS base: `725bf475a`
accel reference: `zdy/accel = 255a78048` (2026-10-21)

Each row is one LTS commit. "Adaptation" lists the LTS-specific changes
made because the accel tree has diverged (renames/refactors/new APIs).

## WP1 - DeltaSpin / nspin=4

| LTS commit | accel source | Adaptation |
|---|---|---|
| `3f261db88` | `b72b281eb` | LTS memory API (`setmem_complex_op` with ctx). |
| `5968babab` | `a2b8bc30f` (not in accel history) | Manual: three `ig0 > 0` -> `ig0 >= 0` in `charge_mixing_residual.cpp`. |
| `5ea2c233c` | `d52cdf022` | Use LTS `Cpzhemm` wrapper; add tests with LTS `module_hsolver/genelpa/elpa_solver.h` paths. |
| `359f73459` | `1457e26fe`, `698e18762` | H construction lives in `module_gint/gint_k_pvpr.cpp`; final accel conventions used (dftu_pw unchanged, My positive, dspin force 2x unconditional). |
| `d4d0b7986` | `2cff4f563`, `457d9dd4a`, `870ce1589` | CPU/CUDA/ROCm force+stress; accel 255a has the same final sign. |
| `1bfa4cda3` | `8801a66ee` | LTS keeps `onsite_projector.cpp`; add `npol`/`isk` to the DeltaSpin force kernel and call site; port CPU/CUDA/ROCm. |

## WP2 - XC

| LTS commit | accel source | Adaptation |
|---|---|---|
| `881243914` | `0cab77066`, `3b78838b2` | Applied to `v_xc_meta` in `xc_functional_libxc_vxc.cpp`. |
| `8577fc5c9`, `10c259b61` | `98282bfe0` | Applied to `xc_functional_gradcorr.cpp::grad_wfc`. LTS adaptation: build small Real-precision host copies of gcar/kvec_c and upload via TensorMap; the device-only `get_gcar_data<Real>()` caches are not guaranteed populated in the stress path and caused an illegal access in single-precision GPU SCAN. |
| `f056f01e8` | `4bc459bd5` / part of `255a78048` | New overloads in LTS `xc_functional_libxc_wrapper_{gcxc,tauxc}.cpp`; LTS signatures have no hybrid_alpha/hse_omega. |

`0e1a2cf1d` (split GGA/mGGA threshold masks, WP2-4) is **not** applied:
it is an optional numeric change requiring refreshed HSE references.

## WP3 - single-precision GPU

| LTS commit | accel source | Adaptation |
|---|---|---|
| `a5bdd4961` | `40d45f438`, `693ed652a`, `c7e3e899d` | LTS file names `stress_func_cc.cpp` / `stress_func_mgga.cpp`. |
| `e5d4bf84a` | `edd01d71a` | LTS file names `stress_func_{ewa,loc,nl}.cpp`; LTS memory API. |
| `9db3ac422` | `7d6958fb8` | LTS memory API; enabled only for `!double_grid && !use_uspp` (see NUMERIC_DIFF). Validation shows improved PBE stress but a larger SCAN stress deviation; retained to match accel behaviour and because energy/tau are more accurate. |
| `f5f86a5a0` | `6c102559c` | LTS path `module_hsolver/kernels/cuda/math_kernel_op.cu`; `cublasErrcheck`. |
| `4b584d86d`, `e135fb721` | `146ebdc4b` | LTS uses `dnevx_op` instead of `ct::kernels::lapack_heevx`; host round trip for the small reduced matrix; type dispatch for real/complex T. |
| `6d1350db8`, `6333810bb` | `4cc4a9769` | Force/stress/xc float GPU instantiations and FPTYPE lambda; onsite-projector float GPU instantiations deferred because that LTS code is not float-safe. |

`0232ccfe5` (WP3-5, single-pass CUDA float QR) is **not** applied: LTS has
no `lapack_geqrf_inplace`/QR kernel. Implementing it would require porting a
new container QR kernel, which is outside this backport.

## WP4 - GPU stability / large systems

| LTS commit | accel source | Adaptation |
|---|---|---|
| `7a1c45edd` | `2e7a4a32f` (reverted in LTS form) | Reverted: the LTS force path passes the host `gcar` pointer to `transfer_gcar()`, so a device-to-host copy raises `cudaErrorInvalidValue`. Kept `std::vector::assign`. |
| `0fc1675c7` | `9c761d854` | LTS lacked `Parallel_Reduce::MPI_Type`; add the trait without changing existing reductions. |
| `8efdacb13` | `c9f77f8de` | LTS keeps `normalize_op` in `module_hsolver/kernels/math_kernel_op.{cpp,cu,hip.cu}`. |
| `cb4c38e93` | `fc16d9523` (not in accel history) | Manual size_t offset arithmetic; accel has since refactored Psi for paging. |
| `14f9cfe0e` | `e2843cc37` (not in accel history) | Manual host CPU FFT backends in LTS `FFT_Bundle::initfft`; accel has refactored FFT_Bundle. |

## WP5 - general fixes

| LTS commit | accel source | Adaptation |
|---|---|---|
| `fdeae139e` | `f26d7662a`, `64d11d0d7`, `5517a5849`, `33cedb457` | LTS names `ekinetic_new.cpp`/`overlap_new.cpp`; only remaining serial-wrapper site is `write_dmr.cpp`. |

Not applied from WP5: `6dfa7b5a0` (FIRE), `5d79cc5e8` (BFGS),
`4ff7cff6b` (no-plane-wave warning; LTS has no matching change). They are
independent and can be backported separately.

## WP6 - optional performance

`5d2582d72` (count_pw_st OpenMP) is applied (commit after the WP5 batch).
Not applied in this batch: `d05769ab7`, `f4af81009`, `b5ff54cbe`,
`e943b0f9f`, `84ca04b22`, `078df41e8`, `f4548f424`.
These need benchmark/regression gating.

## Validation performed

See `backport/VALIDATION.md` for the full matrix, numbers and
limitations. Summary:

Builds:
- CPU (`build_nspin4`, LCAO+ELPA, tests ON): pass.
- CPU full (`ENABLE_LIBXC=ON`, `ENABLE_LIBRI=ON`): pass, including a clean
  build from a fresh source archive.
- CUDA (`build_gpu`, `ENABLE_LIBXC=ON`): pass.
- CUDA single (`ENABLE_FLOAT_FFTW=ON`): pass.

Unit tests:
- `HSolver_dav`, `HSolver_dav_float`, `HSolver_LCAO` (includes the new
  genelpa upper-triangle tests): pass.
- `hcontainer_test`, `hcontainer_complex_test`, `hcontainer_output_test`,
  `hcontainer_readCSR_test`: pass.
- `pw_test`, `basis_pw_serial`, `basis_pw_k_serial`: pass.
- `XCTest_PBE`, `XCTest_HSE`, `XCTest_PZ_SPN`, `XCTest_GRADCORR`,
  `XCTest_SCAN`, `XCTest_VXC`, `XC_Functional_UTs`: pass (LibXC build).

CPU integration (4 MPI x 1 OMP, tests/integrate):
- `101_PW_upf201_uspp_Fe`, `102_PW_DA_davidson`,
  `102_PW_DS_davsubspace`, `106_PW_PU_chargemixing`, `140_PW_15_SO`,
  `160_PW_DJ_PK_PU_SO`, `201_NO_OU_pseudopots`, `206_NO_PU_chargemixing`,
  `207_NO_KP_OHS_SPIN4`, `240_NO_KP_15_SO`, `260_NO_DJ_PK_PU_SO`: pass
  (260 reference refreshed, see NUMERIC_DIFF).
- HSE `281`, `282`, `381`, `382`, `383`: pass with the LibXC+LibRI build.

GPU integration (4 MPI x 1 OMP, CUDA):
- `102_PW_CG_GPU`, `102_PW_BPCG_GPU`, `931_NO_H20_GPU`,
  `934_NO_Si2_dzp_GPU`, `934_NO_Si2_dzp_ns2_GPU`,
  `935_NO_Si2_tzdp_ns2_k_GPU`: pass.
- `102_PW_DA_davidson_GPU` segfaults in cuBLAS in the **unmodified LTS
  baseline** as well; not a regression from this backport.

Single precision GPU vs double CPU (custom cases, 1 MPI x 1 OMP):
- PBE: dE = 0.147 meV/atom, max force error ~0.006 eV/A,
  max stress error ~0.24 kbar. Baseline LTS errors were 0.150 meV/atom
  and ~0.48 kbar, so energy/stress improve.
- SCAN (nspin=1, cal_stress): dE = 0.135 meV/atom (baseline 0.139),
  max force error ~0.0097 eV/A, max stress error ~0.59 kbar
  (baseline ~0.22 kbar). The larger SCAN stress deviation comes from the
  double-precision k-point tau accumulation (WP3-2); it is ~0.005% of the
  ~12,500 kbar stress and is recorded here as a known trade-off.

Limitations:
- ROCm was not built.
- LTS aborts `sc_mag_switch` in the input checker, so a full DeltaSpin
  integration test cannot run without removing that guard. A smoke run
  with the guard disabled in a scratch build reached the DeltaSpin inner
  loop and produced finite per-atom magnetizations, but was stopped after
  50 minutes without outer-SCF convergence.
- Single precision requires `-DENABLE_FLOAT_FFTW=ON`; without it even the
  unmodified LTS baseline fails at START CHARGE.
