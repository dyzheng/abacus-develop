# Numeric differences introduced by the accel backport

Only changes that can alter results are listed. Purely compile-time,
memory-safety or performance changes are marked "none".

## Result-changing fixes

| LTS commit | Scope | Effect |
|---|---|---|
| `5968babab` | nspin=4 charge mixing | The G=0 magnetization term is now included on the rank that owns `ig_gge0 == 0`. Previously that rank silently dropped it. |
| `5ea2c233c` | LCAO genelpa | `generalized_eigenvector()` consumes only the authoritative upper triangle of the Hermitian H. Results change only when the lower triangle was stale; restores spin-rotation invariance. |
| `359f73459` | LCAO nspin=4 H / DFT+U / DeltaSpin | Fixes sigma_y signs in H construction, transfer_vu and lambda, uses H(-R)=H(R)^dagger, fixes rho_y in the spinor->Pauli DM path, and removes the spurious DFT+U nspin=4 VU/2 factor and force factor 2. |
| `d4d0b7986` | PW nspin=4 DeltaSpin force/stress | Swaps the imaginary part of the up-down/down-up Pauli coefficients (`lambda_x -/+ i lambda_y`). Y-component force/stress sign changes. |
| `1bfa4cda3` | PW nspin=2 DeltaSpin output/force | Correct becp stride and spin channel for npol=1, spin_sign in the force kernel, and per-atom labels in Magnetism/Magnetic force tables. |
| `881243914` | mGGA XC (nspin=1/2) | Removes the `grad(rho)` threshold from the mGGA mask; grid points with `grad(rho)=0` now contribute to etxc/vrho/vtau. Expected to be a tiny correction for SCAN/R2SCAN. |
| `8efdacb13` | Davidson SchmidtOrth / DavSubspace normalize | Degenerate (near-zero-norm) directions are skipped instead of aborting. In practice this removes aborts; if such a direction was previously renormalized, the basis may differ. |

## Test references

The LTS integration test data is not tracked in this repository (it is
shipped via the test archive). Cases whose references are expected to move
if regenerated:

- `01_PW/099_PW_DJ_SO` (dftu_pw convention is intentionally unchanged in
  this backport, but the H construction and genelpa changes can affect it)
- `03_NAO_multik/scf_angle_spin4`
- `03_NAO_multik/scf_out_dos_spin4`
- `03_NAO_multik/scf_u_spin4`
- `tests/integrate/260_NO_DJ_PK_PU_SO/result.ref` **updated** in this
  branch: etot -6789.28164063, force 11.327294, stress 4699.487919. The
  force is unchanged to ~0.002 and the stress tensor now agrees with the
  accel reference; the old 4894.90 reference came from the pre-fix code.
- `17_DS_DFTU/14,15,16,19,21,41` (nspin=4 DeltaSpin sign/convention)
- `17_DS_DFTU/12,18` (nspin=2 DeltaSpin output/force)
- `tests/08_EXX/01-06` only if WP2-4 is enabled later (not in this batch)

## Precision-only changes (no algorithmic change)

- `a5bdd4961`, `e5d4bf84a`: double accumulators for stress.
- `9db3ac422`: double host accumulation of k-point rho/tau on GPU single.
  LTS limitation: enabled only for the norm-conserving path
  (`!double_grid && !use_uspp`); USPP keeps the previous behaviour until
  the accel USPP/charge-density refactor is ported.
- `f5f86a5a0`: blocked single-precision CGEMM for k > 2048.
- `4b584d86d`, `e135fb721`: GPU-single reduced Davidson solve in double.
- `6d1350db8`: float GPU instantiations / FPTYPE lambda.

## No numeric impact

- `3f261db88` Davidson refresh (restart only; converged runs unchanged).
- `8577fc5c9` grad_wfc tensor accessor.
- `f056f01e8` LibXC per-thread init cache.
- `c78ba665d`, `0fc1675c7`, `cb4c38e93`, `14f9cfe0e` memory/type/FFT
  backend fixes.
- `fdeae139e` general fixes (except the HContainer preservation, which
  fixes previously-zero HR output).
