# DeltaP FD Force Validation (two-group protocol)

## Purpose
Quantify the gap between finite-difference (FD) forces and `cal_force` analytic
forces for `deltap_corr=1`, and isolate the missing analytic terms
(A2 = ∂τ/∂R, B = H_HK force, C = λ·dγ/dR).

Differential object: `E' = !FINAL_ETOT_IS` (already includes `dp_escon`,
see `source/source_estate/fp_energy.cpp` `calculate_etot()`).

## Protocol (T7-b)
1. **Step 0 — base run**: SCF with `deltap_corr=1`, λ re-converged
   (`deltap_lambda_step 0.01`). Extract E'(R0), per-atom λ* (full precision)
   and analytic forces.
2. **Group 1 — frozen λ**: displace each atom ±δ along x/y/z, freeze λ at the
   base λ* (`deltap_lambda_init_file` + `deltap_lambda_step 0.0`), re-converge
   SCF. FD force vs analytic force. Expected residual ≈ A2 + C + B (missing
   terms).
3. **Group 2 — re-converged λ**: same displacements, λ re-converged at each
   geometry (`deltap_lambda_step 0.01`). This is the relax-usable force path.
4. FD force: `F_FD = -(E_{R+δ} - E_{R-δ}) / (2δ)`.
5. Criterion: `|F_FD - F_ana| < 5e-4 Ry/Bohr = 0.0128555 eV/Å`.

## Prerequisites
- ABACUS binary: default `/root/abacus-develop/build/abacus_basic_para`,
  override with `ABACUS=...`
- Pseudopotentials/orbitals at `/root/pporb/apns-*` (paths in INPUT)
- python3 (STRU parsing/displacement, extraction, comparison)
- MPI optional (`run_fd.sh <system> <delta_bohr> <nproc> <group>`)

## Usage
```bash
cd tests/deltap_fd_force
bash run_fd.sh h2o1 0.005 1 both     # full two-group matrix (18+18 runs)
bash run_fd.sh h2o1 0.005 1 1        # group 1 only (frozen λ)
bash run_fd.sh h2o1 0.005 1 2        # group 2 only (λ re-converged)
```
δ default 0.005 Bohr = 0.002646 Å. STRU may be `Cartesian_angstrom` (h2o1) or
`Direct` (lattice-matrix conversion handled inside the script).

## Systems
- `h2o1/` — single H₂O (O 2s2p1d / H 2s1p), 15.873 Å cubic box, 1×1×2 Gamma
  (2 k), `deltap_gdir=3`, `deltap_target_file` 3×0 (all λ active).
  Replaces the old 4-molecule 10 Å box (2×2×2 showed γ-branch jumps / SCF
  non-convergence, 64-k grid too costly).
- `h2o/`, `bn/` — legacy layouts kept for reference.

## Status (T7-b conclusions)
- **FAIL**: group 1 residual 0.87–5.33 eV/Å (68–415× criterion), group 2 worse
  (γ ≫ t, constraint not active, plus γ-branch swapping for H1/H2).
- Root cause isolation (O1-z):
  - **B (H_HK analytic force missing) ≈ 2.6 eV/Å — dominant. Must be
    implemented (guide F6 upgraded from "not implemented" to "required").**
  - τ-unit inconsistency: code uses lat0-unit Cartesian position, spec says
    fractional coordinates (B-6, BLOCKER, user decision pending).
  - A2 (∂τ/∂R) + C (λ·dγ/dR) ≈ 0.26 eV/Å remaining after fraction-τ + H_HK off.
- FD itself is trustworthy: pure-SCF control agrees to 0.8%; δ-linearity
  0.04%; λ×10 scales slope ×10.8.
- Details: `docs/superpowers/specs/2026-08-02-deltap-force-stress-t7b.md`.

## Notes
- `deltap_lambda_init_file`: per-atom λ initial values (one per line, `nat`
  lines), overrides scalar `deltap_lambda_init`. Used by group 1 to freeze λ.
- Output dirs (`base/`, `disp_*`) and `lambda_star.dat` are gitignored.
