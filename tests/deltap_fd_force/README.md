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
- `OMP_NUM_THREADS=1` is exported by `run_fd.sh` (and required for all serial
  screening runs): without it, serial LCAO runs hang in the FFTW OMP thread
  pool (`recip2real` barrier spin) on the current environment.

## Tier-1 systems (correctness verdict systems)

Asymmetric small molecules for decisive FD / branch experiments, built
2026-08-03. Rationale: symmetric H2O (h2o1) has degenerate H1/H2 Berry-phase
branches that swap under ±δ displacements, polluting λ*(R) trajectories;
Tier-1 systems remove the symmetry degeneracy.

| System | Dir | Geometry | Why |
|--------|-----|----------|-----|
| HF | `hf/` | z-aligned, bond 0.9168 Å, centered in 15.873 Å box | asymmetric diatomic, no branch degeneracy, light (cheap FD at ecutwfc=100) |
| CO | `co/` | z-aligned, bond 1.128 Å, same box | asymmetric diatomic, heavier π-space (2nd candidate) |
| H2O asym | `h2o_asym/` | h2o1 with H2 shifted (+0.03 x, +0.05 z Å) | breaks C2v, keeps 3-atom coverage, fallback |

All three: same box/KPT (Gamma 1×1×2, gdir=3 k-string) as h2o1; target.dat
all-zero (overridden per experiment); no INPUT checked in — `run_fd.sh`
generates INPUT (override `ECUTWFC/ECUTRHO/SCF_THR` env for production
settings: 100 / 400 / 1e-8).

### Screening protocol (before a system is accepted as Tier-1)
1. **γ(λ) smoothness**: 3 frozen-λ points (0, ±5e-3 Ry via
   `deltap_lambda_init_file` + `deltap_lambda_step 0.0`); per-atom γ_report
   must be monotonic in λ with no 2π-quantum jumps.
2. **Branch stability**: displace each atom ±δ (0.005 Bohr) along z at frozen
   λ; compare `deltap_branch.dat` / γ continuity — zero branch flips allowed.
3. **Sync convergence**: base run (λ step 0.01, no inner loop) must converge
   within scf_nmax=100 without oscillation.

A system passing all three replaces h2o1 as the verdict system for the
stationary-λ group-2 FD (branch-decomposition plan,
`docs/superpowers/specs/2026-08-03-deltap-branch-decomposition-plan.md`).
