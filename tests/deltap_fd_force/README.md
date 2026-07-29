# DeltaP FD Force Validation Test

## Purpose
Validate C-02 force/stress corrections by comparing finite-difference (FD)
forces with cal_force analytic forces under frozen constraint lambda.

## Protocol
1. Run base SCF + deltap_corr with fixed lambda
2. For each atom, displace ±δ along gdir, re-converge SCF with same lambda
3. Compute FD force: F_FD = -(E_{+δ} - E_{-δ}) / (2δ)
4. Compare with cal_force output from base run

## Prerequisites
- ABACUS binary (abacus_basic_para) in PATH or set ABACUS=
- Pseudopotentials at /root/pporb/apns-pseudopotentials-v1
- Orbitals at /root/pporb/apns-orbitals-efficiency-v1
- MPI (optional) and python3

## Usage
```bash
cd tests/deltap_fd_force
bash run_fd.sh h2o 0.005 1
```

## Acceptance Criteria
| Check | Criteria |
|-------|----------|
| H2O forces | |F_FD - F_analytic| / |F_FD| < 5% for all atoms |
| λ frozen | dp_escon consistent across displacements (±0.1 μRy) |
| Stress | Output contains finite stress (no NaN) |

## Notes
- The ∂τ/∂R term and H_HK force are not yet implemented.
  If FD-analyic discrepancy > 5%, it indicates these terms are
  non-negligible and need implementation.
- This test isolates only the ΔP constraint force (force_deltap),
  not the total force.  Run with and without deltap_corr=1
  to quantify the ΔP-specific contribution.
