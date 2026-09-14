# III-1 — charge-transfer pair + CDFT/Marcus interface (H2O dimer)

Pilot for validation-case III-1 (`2026-09-09-validation-case-suite-design.md` §3):
the framework's flagship physics interface — two *fragment* charge constraints in
one run (v2 target list), used the way CDFT uses them.

## System

`h2o_dimer/` — water dimer with a clean donor/acceptor assignment:

- acceptor fragment = atoms `[0, 2, 3]` = O1 + its two H;
- donor fragment    = atoms `[1, 4, 5]` = O2 + its two H (O2-H...O1 hydrogen bond,
  O1...H = 2.02 A, O1...O2 = 2.98 A);
- LCAO, gamma-only, 15 A cubic box, nelec 20, ecutwfc 60 / ecutrho 240,
  O_gga_7au_60Ry_2s2p1d + H_gga_8au_60Ry_2s1p, O.upf + H_ONCV_PBE-1.0.upf.

Atom indices are global and in STRU block order: the two O first (0, 1), then the
four H (2..5) — which is why the fragments are not contiguous.

## Constraint

v2 target file, both constraints active in the same run, net-zero transfer:

```json
{"constraints": [
  {"type": "charge", "target":   delta, "atoms": [0, 2, 3]},
  {"type": "charge", "target":  -delta, "atoms": [1, 4, 5]}]}
```

`constraint_target_mode delta` => each target is an offset from that fragment's
free reference reading, which the framework measures itself (so no separate
unconstrained run is needed for the reference).

## Usage

```bash
bash run_ct_scan.sh 4                      # default: 0, +-0.05/0.10/0.15/0.20 e
DELTAS="0 0.02 -0.02" bash run_ct_scan.sh 4   # custom grid
```

Runs land in `WORKROOT` (default `/tmp/ct_dimer/<tag>`); each point writes
`results/<tag>.audit` (audit lines + final energy + wall clock) and the runner
finishes with `tools/extract_scan.py`, which regenerates `results/summary.txt`.

`OMP_NUM_THREADS=1` is pinned by the runner (reproducibility discipline, §3.5);
serial single-point wall clock is 20-190 s.

## What to read out of an audit file

```
CONSTRAINT_AUDIT nconstraint=2 ... total_charge=20 nelec=20 maxdev=5.5e-16
CONSTRAINT_AUDIT c[0] kind=charge q=... t=... mu=... res=...     <- acceptor
CONSTRAINT_AUDIT c[1] kind=charge q=... t=... mu=... res=...     <- donor
```

- `q` is the Becke-weighted fragment charge, `t` the (delta-shifted) target,
  `mu` the fragment's Lagrange multiplier in Ry;
- `total_charge == nelec` is the partitioning sum rule — it held to ~6e-16 at
  every scanned point;
- `mu_acc == -mu_don` is the antisymmetry of a net-zero CT pair.
