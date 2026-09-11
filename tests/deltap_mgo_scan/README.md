# I-1: MgO bulk wide-charge scan (R4 verdict experiment)

Plan: `docs/superpowers/plans/2026-09-09-p0-cases-feomgo.md` (case I-1).

**Question (single, decisive)**: is the linear reachable domain of the charge
constraint on an ionic solid significantly wider than the +-0.3 e window
measured on H2O?  A negative answer is equally important: it would force the
design document's applicability claim ("DeltaQ is legitimate for ionic /
defect systems") to be tightened.

## System and settings

- MgO rocksalt, 8-atom conventional cell, a = 4.211 Angstrom (experimental
  room-temperature lattice constant, `LATTICE_CONSTANT 1.8897259886` Bohr
  = 1 A so the vectors read in Angstrom);
- LCAO, `Mg_gga_8au_100Ry_4s2p1d.orb` + `O_gga_7au_60Ry_2s2p1d.orb`,
  ONCV PBE pseudopotentials (`Mg_ONCV_PBE-1.0.upf`, `O_ONCV_PBE-1.0.upf`),
  all in `tests/PP_ORB`;
- atom indices: **Mg = 0..3** (fcc sites), **O = 4..7** (the other sublattice);
- 2x2x2 Monkhorst-Pack, `symmetry 0` (the constraint potential is not
  symmetry-adapted; the H2O constraint cases also run symmetry 0), KPAR=1,
  `scf_thr 1e-8`, `ks_solver genelpa`, Gaussian smearing 0.002 Ry;
- `nelec = 4*10 (Mg, 2s2p semicore) + 4*6 (O) = 64`.

## Steps

| step | content | criterion |
|---|---|---|
| S1 | grid ladder, delta = 0 (reference phase = the mu = 0 observation) | Q_ref(O) stable to < 3e-5 e (= `constraint_thr`/3) |
| S1X | ecutwfc/ecutrho separation | LCAO forces ecutrho = 4*ecutwfc, so the two knobs are not independent |
| S2 | reference at the calibrated grid, full coverage (O + Mg sublattices) | converged; Q_ref(O)/Q_ref(Mg) in the literature band; `total_charge == nelec`, maxdev ~ 1e-16 |
| S3 | single-O scan +-0.3/0.5/0.8/1.0 e | all points CONVERGED, no fuse; mu(delta) monotone; record the linear window |
| S4 | single-Mg scan +-1.0 e | Mg(0) side reachable or not; Mg(3+) side expected UNREACHABLE (physical fuse case) |
| S5 | verdict vs the H2O kappa ~ 1.7 e/Ry, +-0.3 e window | is the ionic linear window >= 2x the H2O one? |

## Usage

```
bash run_mgo_scan.sh S1 [nproc]      # grid ladder (S1_GRIDS="60:240 80:320" to override)
bash run_mgo_scan.sh S2 [nproc]      # reference run (GRID_ECUT/GRID_RHO to override)
bash run_mgo_scan.sh S3 [nproc]      # single-O scan (S3_CHAINS to split the two sides)
bash run_mgo_scan.sh S4 [nproc]      # single-Mg scan (DELTAS_MG)
bash run_mgo_scan.sh S5 [nproc]      # summary table + slopes
```

Work directories live under `/tmp/mgo_scan/<name>`; the ABACUS logs are copied
to `results/<name>.log`.  `ABACUS=...` overrides the binary (default
`build/abacus_basic_para`, the debug build matched to every prior constraint
evidence item).

Useful overrides: `WORKROOT`, `GRID_ECUT`/`GRID_RHO`, `SCF_NMAX` (default 800:
the outer loop needs one SCF re-convergence per `step_max = 0.05 Ry` mu step,
so the historical 200 is far too low for large |delta|), `DELTAS`, `S3_CHAINS`
and `DELTAS_MG`.

## Calibrated settings (2026-09-11)

- **Production grid: ecutwfc 60 / ecutrho 240** (LCAO forces ecutrho = 4 x
  ecutwfc, so the S1 ladder is a single grid-density knob).  The Q_ref(O)
  ladder oscillates at the ~3e-4 e level up to 160/640 and 200/800 died
  (memory), so the planned < 3e-5 e absolute stability criterion is not
  attainable; the grid's Q_ref offset (1.6e-3 e over the whole ladder, i.e.
  4e-4 e/atom) is 3 orders below the +-0.3..1.0 e signal.
- **Binary**: `build_rel/abacus_basic_para` (-O3) after proving it reproduces
  the debug binary's Q_ref / maxdev / M3b bit-for-bit at both 60/240 and
  160/640.  Speedup only ~1.3x at 160/640; the coarse grid is the real lever
  (11 s vs 57 s per free SCF).
- **Cost driver**: `MuSolverParams::step_max = 0.05 Ry` (hard-coded) caps the
  outer-loop mu step, so a point needs |mu*|/0.05 outer steps, each costing one
  SCF re-convergence (~15-20 iterations here).

## Notes / caveats

## Results (2026-09-11) — verdict: PASS

Single-O charge constraint is linear within 4.0% down to **−1.0 e** and within
12.2% up to **+0.8 e** (only +0.8 → +1.0 e exceeds the 20% band, at +23.0%);
i.e. a linear window of at least **±0.8 e ≈ 2.7x the H2O ±0.3 e window**.
kappa = |dmu/ddelta| is 1.90-2.53 Ry/e on O (H2O: 1.7) — the ionic system is
*stiffer*, not softer: the wider window comes from the energy surface staying
near-quadratic, not from a softer response.  Mg: +1.0 e -> 1.305 Ry/e,
−1.0 e -> 2.115 Ry/e, both reachable (the predicted Mg(3+) fuse did **not**
happen).  Reference state: sum rule `total_charge = 64 = nelec` bit-exact,
maxdev 6.7e-16, Becke Mg +1.4427 / O −1.4427 e (literature band ±1.0-1.5 e).
Independent checks: mu*(+0.8 e) agrees to 0.018% between the 60/240 and 80/320
grids, and mu*(midpoint) = −dE_tot/ddelta closes to <0.8% on all eight deltas.

Full write-up: `docs/superpowers/specs/2026-09-11-i1-mgo-charge-scan.md`;
extracted audit table in `results/summary.txt`, raw logs in `results/logs/`.

- The Becke partition is a **definition-level** reading: MgO literature
  Hirshfeld/Becke charges live in a +-1.0..1.5 e band.  S2 compares inside
  that band, it is not an exact-value comparison.
- Periodic images contribute to the Becke weights (M1 sums them); S2's
  full-coverage sum rule is the first bulk test of that partition.
