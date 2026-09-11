# II-1: FeO spin constraint (TM d-moment capability + DFT+U coexistence)

Plan: `docs/superpowers/plans/2026-09-09-p0-cases-feomgo.md` (case II-1).

**Verification questions**

1. can the constraint framework drive a transition-metal d local moment at
   production quality?
2. is DFT+U + constraint simultaneous activation compatible?  The two live in
   independent potential channels (the on-site DFT+U projection and the
   veff-grid constraint injection) and had never been switched on together.
3. (deferred, II-1b) quantitative mu-vs-DeltaSpin-lambda attribution.

## System: the plan's basename is wrong

The plan inherits `tests/17_DS_DFTU/11_PW_DFTU_S2_FeO`.  That case's `STRU`
contains `TOTAL ATOM NUMBER = 2` with species `Fe` only: it is the **Fe
sublattice** of the rocksalt cell (Fe at (0,0,0) and (1/2,1/2,1/2) — exactly
the Fe positions of case 50), not FeO, despite the directory name.  The real
FeO case is `tests/17_DS_DFTU/50_FeO_O_first_Fe_second`, which this scan
inherits.

## Settings

- FeO rocksalt primitive cell, 2 O + 2 Fe, a = 8.190 Bohr (the inherited
  baseline cell); `LATTICE_CONSTANT 8.190`;
- **atom indexing: 0,1 = O; 2,3 = Fe** (Fe 2 carries `mag +2`, Fe 3 `mag -2`,
  i.e. the reference state is AFM);
- PW, `dav_subspace`, nspin 2, `symmetry 0`, Gamma-only (1x1x1) as in the
  inherited case, ecutwfc 50, Gaussian smearing 0.01 Ry;
- DFT+U: `orbital_corr -1 2`, `hubbard_u 0 5.0`, `onsite_radius 3.0` — O
  carries no U;
- the reference state converges to Fe moments +-3.485 uB
  (`atomic mag: 2 3.48496887` / `atomic mag: 3 -3.48496515`), a textbook
  DFT+U FeO high-spin AFM;
- constraint: `type spin`, `weight becke`, `target_mode delta`,
  `mu_max 5.0`, `thr 1e-4`, single atom index 2.

## Steps

| step | content | criterion |
|---|---|---|
| S0B | unconstrained FeO, `scf_thr 1e-8`, `out_chg 1` | warm-start donor: `out_chg 1` is what makes ABACUS also write the DFT+U `onsite.dm`, without which `init_chg file` aborts in `Plus_U::read_occup_m` |
| S1 | baseline reproduction at the harness threshold (`scf_thr 1e-6`) | matches `50_FeO_O_first_Fe_second/result.ref` |
| S1X | threshold ladder: baseline at 1e-8; constraint on at 1e-8; constraint on at 1e-7; constraint on at 1e-7 + `mixing_beta 0.2` | does the *constrained* SCF reach `drho < scf_thr` at all, and is the mu reading mixing-independent? |
| S2 | reference phase + first constrained point, delta = +0.1 uB | CONVERGED, outer steps <= 10, |mu| < 1 Ry |
| S2R | delta = 0 no-op check | the reference phase already satisfies the target |
| S3 | six-point scan +-0.1 / +-0.3 / +-0.5 uB on Fe atom 2 | all points CONVERGED, no branch switch |
| S4 | fuse case delta = +3.0 uB | UNREACHABLE at the mu cap, no divergence |
| S5 | DeltaSpin control (case 12 settings) at the state matched to S2 | mu vs lambda sign/magnitude |
| S6 | summary table | mu(delta), kappa, on-site moment response |

## Usage

```
bash run_feo_spin_scan.sh S0B [nproc]     # warm-start donor (out_chg 1)
RESTART=/tmp/feo_spin/S0B_warmstart/OUT.autotest \
  bash run_feo_spin_scan.sh S3 [nproc]    # hot-started six-point scan
bash run_feo_spin_scan.sh S1X [nproc]     # threshold + mixing ladder
bash run_feo_spin_scan.sh S5  [nproc]     # DeltaSpin control
bash run_feo_spin_scan.sh S6              # table from the work tree
```

Work directories live under `/tmp/feo_spin` (`WORKROOT` overrides); the ABACUS
binary defaults to `build_rel/abacus_basic_para` (`ABACUS` overrides).
Useful overrides: `SCF_THR`, `SCF_NMAX`, `MIXB`, `MIXB=0.2`, `DELTAS`.

## Re-anchor round (2026-09-11, second entry) -- S1T triage + outer-step caps

The II-1a baseline above was shown to be the *metastable* solution, so this
round first settles which solution the scan must be anchored on, then removes
the framework defect that II-1a found.

**S1T triage** (`run_feo_baseline_triage.sh`, records in `results/triage/`):
nine unconstrained SCFs, all converged.  The SCF basin is selected by the STRU
`mag` guess: `mag 2.0/1.0/0.2` cold start -> the harness state
(Fe +-3.4850 uB, -7652.3958757 eV), `mag 4.0` cold start -> the LOWER state
(Fe +-3.7149 uB, -7653.0079658 eV).  Both basins are stationary under warm
re-entry.  Neither is k converged: 2x2x2 collapses both guesses onto
+-1.477 uB / -7655.1237 eV, 4x4x4 gives +-3.097 uB / -7655.6855 eV.  Verdict:
the Gamma-only cell cannot anchor a quantitative kappa scan.

**Framework** (`constraint_step_max`, `constraint_step_probe`): two new INPUT
keys bound the outer step.  `step_probe` applies to the history-free FIRST step
only -- without a measured secant slope that step always sat at the
`step_max` cap, which is the fixed overshoot II-1a diagnosed.  Defaults
(0.05 Ry / 0.0 = "use step_max") reproduce the previous behaviour bit for bit;
`0 <= step_probe <= step_max` is enforced at configure time.

**S3L re-anchored scan** (`run_feo_spin_scan.sh S3L`, records in
`results/s3l/`): anchor = the lower state, `Q_ref = 3.384107449 uB`.
Only +-0.1 uB is reachable: `mu* = -0.0655152 / +0.0594118 Ry`,
`dE = +0.04392 / +0.04139 eV`, against the linear-response prediction
`0.5*|delta|*|mu*| = 0.04456 / 0.04042 eV` (1.4% / 2.4%).  +-0.3 and +-0.5
fail even with `step_max` cut to 1/5: the SCF is *bistable at fixed mu*
(Q oscillates 3.61 <-> 4.08 at mu ~ -0.17 Ry) and on the negative side the
moment collapses to Q ~ 2.07 and will not return even at mu ~ 0.

**Becke vs on-site decoupling**: at that collapsed point the Becke observable
reads Q = 2.1012 uB while the Fe on-site moments are still +3.5180 / -3.6585 uB
(anchor 3.7149).  The constraint moves the Becke-weighted moment without moving
the physical d local moment, which makes the II-1b observable question a
prerequisite for II-1's capability claim rather than a follow-up.

Full write-up: `docs/superpowers/specs/2026-09-11-ii1b-baseline-triage-and-step-cap.md`.

## Protocol findings (2026-09-11)


0. **The inherited baseline's reference state is metastable, and that is the
   root cause of everything below.**  A plainly *unconstrained* FeO SCF
   relaxed from a slightly perturbed density converges (drho 3.4e-9) to a
   second AFM solution at Fe moments +-3.7148 uB and
   E_tot = -7653.0079614903 eV -- **0.612 eV below** the state that
   `50_FeO_O_first_Fe_second/result.ref` records (Fe +-3.4850 uB,
   E = -7652.3958756613 eV).  Any perturbation, including a correct small
   constraint potential, can tip the SCF into the lower solution, and the
   outer loop only tests `|Q - target| < thr`, so it cannot tell "reached the
   target" from "changed state".  Do not use this baseline for constraint
   validation before establishing which solution is the ground state
   (2x2x2 MP, or an explicit multi-guess search in S1).
1. **The constraint needs `scf_thr = 1e-7` here, not 1e-8.**  The unconstrained
   FeO SCF reaches `drho = 8.7e-9` in 18 iterations; the *constrained* SCF at
   `scf_thr 1e-8` never gets below `drho ~ 3e-6` (400 iterations, energy still
   drifting) and therefore never crosses the outer-loop gate.  At 1e-7 the
   same point converges in 8 outer steps / 90 SCF iterations.  The audit
   reading floor tracks the SCF floor (5.8e-5 residual at 1e-7), so the
   observable is SCF-limited, not partition-limited.
   *(`conv_esolver = (drho < scf_thr)` -- `module_charge/chgmixing.cpp`.)*
2. **The mu reading is mixing-independent**: `mixing_beta 0.4` and `0.2` give
   mu* agreeing to 6.2e-5 relative (ad-hoc pair) / 8.3e-4 (scripted pair),
   against a 6.7e-7 spread over three independent `mixing_beta 0.4` runs.
3. **The reference observable is exact and protocol-independent**:
   `Q_ref = 3.137543567 uB` in all six cold-started scan points (nine digits),
   and `3.137560545 uB` when warm started (the 1.7e-5 difference is the restart
   file precision).
3b. **delta = 0 through the full constraint path is a no-op to the SCF noise
   floor**: S2R gives -7652.3958756612619254 eV against the unconstrained
   -7652.3958756612628349 eV (9.1e-7 eV apart), with `res = 0` exactly,
   `mu = 0`, `e_con = 0`, and it converges on the first outer step.
4. **AFM FeO is multistable and the outer loop has no branch guard.**  The SCF
   is free to fall into a different magnetic solution during a mu re-drive, and
   the secant then accepts whatever state satisfies the target.  The clearest
   evidence is the delta = +0.3 point warm started from S0B: its third outer
   step needed 397 SCF iterations (44 -> 441), overshot to Q = 3.534, and
   locked on at mu* = -0.0343861 Ry -- a *smaller* |mu| than the delta = +0.1
   point (mu* = -0.0735952).  A single-valued response would need roughly
   -0.22 Ry there.
5. **The hard-coded `MuSolverParams::step_max = 0.05 Ry` is a fixed first-step
   overshoot.**  With no history the solver falls back to
   `kappa = kappa_min = 0.3`, so the very first step is always the cap.  On
   MgO (I-1) the energy surface is smooth and the overshoot is harmless; on
   FeO, `mu = +0.05 Ry` on Fe 2 already flips the AFM branch, so the entire
   "reduce the moment" side is unreachable.
6. **The Becke-weighted spin observable is not the d local moment.**  In the
   off-branch states the audit reports `q ~ 1.5-1.9` while ABACUS still prints
   `atomic mag: 2 ~ 3.7`.  A constraint written on the Becke weight is not a
   constraint on the local moment, which is exactly the mu-vs-lambda question
   deferred to II-1b.

## Results (2026-09-11)

**Verdict: verification question 2 (DFT+U + constraint coexistence) passes;
question 1 (TM d-moment capability) passes with one trustworthy point;
the plan's assumed scan window does not hold, and the cause is the inherited
baseline, not the constraint framework.**

- S1 baseline reproduction: |dE| = 4.0e-12 eV vs `result.ref` -- PASS.
- S2 / S3 delta=+0.1 uB cold: **CONVERGED**, mu* = -0.0735952 Ry,
  Q: 3.137543567 -> 3.237604506 uB, E_tot = -7652.3447694 eV (0.051 eV *above*
  the reference, i.e. on the reference branch).  Three independent runs agree
  to 6.7e-7 relative.
- S2R delta=0: bit-exact no-op -- PASS.
- Case 11 (Fe sublattice, *not* FeO): constraint + DFT+U same run **CONVERGED**,
  mu* = -0.008107256492 Ry, maxdev = 0.
- Everything else in the six-point scan fails or lands on the metastable
  branch: cold start converges only at +0.1 uB; hot start converges at +0.3
  and +0.5 uB but 0.60/0.32 eV *below* the reference; the whole
  "reduce the moment" side is unreachable because the hard-coded
  `step_max = 0.05 Ry` first step already flips Fe 2.

See `results/summary.txt` and the trimmed audit trails in `results/audit/`.
Full write-up: `docs/superpowers/specs/2026-09-11-ii1-feo-spin-scan.md`.
