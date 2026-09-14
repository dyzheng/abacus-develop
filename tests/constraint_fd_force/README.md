# Constraint force FD (Task 2.6 / V1)

Finite-difference acceptance of the constraint force: the analytic force of a
constrained SCF must equal the energy derivative, `F_FD = -(E_+ - E_-)/(2 delta)`,
with the **raw** `FINAL_ETOT_IS` as the observable (never `E' = E - mu*t`, which
carries the `t*dmu/dR` envelope pseudo-term; see the 2026-09-07 attribution).

- Criterion: `|F_FD - F_ana| < 0.0128555 eV/A` (5e-4 Ry/Bohr), not waived.
- Grid (R7): `ecutwfc=100`, `ecutrho=400`, `scf_thr=1e-8`; `delta = 0.005 Bohr`.
- Base: constrained SCF at R0 in delta mode; its converged density is reused as
  the restart for every leg, and its `t*` (first audit `t`) is frozen.
- Channels: `charge` (PW `211_PW_constraint_h2o`, LCAO `212_NAO_constraint_h2o`)
  and `spin` (PW `tests/01_PW/212_PW_constraint_h2o_spin`, LCAO carrier
  `cases/212_NAO_constraint_h2o_spin`).

## Run

```sh
# charge channel (default carriers)
CASE=tests/01_PW/211_PW_constraint_h2o TEST_FORCE=1 \
  bash tools/run_constraint_fd.sh pw 0.005 4 3

# spin channel, V1 rerun protocol: fixed-mu + LCAO full axes
ABACUS=/root/abacus-develop/build_rel/abacus_basic_para TEST_FORCE=1 \
  CASE=$PWD/cases/212_NAO_constraint_h2o_spin \
  bash tools/run_constraint_fd.sh lcao 0.005 4 3

# single axis smoke (e.g. O-z), PW + dav_subspace
ONLY=0_2 KS_SOLVER=dav_subspace TEST_FORCE=1 \
  CASE=$PWD/../01_PW/212_PW_constraint_h2o_spin \
  bash tools/run_constraint_fd.sh pw 0.005 4 2
```

Switches: `ONLY=iat_axis` (one axis), `CASE=` (carrier dir), `TEST_FORCE=1`
(per-term force dump, needed by the standard checks), `FIXED_MU=1|0`
(default 1: legs freeze `mu` at the base `mu*` via `ABA_CONSTRAINT_FIXED_MU`;
0 restores the legacy in-leg re-optimization), `KS_SOLVER=` (pin a solver),
`ECUTWFC`/`ECUTRHO`/`SCF_THR`, `RESDIR`/`TAG`/`WORKROOT`.

## Archiving (mandatory for long runs)

The scratch work dir under `WORKROOT` (`/tmp/cfd_<basis>`) is disposable. Every
base/leg **audit, force block, timing and the FD table** is copied into
`RESDIR/<TAG>/` as it completes, where `TAG` defaults to
`<basis>_<case>_<timestamp>`:

| file | content |
|---|---|
| `summary.txt` | run header (case/binary/grid/protocol), `E0`, `t*`, `mu*`, base forces, FD table |
| `base.audit` | audit lines + wall clock of the base run |
| `leg_<iat>_<axis>_<sign>.audit` | same for one leg |
| `legs.tsv` | `iat axis sign E mu` per leg |

This is the fix for the 2026-09-10 / 2026-09-14 V1 batches, which ran 18 legs
into `/tmp` only and lost every leg-level result when the session was cleaned.

## Evidence policy

Only this README, `results/*` (audits, `summary.txt`, `legs.tsv`) and the case
carriers are committed -- never full SCF output. `results/` keeps one subdir per
accepted run; a superseded run is dropped in the follow-up commit.
