# Real-space weight constraint (phase 1) — PW H2O integration smoke

Single-point PW SCF of H2O (15 Å box) with one charge constraint on the O
fragment: `{"targets": [0.1], "atoms": [[0]]}` in delta mode.

Run (serial):
```
OMP_NUM_THREADS=1 /root/abacus-develop/build/abacus_basic_para
```

Expected (see `OUT.constraint_h2o/running_scf.log`):
- Reference SCF (mu = 0) converges, Q_ref(O) = 6.2555 e.
- Constrained phase: mu -> -0.1765 Ry, outer step 6 CONVERGED
  (|Q - target| = 3.06e-5 < constraint_thr = 1e-4).
- Audit lines `CONSTRAINT_AUDIT nconstraint=1 e_con=... total_charge=...
  nelec=8 maxdev=2.2e-16`.
- Final: `[constraint] final status: CONVERGED`.

V1 sum rule variant: use `{"targets": [0, 0, 0], "atoms": [[0], [1], [2]]}`
(delta = 0) — converges at the reference with `total_charge=8 nelec=8`
and per-atom charges O=6.2555, H=0.87227 each.

V3 scan variants (delta = +-0.05/0.1/0.2/0.3): all CONVERGED, no mu capping
(|mu*| <= 0.56 Ry with constraint_mu_max = 5.0 Ry).  An unreachable target
(delta = +5.0 e) with constraint_mu_max = 0.5 Ry fuses to UNREACHABLE and
reports the Q(mu) endpoint.
