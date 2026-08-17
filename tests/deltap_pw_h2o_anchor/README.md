# DeltaP PW Gamma anchor (Stage 4.1 anchor rebuild #3)

Production-grade PW (plane-wave basis) anchor for the operator-mode Gamma
accounting.  Regenerates the reference values recorded in
`docs/superpowers/specs/2026-08-17-deltap-anchor3-rebuild.md`.

Setup (serial, single task):
- STRU: H2O in a 15 Bohr box (same geometry as `tests/deltap_pw_h2o`)
- KPT: 1x1x4 Monkhorst-Pack (must be `Monkhorst-Pack`, not `Monkhorst_Pack`)
- INPUT: ecutwfc=80 / ecutrho=320 / scf_thr=1e-8 / scf_nmax=100,
  deltap_switch=true, deltap_inner_thr=1e-3, no target, lambda_init=0

Expected results (build/abacus_basic_para, OMP_NUM_THREADS=1):
- lambda=0:          Gamma/atom = (5.236100, 1.348236, 1.348093)
                     FINAL_ETOT_IS = -466.94573912497 eV
- lambda=-0.01 (frozen, deltap_lambda_step=0 / deltap_lambda_init=-0.01):
                     Gamma/atom = (5.248896, 1.352094, 1.351939)
                     escon = +0.079529 Ry (= -lambda * sum Gamma  identity)
                     FINAL_ETOT_IS = -466.9443817682981 eV

The escon identity check is the PW-side analogue of the LCAO operator-mode
escon identity (T2): escon = -lambda * sum_I Gamma_I.

Run:  OMP_NUM_THREADS=1 /root/abacus-develop/build/abacus_basic_para
in a copy of this directory (the OUT.pw_h2o/ output dir is written in-place).
