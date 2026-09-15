# v1_lcao_mixed_fullaxis — PARTIAL (6/9 axes)

Mixed channel (charge+spin on one atom) LCAO force FD at R7 (ecutwfc 100 /
ecutrho 400 / scf_thr 1e-8), `delta = 0.005 Bohr`, release binary, np4,
MAXJOBS=2, `SCF_NMAX=1200`.  The sweep was stopped after 12 of 18 legs; the base
run and the legs for axes **O-x, O-y, O-z, H1-x, H1-y, H1-z** are archived here.

Base: `E0 = -466.1435161607209920 eV`, `t* = 6.505555755 / 0.1`,
`mu* = -0.2256411084 / -0.0936213988 Ry`,
Sigma pre-compensation z = -0.001633 eV/A, consistency 2.154e-06 eV/A.

| axis | F_FD (eV/A) | F_ana (eV/A) | \|d\| (eV/A) | margin | verdict |
|---|---|---|---|---|---|
| O-x | -0.0000141596 | -0.0000459404 | 3.178e-5 | 404x | PASS |
| O-y | -0.0000132746 | -0.0000452320 | 3.196e-5 | 402x | PASS |
| O-z | -2.8081114004 | -2.8065575071 | 1.554e-3 | 8.3x | PASS |
| H1-x | -2.3221877709 | -2.3210530686 | 1.135e-3 | 11.3x | PASS |
| H1-y | -0.0000001077 | +0.0000226161 | 2.272e-5 | 566x | PASS |
| H1-z | +1.4032772845 | +1.4032787493 | 1.465e-6 | 8776x | PASS |
| H2-x | -- | -- | -- | -- | NOT RUN |
| H2-y | -- | -- | -- | -- | NOT RUN |
| H2-z | -- | -- | -- | -- | NOT RUN |

To finish: re-run the sweep (same TAG overwrites this dir, ~3 h) or top up the
three missing axes with `ONLY=2_0/2_1/2_2` and a separate TAG.
See `docs/superpowers/specs/2026-09-15-mixed-channel-force-fd.md` section 5.
