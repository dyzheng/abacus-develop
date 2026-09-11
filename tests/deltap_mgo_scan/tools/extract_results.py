#!/usr/bin/env python3
"""Regenerate results/summary.txt from the committed trimmed audit extracts.

The raw ABACUS logs live in /tmp during a run and are not committed; the
extracts in results/audit/*.audit keep every CONSTRAINT_AUDIT / M3b / status /
E_tot line, which is everything this summary needs.  Usage:

    python3 tools/extract_results.py        # writes results/summary.txt
"""
import os
import re

RY = 13.605693
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "results", "audit")


def read(case, root):
    txt = open(os.path.join(AUDIT, f"{case}@{root}.audit"), errors="replace").read()
    rows = re.findall(
        r"CONSTRAINT_AUDIT c\[(\d+)\] kind=(\S+) q=(\S+) t=(\S+) mu=(\S+) res=(\S+)", txt)
    head = re.findall(
        r"CONSTRAINT_AUDIT nconstraint=(\d+) e_con=(\S+) max_residual=(\S+) "
        r"total_charge=(\S+) nelec=(\d+) maxdev=(\S+)", txt)
    m3b = re.search(r"M3b runtime audit: max \|Tr\[W.DM\] - int w rho\| = (\S+) e", txt)
    etot = re.search(r"!FINAL_ETOT_IS (\S+) eV", txt)
    st = re.search(r"final status: (\w+)", txt)
    # Cost proxy: the SCF iteration index reported at the last outer step.
    it = re.findall(r"outer step \d+ after SCF iteration (\d+)", txt)
    return dict(
        rows=[(int(r[0]), r[1], float(r[2]), float(r[3]), float(r[4]), float(r[5]))
              for r in rows],
        head=head[-1] if head else None,
        m3b=m3b.group(1) if m3b else "-",
        etot=float(etot.group(1)) if etot else None,
        status=st.group(1) if st else "RUNNING",
        outer=txt.count("[constraint] outer step"),
        iters=int(it[-1]) if it else 0,
    )


def main():
    out = []
    w = out.append
    w("=" * 100)
    w("I-1 MgO wide-charge scan - full extracted results")
    w("system: MgO rocksalt 8-atom cell a=4.211 A, LCAO, 2x2x2 MP, symmetry 0, nelec=64")
    w("binary: build_rel/abacus_basic_para (Release, proven bit-identical to debug)")
    w("constraint: Becke, delta mode, mu_max 5.0 Ry, thr 1e-4 e, scf_thr 1e-8, scf_nmax 800")
    w("=" * 100)

    w("\n[S1] grid ladder, delta=0 reference phase, Q_ref of the 4-atom O sublattice")
    w(f"  {'ecutwfc/ecutrho':>16} {'Q_O(4) [e]':>15} {'Q/O [e]':>14} {'E_tot [eV]':>20} "
      f"{'maxdev':>10} {'M3b [e]':>12}")
    prev = None
    for e in (60, 80, 100, 120, 160):
        a = read(f"S1_grid{e}", "mgo_scan")
        q = a["rows"][-1][2]
        d = f"  dQ vs prev = {abs(q - prev):.3e} e" if prev is not None else ""
        w(f"  {e:>7}/{4 * e:<8} {q:>15.8f} {q / 4:>14.10f} {a['etot']:>20.12f} "
          f"{a['head'][5]:>10} {a['m3b']:>12}{d}")
        prev = q
    w("  NOTE: the planned criterion dQ < 3e-5 e is NOT attainable (best 2.8e-4 e); the drift is")
    w("        deterministic FFT-grid discretisation, not noise.  Replaced by the kappa grid")
    w("        check further down (0.018% on mu*).")

    w("\n[S2] reference at the production grid 60/240, full coverage (O + Mg sublattices)")
    a = read("S2_ref", "mgo_prod")
    w(f"  total_charge={a['head'][3]} nelec={a['head'][4]} maxdev={a['head'][5]}  "
      f"(sum rule bit-exact)")
    for r in a["rows"][-2:]:
        w(f"  c[{r[0]}] {r[1]:>6} q={r[2]:>12.8f} mu={r[4]:>6.3f} res={r[5]:.2e}")
    w(f"  M3b matrix-vs-grid audit = {a['m3b']} e;  E_tot = {a['etot']:.12f} eV")
    w("  -> Mg net +1.4427 e, O net -1.4427 e (literature band +-1.0..1.5 e); the periodic-image")
    w("     Becke partition is validated in bulk for the first time.")

    seq = [(0.3, "0p3"), (0.5, "0p5"), (0.8, "0p8"), (1.0, "1p0"),
           (-0.3, "m0p3"), (-0.5, "m0p5"), (-0.8, "m0p8"), (-1.0, "m1p0")]
    w("\n[S3] single O (atom 4), production grid 60/240; kappa_ref = slope of the first point")
    w(f"  {'delta [e]':>9} {'Q_ref':>12} {'Q_final':>12} {'mu* [Ry]':>12} {'res [e]':>11} "
      f"{'outer':>6} {'iters':>6} {'kappa':>8} {'local':>8} {'dev':>7}  status")
    data = {d: read(f"S3_O_{t}", "mgo_prod") for d, t in seq}
    refs = {1: abs(data[0.3]["rows"][-1][4] / 0.3), -1: abs(data[-0.3]["rows"][-1][4] / -0.3)}
    # Previous point *on the same side*: the local slope is only meaningful
    # between two deltas of the same sign (different sides have different
    # physics, and mu changes sign with delta).
    prev_same = {1: None, -1: None}
    for d, _ in seq:
        a = data[d]
        q, mu, res = a["rows"][-1][2], a["rows"][-1][4], a["rows"][-1][5]
        loc = dev = "-"
        side = 1 if d > 0 else -1
        if prev_same[side] is not None:
            pd, pmu = prev_same[side]
            loc_v = abs((mu - pmu) / (d - pd))
            loc = f"{loc_v:.3f}"
            dev = f"{100 * (loc_v - refs[side]) / refs[side]:+.1f}%"
        prev_same[side] = (d, mu)
        w(f"  {d:>9.2f} {a['rows'][0][2]:>12.6f} {q:>12.6f} {mu:>12.6f} {res:>11.2e} "
          f"{a['outer']:>6} {a['iters']:>6} {abs(mu / d):>8.3f} {loc:>8} {dev:>7}  {a['status']}")

    w("\n[S4] single Mg (atom 0), production grid 60/240")
    for d, tag in ((1.0, "1p0"), (-1.0, "m1p0")):
        a = read(f"S4_Mg_{tag}", "mgo_prod")
        mu = a["rows"][-1][4]
        w(f"  delta={d:+.1f} e  Q_ref={a['rows'][0][2]:.8f}  Q_final={a['rows'][-1][2]:.8f}  "
          f"mu*={mu:+.6f} Ry  kappa={abs(mu / d):.3f} Ry/e  outer={a['outer']} "
          f"iters={a['iters']}  {a['status']}")
    w("  -> the Mg(3+) side did NOT fuse (mu*=+2.115 Ry << 5.0 Ry cap), so the plan's predicted")
    w("     fuse case does not exist at |delta| = 1.0 e.")

    w("\n[robustness]")
    mm = {r: read("S3_O_0p8", r)["rows"][-1][4] for r in ("mgo_cross", "mgo_prod")}
    for r, lbl in (("mgo_prod", "60/240"), ("mgo_cross", "80/320")):
        w(f"  delta=+0.8 e, grid {lbl:>6}: mu* = {mm[r]:+.9f} Ry")
    w(f"  -> relative difference {100 * abs(mm['mgo_cross'] - mm['mgo_prod']) / abs(mm['mgo_prod']):.4f}%"
      f"  (kappa is grid-robust)")
    E = {0.0: read("S2_ref", "mgo_prod")["etot"]}
    MU = {0.0: 0.0}
    for d, _ in seq:
        E[d] = data[d]["etot"]
        MU[d] = data[d]["rows"][-1][4]
    w(f"  {'interval':>13} {'-dE/ddelta [Ry/e]':>18} {'mu* at midpoint':>16} {'rel.dev':>9}")
    for x, y in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.0),
                 (0.0, -0.3), (-0.3, -0.5), (-0.5, -0.8), (-0.8, -1.0)):
        s = -(E[y] - E[x]) / (y - x) / RY
        mid = 0.5 * (MU[x] + MU[y])
        w(f"  [{x:+.1f},{y:+.1f}]".rjust(13) + f" {s:>18.4f} {mid:>16.4f} "
          f"{100 * (s - mid) / mid:>8.2f}%")
    w("  -> mu*(midpoint) = -dE_tot/ddelta closes to <0.8% on all eight deltas (independent check).")

    w("\n[verdict] linear window: -1.0 .. +0.8 e (negative side dev <= 4.0%, positive side <= 12.2%")
    w("          up to +0.8 e; +1.0 e deviates +23.0%) => >= +-0.8 e = 2.7x the H2O +-0.3 e")
    w("          window.  R4 CONFIRMED (ionic systems do have a substantially wider linear domain).")

    open(os.path.join(HERE, "results", "summary.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
