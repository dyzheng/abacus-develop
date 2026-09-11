#!/usr/bin/env python3
"""S6 summary: mu(delta) response table for the FeO spin scan.

Usage: summarize.py <workroot>

Reads <workroot>/<case>/OUT.autotest/running_scf.log for every S3 point and
prints the reference moment Q_ref (the mu = 0 observation of that run's
reference phase), the final moment Q, the converged multiplier mu*, the outer
step count, the ABACUS on-site Fe moments and the cumulative stiffness
kappa = |mu*/delta|.
"""
import os
import re
import sys

DELTAS = [0.1, 0.3, 0.5, -0.1, -0.3, -0.5]


def tag(delta):
    return f"{delta:.1f}".lstrip("+").replace("-", "m").replace(".", "p")


def read_case(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors="replace").read()
    rows = re.findall(
        r"CONSTRAINT_AUDIT c\[0\] kind=\S+ q=(\S+) t=(\S+) mu=(\S+) res=(\S+)", txt)
    if not rows:
        return None
    st = re.search(r"final status: (\w+)", txt)
    mag2 = re.findall(r"atomic mag: 2 (\S+)", txt)
    mag3 = re.findall(r"atomic mag: 3 (\S+)", txt)
    etot = re.search(r"!FINAL_ETOT_IS (\S+) eV", txt)
    return dict(
        rows=[tuple(map(float, r)) for r in rows],
        status=st.group(1) if st else "?",
        outer=txt.count("[constraint] outer step"),
        m2=float(mag2[-1]) if mag2 else None,
        m3=float(mag3[-1]) if mag3 else None,
        etot=float(etot.group(1)) if etot else None,
    )


def main():
    work = sys.argv[1]
    print(f"  {'delta':>7} {'Q_ref':>12} {'Q_final':>12} {'mu(Ry)':>12} "
          f"{'outer':>5} {'status':>10} {'M_on(2)':>9} {'M_on(3)':>9} {'kappa':>7}")
    for d in DELTAS:
        r = read_case(os.path.join(work, f"S3_fe2_{tag(d)}", "OUT.autotest",
                                   "running_scf.log"))
        if r is None:
            print(f"  {d:>7.2f}  MISSING")
            continue
        qref, q, mu = r["rows"][0][0], r["rows"][-1][0], r["rows"][-1][2]
        kappa = abs(mu / d) if d else float("nan")
        print(f"  {d:>7.2f} {qref:12.6f} {q:12.6f} {mu:12.6f} {r['outer']:>5} "
              f"{r['status']:>10} {r['m2']:9.5f} {r['m3']:9.5f} {kappa:7.3f}")
    print("\n  H2O spin-channel reference kappa = 1.38 e/Ry "
          "(docs/superpowers/specs/2026-08-31-v1-v3-validation.md)")


if __name__ == "__main__":
    main()
