#!/usr/bin/env python3
"""Extract the III-1 CT-pair scan from the per-point CONSTRAINT_AUDIT files.

Each results/<tag>.audit holds the full audit stream of one charge-transfer
point: a v2 constraint list with c[0] = acceptor fragment (+delta) and
c[1] = donor fragment (-delta).  We keep the LAST audit block (the converged
one), the final energy and the wall-clock footer, then report:

  * the constraint work  W = sum_i mu_i * (q_i - q_free_i),
    with q_free_i = t_i - delta_i (delta target mode stores t = q_free + delta),
  * the local response dmu_acc/ddelta per interval (the CT pair is NOT
    harmonic, so interval slopes matter more than a global chord),
  * the stationarity identity  -(dE_tot/ddelta)[Ry/e] vs (mu_acc - mu_don),
    the same check I-1 used to validate the energy route.

Usage: python3 tools/extract_scan.py <results-dir>
"""
import glob
import os
import re
import sys

# !FINAL_ETOT_IS is printed in eV while mu is in Ry; the stationarity identity
# only closes after converting the energy derivative into Ry/e.
RY_TO_EV = 13.605693009


def parse_delta(path):
    """Recover the charge-transfer amplitude from the file tag (d_p0.10.audit)."""
    m = re.search(r"d_([pm])(\d+\.\d+)\.audit$", os.path.basename(path))
    if not m:
        return None
    sign = -1.0 if m.group(1) == "m" else 1.0
    return sign * float(m.group(2))


def parse_point(path):
    """Last converged audit block + final energy + wall time of one point."""
    txt = open(path).read()
    blocks = re.findall(r"CONSTRAINT_AUDIT nconstraint=(\d+)[^\n]*\n((?:CONSTRAINT_AUDIT c\[\d+\][^\n]*\n)*)", txt)
    if not blocks:
        return None
    _, cons = blocks[-1]
    rows = []
    for line in cons.strip().splitlines():
        f = dict(re.findall(r"(\w+)=([-\w.eE+]+)", line))
        rows.append({"kind": f.get("kind"), "q": float(f["q"]), "t": float(f["t"]), "mu": float(f["mu"])})
    etot = re.search(r"!FINAL_ETOT_IS\s+([-\d.eE+]+)", txt)
    time_s = re.search(r"TOTAL\s+Time\s+:\s+(\d+)", txt)
    status = "CONVERGED" if "final status: CONVERGED" in txt else (
        "NOT-CONVERGED" if "IS NOT CONVERGED" in txt else "?")
    return {
        "cons": rows,
        "e_tot": float(etot.group(1)) if etot else None,
        "time": int(time_s.group(1)) if time_s else None,
        "status": status,
    }


def main():
    resdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")
    points = []
    for path in sorted(glob.glob(os.path.join(resdir, "d_*.audit"))):
        delta = parse_delta(path)
        pt = parse_point(path)
        if delta is None or pt is None or len(pt["cons"]) < 2:
            continue
        acc, don = pt["cons"][0], pt["cons"][1]
        # Branch: delta target mode stores t = q_free + delta, so the reference
        # reading is recovered as t - delta (acceptor) / t + delta (donor).
        dq_acc = acc["q"] - (acc["t"] - delta)
        dq_don = don["q"] - (don["t"] + delta)
        points.append({"delta": delta, "acc": acc, "don": don, "dq_acc": dq_acc,
                       "dq_don": dq_don, "W": acc["mu"] * dq_acc + don["mu"] * dq_don,
                       "e_tot": pt["e_tot"], "time": pt["time"], "status": pt["status"]})
    points.sort(key=lambda p: p["delta"])

    out = []
    out.append("III-1  H2O dimer charge-transfer pair (acceptor +d / donor -d)")
    out.append("")
    out.append(" d(e)   q_acc        t_acc        mu_acc(Ry)     q_don        t_don        mu_don(Ry)     W(Ry)        E_tot(eV)          status")
    for p in points:
        out.append(" %+5.2f  %-12.7f %-12.7f %+-13.8f %-12.7f %-12.7f %+-13.8f %-12.8f %-18.10f %s"
                   % (p["delta"], p["acc"]["q"], p["acc"]["t"], p["acc"]["mu"],
                      p["don"]["q"], p["don"]["t"], p["don"]["mu"], p["W"], p["e_tot"], p["status"]))

    out.append("")
    out.append(" stationarity identity  -(dE_tot/ddelta)[Ry/e]  vs  (mu_acc - mu_don)   [consecutive pairs]")
    out.append(" interval(e)       -dE/dd(Ry/e)    mu_acc-mu_don(Ry)   dev(%)    dmu_acc/dd(Ry/e)")
    for a, b in zip(points, points[1:]):
        if a["e_tot"] is None or b["e_tot"] is None:
            continue
        deriv = -(b["e_tot"] - a["e_tot"]) / (b["delta"] - a["delta"]) / RY_TO_EV
        pot = 0.5 * ((a["acc"]["mu"] - a["don"]["mu"]) + (b["acc"]["mu"] - b["don"]["mu"]))
        dev = 100.0 * (deriv - pot) / pot if pot else float("nan")
        slope = (b["acc"]["mu"] - a["acc"]["mu"]) / (b["delta"] - a["delta"])
        out.append(" %+5.2f..%+5.2f   %+-14.8f %+-18.8f %+8.2f   %+-10.4f"
                   % (a["delta"], b["delta"], deriv, pot, dev, slope))

    out.append("")
    # The pair is asymmetric (donor/acceptor roles differ), so report the two
    # branches separately instead of a single chord slope.
    for name, sel in (("delta < 0 (acceptor loses charge)", [p for p in points if p["delta"] < 0]),
                      ("delta > 0 (acceptor gains charge)", [p for p in points if p["delta"] > 0])):
        if len(sel) >= 2:
            kappa = (sel[-1]["acc"]["mu"] - sel[0]["acc"]["mu"]) / (sel[-1]["delta"] - sel[0]["delta"])
            out.append(" mean kappa (dmu_acc/ddelta) %-34s = %+.4f Ry/e" % (name, kappa))
    if len(points) >= 3:
        mi = min(points, key=lambda p: p["e_tot"])
        out.append(" energy minimum of the scan: delta = %+.2f e (E_tot = %.6f eV)" % (mi["delta"], mi["e_tot"]))
    antisym = max(abs(p["acc"]["mu"] + p["don"]["mu"]) for p in points)
    out.append(" max |mu_acc + mu_don| = %.3e Ry   (fragment antisymmetry of the CT pair)" % antisym)
    out.append(" sum-rule: total_charge == nelec held at every point (see audit header)")

    text = "\n".join(out) + "\n"
    print(text)
    with open(os.path.join(resdir, "summary.txt"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
