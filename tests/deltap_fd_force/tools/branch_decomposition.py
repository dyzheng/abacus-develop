#!/usr/bin/env python3
"""DeltaP branch-vs-smooth decomposition of dlambda*/dR (2026-08-03).

Quantifies how much of the stationary-lambda geometry drift (dlambda*/dR)
comes from the target-aware branch-selection steps (gamma_report = gamma_raw + b,
b quantized by 2*pi*w_In) vs the smooth physical response (gamma_raw).

Inputs: per-geometry SCF run logs with [rawG] (pre-branch gamma) and
[DeltaP P3] (branch-selected gamma_report, lambda) lines, plus E' from the
OUT.<suffix>/running_scf.log FINAL_ETOT_IS line.

Usage:
    python3 branch_decomposition.py <runs.json>

runs.json: list of
  {"label": "base", "geom_z": 7.9365, "lambda_O1": 0.0, "log": "run.log",
   "etot_log": "OUT.x/running_scf.log"}
The target t (atom 0) is given via --target.  lambda is the per-atom lambda
vector as "(lO, lH, lH)"; only lambda_O1 is used for the scan (gdir=3).

Output: per-geometry (lambda*, lambda*_cont, branch contribution) table and
the FD-relevant dlambda*/dR decomposition.
"""
import argparse
import json
import math
import re
import sys

RAW_RE = re.compile(r"\[rawG\] .*?\u03b30=([-+0-9.eE]+) \u03b31=([-+0-9.eE]+) \u03b32=([-+0-9.eE]+)")
P3_RE = re.compile(
    r"\[DeltaP P3\] iter=\d+ .*?\|[^|]*-t\|=([-+0-9.eE]+) escon=")
# P3 gamma values are 3-decimal; use them only as fallback.  Prefer the
# full-precision gamma_report from deltap_branch.dat (row=atom, col=gdir-1).


def parse_rawg(path):
    """Return final-iteration raw gamma [g0, g1, g2] (z column = gdir 3)."""
    last = None
    with open(path) as fh:
        for line in fh:
            m = RAW_RE.search(line)
            if m:
                last = [float(m.group(i)) for i in (1, 2, 3)]
    return last


def parse_p3(path):
    """Return last [DeltaP P3] line's |g-t| and lambda vector."""
    last = None
    with open(path) as fh:
        for line in fh:
            if "DeltaP P3" in line:
                groups = re.findall(r"=\(([^)]*)\)", line)
                gm = re.search(r"\|[^|]*-t\|=([-+0-9.eE]+)", line)
                if groups and gm:
                    lam = [float(x) for x in groups[-1].split(",")]
                    last = (float(gm.group(1)), lam)
    return last


def parse_branch_file(path, nrow=3, gdir=3):
    """Return per-atom reported gamma (row=atom, column=gdir-1)."""
    try:
        with open(path) as fh:
            lines = fh.read().splitlines()
    except FileNotFoundError:
        return None
    # first line is nrow; following lines are rows
    rows = []
    for line in lines[1:]:
        toks = line.split()
        if len(toks) >= gdir:
            rows.append(float(toks[gdir - 1]))
        if len(rows) >= nrow:
            break
    return rows if len(rows) == nrow else None


def parse_etot(path):
    """FINAL_ETOT_IS from OUT running_scf.log (last)."""
    val = None
    with open(path) as fh:
        for line in fh:
            m = re.search(r"!FINAL_ETOT_IS\s+([-+0-9.eE]+)", line)
            if m:
                val = float(m.group(1))
    return val


def solve_lambda_star(gamma_curve, t, lo=-0.05, hi=0.05):
    """Solve gamma_raw(lambda) = t by linear interpolation of (lam, gamma) pts."""
    pts = sorted(gamma_curve)
    if len(pts) < 2:
        return None
    # find bracketing segment
    for (l1, g1), (l2, g2) in zip(pts, pts[1:]):
        if (g1 - t) * (g2 - t) <= 0:
            if abs(g2 - g1) < 1e-12:
                return (l1 + l2) / 2
            return l1 + (t - g1) * (l2 - l1) / (g2 - g1)
    # extrapolate with the steepest (largest |slope|) segment
    segs = sorted(
        (abs((g2 - g1) / (l2 - l1)), l1, g1, l2, g2)
        for (l1, g1), (l2, g2) in zip(pts, pts[1:])
        if l2 != l1)
    if not segs:
        return None
    _, l1, g1, l2, g2 = segs[-1]
    slope = (g2 - g1) / (l2 - l1)
    return l1 + (t - g1) / slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_json")
    ap.add_argument("--target", type=float, required=True, help="target gamma atom 0")
    ap.add_argument("--depdl", type=float, default=224.4,
                    help="dE'/dlambda_O1 in eV/Ry (measured)")
    ap.add_argument("--twodelta", type=float, default=0.0052917722,
                    help="2*delta in Angstrom")
    ap.add_argument("--delta-version", action="store_true",
                    help="decompose dlambda*/dR instead of Delta lambda")
    args = ap.parse_args()

    runs = json.load(open(args.runs_json))
    t = args.target

    # per-geometry: lambda_O1 -> (gamma_raw, gamma_report, |g-t|, E')
    geoms = {}
    for r in runs:
        raw = parse_rawg(r["log"])
        p3 = parse_p3(r["log"])
        br = parse_branch_file(r.get("branch_file", ""))
        etot = parse_etot(r["etot_log"])
        g_report = br[0] if br else (p3[1] if False else None)
        geoms.setdefault(r["label"], []).append({
            "lam": r["lambda_O1"],
            "g_raw": raw[0] if raw else None,
            "g_rep": g_report,
            "gmt": p3[0] if p3 else None,
            "lam_vec": p3[1] if p3 else None,
            "E": etot,
        })

    print(f"target t (atom0) = {t:.10f}")
    print(f"dE'/dlambda_O1   = {args.depdl} eV/Ry")
    print(f"2*delta          = {args.twodelta} A")
    print()

    # ---- per-geometry table ----
    print("=" * 100)
    Ecol = "E'"
    print("=" * 100)
    print(f"{'geom':<10} {'lam*':>10} {'lam*_cont':>11} {'dlam_branch':>12} "
          f"{'g_raw(lam*)':>12} {'g_rep(lam*)':>12} {Ecol:>14}")
    print("-" * 100)
    lam_star = {}
    lam_cont = {}
    for label in sorted(geoms):
        pts = sorted((p["lam"], p["g_raw"]) for p in geoms[label] if p["g_raw"])
        # lambda* = protocol-converged (min |g-t|) point
        best = min((p for p in geoms[label] if p["gmt"] is not None),
                   key=lambda p: p["gmt"])
        ls = best["lam"]
        lc = solve_lambda_star(pts, t)
        db = (ls - lc) if lc is not None else float("nan")
        gr = best["g_raw"]
        gre = best["g_rep"]
        E = best["E"]
        lam_star[label] = ls
        lam_cont[label] = lc
        print(f"{label:<10} {ls:>10.6f} {lc if lc is not None else float('nan'):>11.6f} "
              f"{db:>12.6f} {gr:>12.6f} {gre if gre is not None else float('nan'):>12.6f} "
              f"{E if E is not None else float('nan'):>14.6f}")
    print()

    # ---- decomposition of dlambda*/dR ----
    labels = [l for l in sorted(geoms) if l in lam_star]
    # need plus/minus pair: find geom labels ending _plus / _minus
    def get(base):
        plus = f"{base}_plus"
        minus = f"{base}_minus"
        return (plus, minus) if (plus in lam_star and minus in lam_star) else (None, None)

    bases = sorted({l.rsplit("_", 1)[0] for l in labels})
    print("=" * 100)
    print("dlambda*/dR decomposition (per FD axis, 2*delta)")
    print("-" * 100)
    for base in bases:
        plus, minus = get(base)
        if plus is None:
            continue
        dls = lam_star[plus] - lam_star[minus]
        dlc = lam_cont[plus] - lam_cont[minus]
        dlb = dls - dlc
        dldr_s = dls / args.twodelta
        dldr_c = dlc / args.twodelta
        dldr_b = dlb / args.twodelta
        print(f"  {base}:")
        print(f"    lambda*    : {lam_star[minus]:+.6f} -> {lam_star[plus]:+.6f}"
              f"   dlam_st/dR = {dldr_s:+.4f} Ry/A")
        print(f"    lambda*_cont: {lam_cont[minus]:+.6f} -> {lam_cont[plus]:+.6f}"
              f"   dlam_cont/dR = {dldr_c:+.4f} Ry/A  (smooth)")
        print(f"    branch     : dlam_branch/dR = {dldr_b:+.4f} Ry/A  (selection shift)")
        # FD leak projections
        leak_s = args.depdl * dls / args.twodelta
        leak_c = args.depdl * dlc / args.twodelta
        leak_b = args.depdl * dlb / args.twodelta
        print(f"    FD leak (dE'/dlam x dlam/2delta):")
        print(f"      observed (lambda*): {leak_s:+.2f} eV/A")
        print(f"      smooth-only (cont): {leak_c:+.2f} eV/A  (continuity-only prediction)")
        print(f"      branch contribution: {leak_b:+.2f} eV/A")
        frac_b = abs(dlb) / (abs(dlc) + abs(dlb)) if (dlc or dlb) else float("nan")
        print(f"      |branch|/(|smooth|+|branch|) = {frac_b:.1%} "
              f"(sign: branch opposes smooth -> {dlb / dlc if dlc else float('nan'):+.2f}x)")
    print()


if __name__ == "__main__":
    sys.exit(main())
