#!/usr/bin/env python3
"""Summarize the re-anchored II-1 scan (S3L) with an explicit branch audit.

Usage: summarize_low_scan.py [workroot] [anchor-etot-eV]

The inherited Gamma-only FeO cell has several low-lying self-consistent
solutions, so a converged outer-loop residual |Q - t| < thr is NOT by itself
evidence that the constraint stayed on the branch it started from.  This tool
applies the three criteria the round's spec fixes:

  B1 energy      E_tot(delta) >= E_anchor - 1e-4 eV.  A constrained density
                 cannot be lower than the unconstrained minimum of its own
                 branch, so a drop below the anchor means the SCF left it.
                 (Valid because the anchor is the LOWER of the two known
                 Gamma-only solutions.)
  B2 linear resp  |dE - 0.5 * sum_a delta_a * mu_a| small.  For a linear
                 response channel the constrained energy cost is exactly
                 0.5 * delta * mu; a large mismatch means a different state.
  B3 moment      the free Fe moment stays within 15% of the anchor's.

A point that fails any of them is reported as OFF-BRANCH regardless of what
the outer loop printed.
"""
import glob
import os
import re
import sys

RY_EV = 13.605693122994


def read_case(d):
    log = os.path.join(d, "OUT.autotest", "running_scf.log")
    if not os.path.isfile(log):
        return None
    txt = open(log, errors="ignore").read()
    rec = {"dir": os.path.basename(d)}

    m = re.findall(r"!FINAL_ETOT_IS\s+(\S+)", txt)
    rec["etot"] = float(m[-1]) if m else None

    m = re.findall(r"\[constraint\] final status: (\w+)", txt)
    rec["status"] = m[-1] if m else "NO_STATUS"

    m = re.findall(r"CONSTRAINT_AUDIT c\[0\] kind=spin q=(\S+) t=(\S+) mu=(\S+) res=(\S+)", txt)
    # Branch: any constraint audit at all -> last line is the final observation,
    # first line is the reference (mu = 0, Q_ref) observation.  Branch: no audit
    # (e.g. a killed run) -> leave the fields unset and report the status only.
    if m:
        q, t, mu, res = (float(v) for v in m[-1])
        rec.update(q=q, q_ref=float(m[0][0]), target=t, mu=mu, res=res)
        rec["outer_steps"] = len(m)
    else:
        rec.update(q=None, q_ref=None, target=None, mu=None, res=None)
        rec["outer_steps"] = 0

    m = re.findall(r"atomic mag:\s+(\d+)\s+(\S+)", txt)
    rec["mags"] = {int(i): float(v) for i, v in m[-4:]} if m else {}
    return rec


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else "/tmp/feo_spin"
    anchor = float(sys.argv[2]) if len(sys.argv) > 2 else None

    donor = read_case(os.path.join(work, "S3L_donor"))
    if anchor is None:
        if donor is None or donor["etot"] is None:
            sys.exit("no S3L_donor record and no anchor given")
        anchor = donor["etot"]
    ref_mag2 = donor["mags"].get(2) if donor else None

    print(f"anchor (S3L_donor): E = {anchor:.10f} eV"
          + (f", Fe2 on-site / Becke Q_ref = {donor['q']}" if donor and donor["q"] else ""))
    print()
    hdr = (f"{'delta[uB]':>9} {'Q_ref':>12} {'Q_final':>12} {'mu*[Ry]':>12} "
           f"{'status':>10} {'E_tot[eV]':>16} {'dE[eV]':>10} {'0.5*d*|mu|':>10} "
           f"{'Fe2[muB]':>9} verdict")
    print(hdr)
    print("-" * len(hdr))

    for d in sorted(glob.glob(os.path.join(work, "S3L_fe2_*"))):
        rec = read_case(d)
        # Branch: a killed mid-flight run has no FINAL_ETOT_IS; still report
        # its last observation and status so the record is complete.
        if rec is None:
            continue
        tag = rec["dir"].split("S3L_fe2_")[1]
        delta = float(tag.replace("m", "-").replace("p", "."))
        de = rec["etot"] - anchor if rec["etot"] is not None else None
        # Constrained energy cost of a linear-response channel with dQ/dmu < 0:
        # E(delta) - E(0) = 0.5 * delta * |mu*| = -0.5 * delta * mu* (mu* and
        # delta carry opposite signs on the standard branch).
        pred = -0.5 * delta * rec["mu"] * RY_EV if rec["mu"] is not None else float("nan")
        mag2 = rec["mags"].get(2)

        verdict = []
        if de is None:
            verdict.append("no-etot")
        if de is not None and de < -1e-4:
            verdict.append("B1:OFF")
        if (de is not None and rec["mu"] is not None
                and abs(de - pred) > max(0.02, 0.15 * abs(pred))):
            verdict.append("B2:OFF")
        if mag2 is not None and ref_mag2 is not None and abs(mag2 - ref_mag2) > 0.15 * abs(ref_mag2):
            verdict.append("B3:OFF")
        if rec["status"] != "CONVERGED":
            verdict.append(rec["status"])
        def fmt(v, w, p=5):
            # Branch: a killed run may not have produced the value yet --
            # print n/a rather than a misleading number.
            return f"{v:>{w}.{p}f}" if v is not None else f"{'n/a':>{w}}"
        print(f"{delta:>9.2f} {fmt(rec['q_ref'], 12, 9)} {fmt(rec['q'], 12, 9)} "
              f"{fmt(rec['mu'], 12, 7)} {rec['status']:>10} "
              f"{fmt(rec['etot'], 16, 7)} {fmt(de, 10)} {fmt(pred, 10)} "
              f"{fmt(mag2, 9)} {' '.join(verdict) if verdict else 'OK'}")


if __name__ == "__main__":
    main()
