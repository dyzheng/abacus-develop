#!/usr/bin/env python3
"""Extract the inner_thr scan metrics from a work root.

Usage: extract_scan.py <workroot> "<thr1 thr2 ...>" "<case1 case2 ...>"

All framework metrics come from OUT.autotest/running_scf.log (that is where the
CONSTRAINT_AUDIT / inner / outer / settle lines are written); only the
wall-clock footer comes from the stdout run.log.

Per run: SCF iteration count (cost proxy), outer / inner mu steps, mix resets,
reset cost ("mixing recovered after K"), settle pass/fail, degrade events,
final mu / residual / target, final status, E_tot and wall time.
"""
import os
import re
import sys

workroot, thrs, cases = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()


def parse(run_log):
    scf = os.path.join(os.path.dirname(run_log), "OUT.autotest", "running_scf.log")
    txt = open(scf, errors="replace").read() if os.path.exists(scf) else ""
    out = {}
    # running_scf.log numbers each SCF iteration with "#ELEC ITER# N".
    elec = re.findall(r"#ELEC ITER#\s+(\d+)", txt)
    out["nscf"] = int(elec[-1]) if elec else 0
    out["outer"] = txt.count("[constraint] outer step")
    out["inner"] = txt.count("[constraint] inner step")
    out["reset"] = txt.count("[constraint] MIX_RESET at")
    out["suppressed"] = txt.count("MIX_RESET SUPPRESSED")
    out["settle_pass"] = txt.count("settle check PASSED")
    out["settle_fail"] = txt.count("settle check FAILED")
    out["degrade"] = txt.count("degraded to OUTER")
    rec = re.findall(r"mixing recovered after (\d+) SCF iteration", txt)
    out["reset_cost_max"] = max((int(k) for k in rec), default=0)
    out["reset_cost_sum"] = sum(int(k) for k in rec)
    m = re.findall(r"\[constraint\] final status: (\w+)", txt)
    out["status"] = m[-1] if m else "?"
    m = re.findall(r"!FINAL_ETOT_IS\s+([-\d.eE+]+)", txt)
    out["etot"] = m[-1] if m else "?"
    rows = re.findall(r"CONSTRAINT_AUDIT c\[(\d+)\] kind=(\S+) q=(\S+) t=(\S+) mu=(\S+) res=(\S+)", txt)
    out["final"] = "; ".join("c[%s]%s q=%s t=%s mu=%s res=%s" % r for r in rows[-2:])
    rt = open(run_log, errors="replace").read() if os.path.exists(run_log) else ""
    m = re.search(r"TOTAL  Time\s*:\s*(\d+)", rt)
    out["wall"] = m.group(1) if m else "?"
    return out


print("  %-14s %5s %4s %4s %4s %5s %6s %4s %4s %3s  %-10s %-22s %6s" %
      ("run", "SCF", "out", "inn", "MIX", "costS", "costMax", "sP", "sF", "deg",
       "status", "E_tot", "wall/s"))
for case in cases:
    cid = case.split("_")[0]
    for thr in thrs:
        tag = thr.replace(".", "p").replace("-", "m")
        name = "%s_thr%s" % (cid, tag)
        run_log = os.path.join(workroot, name, "run.log")
        if not os.path.exists(run_log):
            print("  %-14s  (missing %s)" % (name, run_log))
            continue
        d = parse(run_log)
        print("  %-14s %5d %4d %4d %4d %5d %6d %4d %4d %3d  %-10s %-22s %6s" %
              (name, d["nscf"], d["outer"], d["inner"], d["reset"], d["reset_cost_sum"],
               d["reset_cost_max"], d["settle_pass"], d["settle_fail"], d["degrade"],
               d["status"], d["etot"], d["wall"]))
        print("        final: %s" % d["final"])
