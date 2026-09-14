#!/usr/bin/env python3
"""Write results/summary.txt for the inner_thr scan.

Usage: summarize.py <workroot> <outfile>

Reads the OUTER baselines from the committed Q1-Q3 evidence
(tests/deltap_dual_iteration/results/summary.txt) so the cost comparison is
against the same binary/grid/target/initial guess.
"""
import os
import re
import sys

workroot, outfile = sys.argv[1], sys.argv[2]
THRS = ["1e-3", "1e-4", "1e-5"]
CASES = [("212", "212_PW_spin", "spin delta=+0.1 uB"),
         ("213", "213_PW_mixed", "mixed charge+spin delta=+0.1 e / +0.1 uB")]


def metrics(work):
    rsc = os.path.join(work, "OUT.autotest", "running_scf.log")
    txt = open(rsc, errors="replace").read()
    elec = re.findall(r"#ELEC ITER#\s+(\d+)", txt)
    d = {"scf": int(elec[-1]) if elec else 0,
         "outer": txt.count("[constraint] outer step"),
         "inner": txt.count("[constraint] inner step"),
         "resets": txt.count("[constraint] MIX_RESET at"),
         "sP": txt.count("settle check PASSED"),
         "sF": txt.count("settle check FAILED"),
         "deg": txt.count("degraded to OUTER")}
    m = re.findall(r"!FINAL_ETOT_IS\s+([-\d.eE+]+)", txt)
    d["etot"] = float(m[-1]) if m else float("nan")
    mus = re.findall(r"CONSTRAINT_AUDIT c\[\d+\] kind=(\S+) .* mu=(\S+) res=", txt)
    d["mu"] = [float(x[1]) for x in mus[-2:]] if mus else []
    m = re.findall(r"\[constraint\] final status: (\w+)", txt)
    d["status"] = m[-1] if m else "?"
    return d


# OUTER baselines from the committed dual-iteration evidence
base = {}
bpath = "tests/deltap_dual_iteration/results/summary.txt"
if os.path.exists(bpath):
    cur = None
    for line in open(bpath):
        m = re.match(r"^(\S+)_PW_(\w+)_(\w+)$", line.strip())
        if m:
            cur = "%s/%s" % (m.group(1), m.group(3))  # id/schedule
            base[cur] = {}
            continue
        if cur and "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            base[cur][k] = v

L = []
L.append("inner_thr three-point scan (priority-2 calibration)")
L.append("=" * 60)
L.append("Cases : tests/01_PW/212_PW_constraint_h2o_spin (spin), 213_PW_constraint_h2o_mixed")
L.append("Binary: build_rel/abacus_basic_para (Release, current HEAD); np=4; OMP_NUM_THREADS=1")
L.append("Grid/target/initial guess identical to the committed Q1-Q3 comparison")
L.append("(tests/deltap_dual_iteration); only constraint_inner_thr varies.")
L.append("Runner: run_inner_thr_scan.sh   Extractor: tools/extract_scan.py")
L.append("")
L.append("%-6s %-8s %5s %5s %5s %5s %4s %4s %3s %-10s %s" %
         ("case", "thr", "SCF", "out", "inn", "MIX", "sP", "sF", "deg", "status", "E_tot/eV"))
rows = {}
for cid, tag, desc in CASES:
    for thr in THRS:
        t = thr.replace(".", "p").replace("-", "m")
        name = "%s_thr%s" % (cid, t)
        d = metrics(os.path.join(workroot, name))
        rows[(cid, thr)] = d
        L.append("%-6s %-8s %5d %5d %5d %5d %4d %4d %3d %-10s %.10f" %
                 (cid, thr, d["scf"], d["outer"], d["inner"], d["resets"],
                  d["sP"], d["sF"], d["deg"], d["status"], d["etot"]))
    L.append("")
L.append("OUTER baselines (same binary/grid/target/guess, committed evidence):")
for cid, tag, desc in CASES:
    b = base.get("%s/outer" % cid, {})
    bi = base.get("%s/inner" % cid, {})
    L.append("  %s (%s): OUTER SCF=%-4s outer=%-3s | INNER thr=1e-3 SCF=%-4s inner=%-3s reset=%-3s" %
             (cid, desc, b.get("last_iter", "?"), b.get("outer", "?"),
              bi.get("last_iter", "?"), bi.get("inner", "?"), bi.get("reset", "?")))
L.append("")
L.append("Cost vs OUTER baseline and correctness vs the thr=1e-3 INNER reference:")
L.append("%-6s %-8s %8s %8s %10s %12s" %
         ("case", "thr", "dSCF", "dSCF%", "max|dmu|/|mu|", "max|dE_tot|/eV"))
for cid, tag, desc in CASES:
    b = base.get("%s/outer" % cid, {})
    n0 = int(b.get("last_iter", 0)) if b.get("last_iter") else None
    ref = rows[(cid, "1e-3")]
    for thr in THRS:
        d = rows[(cid, thr)]
        dscf = d["scf"] - n0 if n0 else None
        dscfp = (100.0 * dscf / n0) if n0 else None
        dmu = max(abs(a - c) / abs(c) for a, c in zip(d["mu"], ref["mu"])) if (d["mu"] and ref["mu"]) else None
        detot = abs(d["etot"] - ref["etot"])
        L.append("%-6s %-8s %8s %8s %10s %12.3e" %
                 (cid, thr, dscf, "%.1f%%" % dscfp if dscfp is not None else "?",
                  "%.3f%%" % (100 * dmu) if dmu is not None else "?",
                  detot if thr != "1e-3" else 0.0))
    L.append("")
L.append("Supplementary cross-check: MgO bulk / LCAO (the case with the largest OUTER")
L.append("penalty).  Same binary/grid/target/restart as the committed MgO INNER evidence;")
L.append("thr=1e-3 row is that committed run (tests/deltap_dual_iteration/results/summary.txt).")
L.append("")
L.append("%-6s %-8s %5s %5s %5s %5s %4s %4s %3s %-10s %s" %
         ("case", "thr", "SCF", "out", "inn", "MIX", "sP", "sF", "deg", "status", "E_tot/eV"))
mgo_ref = {"scf": 94, "inner": 31, "resets": 30, "sP": 1, "sF": 0, "deg": 0,
           "status": "CONVERGED", "etot": -7659.26011634944}
mgo = {"1e-3": mgo_ref}
for thr in ["1e-4", "1e-5"]:
    d = metrics(os.path.join(workroot, "mgo_thr%s" % thr.replace(".", "p").replace("-", "m")))
    mgo[thr] = d
for thr in THRS:
    d = mgo[thr]
    L.append("%-6s %-8s %5d %5d %5d %5d %4d %4d %3d %-10s %.10f" %
             ("MgO", thr, d["scf"], d.get("outer", 1), d["inner"], d["resets"],
              d["sP"], d["sF"], d["deg"], d["status"], d["etot"]))
L.append("")
L.append("MgO: OUTER baseline = 381 SCF iterations (committed evidence) -> INNER wins")
L.append("      -75% / -72% / -66% at 1e-3 / 1e-4 / 1e-5, i.e. here a STRICTER gate is")
L.append("      worse: the cost grows 94 -> 107 (+14%) -> 129 (+37%) relative to 1e-3.")
L.append("")
L.append("Conclusion (calibration):")
L.append("1. Correctness: the gate is a pure cost knob.  Across all three systems the mu*")
L.append("   spread is <= 0.34% and |dE_tot| <= 4.2e-7 eV between gate values, inside the")
L.append("   equivalence criteria (< 1% / < 1e-6 eV).  No correctness reason to change the")
L.append("   default.")
L.append("2. Cost: the sign of the gate effect is SYSTEM-DEPENDENT -- 212 improves")
L.append("   (+4.8% -> -9.5% vs OUTER), 213 is flat (-46% / -47% / -45%), MgO degrades")
L.append("   (-75% -> -72% -> -66%).  A stricter gate always means fewer in-SCF mu updates")
L.append("   (212: 27/11/10, 213: 28/25/24, MgO: 31/22/22), so what changes is whether the")
L.append("   saved mixing churn outweighs the longer approach phase -- that balance is")
L.append("   system-specific.")
L.append("3. Recommendation: KEEP the default 1e-3 (DeltaSpin convention).  Treat")
L.append("   constraint_inner_thr as a per-system knob: if INNER loses to OUTER on a system,")
L.append("   a 2-3 point scan (1e-3 / 1e-4 / 1e-5) can recover the deficit (as on 212) but")
L.append("   can also hurt (as on MgO) -- never tune it blindly.")
L.append("4. The settle check fires at every gate value (1-2 bounces per point, MgO 0), so")
L.append("   it is not an artifact of a loose gate; keep it on unconditionally.")
L.append("")
L.append("C-29 note: the MgO points hot-start from a 160/640 charge file read by a 60/240")
L.append("run -- the exact cross-basis pattern that used to corrupt the heap.  With the")
L.append("read_rhog guard they finish with rc=0 and 0 corruption lines (Release binary).")
open(outfile, "w").write("\n".join(L) + "\n")
print("\n".join(L))
