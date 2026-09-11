#!/usr/bin/env python3
"""Rebuild results/audit/*.audit and results/summary.txt from the run work trees.

Usage: extract_results.py

The full ABACUS logs (up to 2.7 MB each) live under the /tmp work roots during
a run; the committed artefact keeps only the lines that carry a number the
II-1a write-up uses, plus the summary tables.  The SOURCES table below is the
authoritative mapping from work-tree directory to committed case name, because
the cold-start scan, the hot-start scan and the ad-hoc diagnostics live in
three different work roots.
"""
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(HERE, "results")
RY = 13.605693

# (committed name, work root, case directory)
SOURCES = [
    ("S1_base_1e6", "/tmp/feo_spin", "S1_base_1e6"),
    ("S1x_base_1e8", "/tmp/feo_spin", "S1x_base_1e8"),
    ("S1x_cstr_1e8", "/tmp/feo_spin", "S1x_cstr_1e8"),
    ("S1x_cstr_1e7", "/tmp/feo_spin", "S1x_cstr_1e7"),
    ("S1x_cstr_1e7_mix02", "/tmp/feo_spin", "S1x_cstr_1e7_mix02"),
    ("S2R_ref_delta0", "/tmp/feo_spin", "S2R_ref_delta0"),
    ("S0B_warmstart", "/tmp/feo_spin", "S0B_warmstart"),
    ("S2_fe2_p01", "/tmp/feo50", "t3_thr1e7"),
    ("S1x_cstr_1e7_mix02_adhoc", "/tmp/feo50", "t3b_mix02_thr1e7"),
    ("H1_fe2_m01_hot", "/tmp/feo50", "H1_m0p1_hot"),
    ("H3_fe2_p03_hot", "/tmp/feo50", "H3_p0p3_hot"),
    ("X_lowstate_unconstrained", "/tmp/feo50", "W2"),
    ("X_case11_fe_sublattice_p01", "/tmp/feo_cstr", ""),
]
for _d in ["0p1", "0p3", "0p5", "m0p1", "m0p3", "m0p5"]:
    SOURCES.append((f"S3cold_fe2_{_d}", "/tmp/feo_spin", f"S3_fe2_{_d}"))
    SOURCES.append((f"S3hot_fe2_{_d}", "/tmp/feo_spin_hot", f"S3_fe2_{_d}"))

KEEP = re.compile(
    r"CONSTRAINT_AUDIT|\[constraint\]|!FINAL_ETOT_IS|"
    r"charge density convergence")
# Per-SCF-iteration magnetic moments are printed dozens of times per run; only
# the converged tail is informative, so they are buffered and flushed last.
MAG = re.compile(r"atomic mag:|Total magnetism|Absolute magnetism")
MAG_TAIL = 6


def trim(src, dst):
    tail = []
    with open(src, errors="replace") as fin, open(dst, "w") as fout:
        for line in fin:
            if KEEP.search(line):
                fout.write(line)
            elif MAG.search(line):
                tail.append(line)
                del tail[:-MAG_TAIL]
        fout.writelines(tail)


def stats(path):
    txt = open(path, errors="replace").read()
    rows = re.findall(
        r"CONSTRAINT_AUDIT c\[(\d+)\] kind=(\S+) q=(\S+) t=(\S+) mu=(\S+) res=(\S+)", txt)
    st = re.search(r"final status: (\w+)", txt)
    etot = re.search(r"!FINAL_ETOT_IS (\S+) eV", txt)
    mag2 = re.findall(r"atomic mag: \d+2? (\S+)", txt)
    return dict(rows=rows, status=st.group(1) if st else "?",
                outer=txt.count("[constraint] outer step"),
                etot=float(etot.group(1)) if etot else None, mag2=mag2)


def main():
    audit = os.path.join(RESDIR, "audit")
    os.makedirs(audit, exist_ok=True)
    lines = []
    for name, root, case in SOURCES:
        src = os.path.join(root, case, "OUT.autotest", "running_scf.log")
        if not os.path.exists(src):
            lines.append(f"  {name}: MISSING ({src})")
            continue
        trim(src, os.path.join(audit, f"{name}.audit"))
        s = stats(src)
        qref = s["rows"][0][2] if s["rows"] else "?"
        q = s["rows"][-1][2] if s["rows"] else "?"
        mu = s["rows"][-1][4] if s["rows"] else "?"
        e = f"{s['etot']:.6f}" if s["etot"] is not None else "?"
        lines.append(f"  {name:32s} Q_ref={qref:>14} Q_final={q:>14} "
                     f"mu={mu:>16} outer={s['outer']:>3} {s['status']:>10} E={e}")
    out = os.path.join(RESDIR, "summary.txt")
    with open(out, "w") as f:
        f.write("II-1 FeO spin-constraint scan (2026-09-11)\n")
        f.write("Regenerate with: python3 tools/extract_results.py\n\n")
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
