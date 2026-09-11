#!/usr/bin/env python3
"""Trim the S3L (re-anchored scan) SCF logs into committable audit extracts.

Usage: extract_s3l_audit.py [workroot] [outdir]

The full running_scf.log files live in /tmp and are both huge and swallowed by
the repo-wide *.log gitignore, so the committed record is a trimmed extract per
case holding the facts the spec cites: the final total energy, the outer-loop
status, the Fe on-site moments, and every CONSTRAINT_AUDIT line (the outer
loop's whole trajectory, which is what the branch analysis reads).
"""
import glob
import os
import re
import sys


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else "/tmp/feo_spin"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "s3l")
    os.makedirs(out, exist_ok=True)
    n = 0
    for d in sorted(glob.glob(os.path.join(work, "S3L_*"))):
        log = os.path.join(d, "OUT.autotest", "running_scf.log")
        # Branch: a case directory with no log (killed before ABACUS wrote one)
        # contributes nothing rather than a bogus empty record.
        if not os.path.isfile(log):
            continue
        txt = open(log, errors="ignore").read()
        lines = []
        lines += re.findall(r"!FINAL_ETOT_IS.*", txt)
        lines += re.findall(r"\[constraint\] final status:.*", txt)[-1:]
        lines += re.findall(r"atomic mag:.*", txt)[-2:]
        lines += re.findall(r"CONSTRAINT_AUDIT.*", txt)
        name = os.path.basename(d) + ".audit"
        open(os.path.join(out, name), "w").write("\n".join(lines) + "\n")
        n += 1
    print(f"wrote {n} extracts to {out}")


if __name__ == "__main__":
    main()
