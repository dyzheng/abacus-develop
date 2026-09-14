#!/bin/bash
# Priority-2 calibration: inner_thr three-point scan on the H2O PW cases 212/213.
#
# Question: the INNER schedule's drho gate `constraint_inner_thr` defaults to
# 1e-3 (DeltaSpin convention, never calibrated here).  How does it trade cost
# against correctness?  Scan 1e-3 / 1e-4 / 1e-5 on
#   212 (PW, spin constraint delta=+0.1 uB) and
#   213 (PW, mixed charge+spin constraints delta=+0.1 e / +0.1 uB).
# All runs: same binary/grid/target/initial guess as the Q1-Q3 comparison
# (tests/deltap_dual_iteration), np=4, OMP_NUM_THREADS=1.
#
# Usage: bash run_inner_thr_scan.sh [nproc]
set -uo pipefail

NPROC="${1:-4}"
ABACUS="${ABACUS:-/root/abacus-develop/build_rel/abacus_basic_para}"
CASEDIR="$(cd "$(dirname "$0")" && pwd)"
SRCROOT="$(cd "$CASEDIR/../01_PW" && pwd)"
WORKROOT="${WORKROOT:-/tmp/inner_thr_scan}"
RESDIR="$CASEDIR/results"
THRS="${THRS:-1e-3 1e-4 1e-5}"
CASES="${CASES:-212_PW_constraint_h2o_spin 213_PW_constraint_h2o_mixed}"
export OMP_NUM_THREADS=1
mkdir -p "$WORKROOT" "$RESDIR"

for case in $CASES; do
    for thr in $THRS; do
        tag=$(echo "$thr" | tr '.-' 'pm')
        name="${case%%_PW*}_thr${tag}"
        work="$WORKROOT/$name"
        # Branch: fresh work dir per point (the Q1-Q3 runs were also cold-start
        # per case; only the schedule parameter varies, never the initial guess).
        python3 - "$work" <<'PY'
import shutil, os, sys
d = sys.argv[1]
if os.path.isdir(d):
    shutil.rmtree(d)
os.makedirs(d)
PY
        cp "$SRCROOT/$case/INPUT" "$SRCROOT/$case/KPT" "$SRCROOT/$case/STRU" \
           "$SRCROOT/$case/constraint_target.json" "$work/"
        # The autotest INPUTs use pseudo_dir ../../PP_ORB (relative to the case
        # directory); the work dir lives under /tmp, so pin the absolute path.
        sed -i "s|^\(pseudo_dir\).*|\1 $SRCROOT/../PP_ORB|" "$work/INPUT"
        sed -i "s|^\(orbital_dir\).*|\1 $SRCROOT/../PP_ORB|" "$work/INPUT"
        {
            echo "constraint_mu_schedule inner"
            echo "constraint_inner_thr   $thr"
            echo "constraint_inner_nmax  200"
        } >> "$work/INPUT"
        echo "===== [$name] case=$case inner_thr=$thr nproc=$NPROC work=$work"
        ( cd "$work" && timeout 1800 mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1 )
        echo "----- [$name] exit=$?"
        # The framework audit lines live in OUT.autotest/running_scf.log; the
        # wall-clock footer lives in run.log.
        grep -E "CONSTRAINT_AUDIT|constraint\]|MIX_RESET|mixing recovered|!FINAL_ETOT_IS|!SCF IS NOT CONVERGED" \
            "$work/OUT.autotest/running_scf.log" > "$RESDIR/$name.audit"
        grep -E "TOTAL  Time|FINISH Time" "$work/run.log" >> "$RESDIR/$name.audit"
        grep -m1 "final status" "$work/OUT.autotest/running_scf.log" | sed 's/^/      /'
    done
done
echo "===== summary ====="
python3 "$CASEDIR/tools/extract_scan.py" "$WORKROOT" "$THRS" "$CASES"
