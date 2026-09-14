#!/bin/bash
# III-1: charge-transfer pair + CDFT/Marcus interface -- H2O dimer scan.
#
# Question (validation-case-suite-design.md section 3): does the framework's
# flagship CDFT interface work on a real donor-acceptor pair -- i.e. can two
# fragment charge constraints (mixed, v2 target list) converge simultaneously,
# and are the constraint work W = sum(mu_i dQ_i) and the forward/backward mu
# difference (Marcus reorganization-energy input) self-consistent?
#
# System: H2O dimer (acceptor O1+2H, donor O2+2H; O1...H = 2.02 A hydrogen
# bond), LCAO, gamma-only, 15 A box, nelec 20.
# Constraint: v2 list, charge +d on the acceptor fragment [0,2,3] and -d on
# the donor fragment [1,4,5]  => net-zero charge transfer of d electrons.
# Scan: DELTAS (default +-0.05/0.10/0.15/0.20 e), delta target mode.
# All runs: same binary/grid/initial guess, np=4, OMP_NUM_THREADS=1.
#
# Usage: bash run_ct_scan.sh [nproc]
set -uo pipefail

NPROC="${1:-4}"
ABACUS="${ABACUS:-/root/abacus-develop/build_rel/abacus_basic_para}"
CASEDIR="$(cd "$(dirname "$0")" && pwd)"
CASE="$CASEDIR/h2o_dimer"
PPDIR="$(cd "$CASEDIR/../PP_ORB" && pwd)"
WORKROOT="${WORKROOT:-/tmp/ct_dimer}"
RESDIR="$CASEDIR/results"
DELTAS="${DELTAS:-0.05 -0.05 0.10 -0.10 0.15 -0.15 0.20 -0.20}"
export OMP_NUM_THREADS=1
mkdir -p "$WORKROOT" "$RESDIR"

for d in $DELTAS; do
    tag="$(python3 -c "d=float('$d'); s='m' if d<0 else 'p'; print('d_%s%.2f' % (s, abs(d)))")"
    work="$WORKROOT/$tag"
    # Branch: fresh work dir per point -- every point is a cold start, so the
    # only varying quantity across the scan is the charge-transfer amplitude.
    python3 -c "import shutil,os; d='$work'; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)"
    cp "$CASE/INPUT" "$CASE/KPT" "$CASE/STRU" "$work/"
    # The case INPUT carries the relative ../../PP_ORB paths; the work dir is
    # under /tmp, so pin absolute paths before running.
    sed -i "s|^\(pseudo_dir\).*|\1 $PPDIR|" "$work/INPUT"
    sed -i "s|^\(orbital_dir\).*|\1 $PPDIR|" "$work/INPUT"
    python3 -c "
import json
d = float('$d')
json.dump({'constraints': [
    {'type': 'charge', 'target':   d, 'atoms': [0, 2, 3]},
    {'type': 'charge', 'target': -d, 'atoms': [1, 4, 5]}]},
    open('$work/constraint_target.json', 'w'))
"
    echo "===== [$tag] delta=$d e nproc=$NPROC work=$work"
    ( cd "$work" && timeout 3600 mpirun -np "$NPROC" "$ABACUS" > run.log 2>&1 )
    rc=$?
    echo "----- [$tag] exit=$rc"
    # Keep only the audit-relevant lines: the CONSTRAINT_AUDIT blocks, the
    # framework notices, the final energy and the wall-clock footer.
    grep -E "CONSTRAINT_AUDIT|constraint\]|MIX_RESET|mixing recovered|!FINAL_ETOT_IS|!SCF IS NOT CONVERGED" \
        "$work/OUT.autotest/running_scf.log" > "$RESDIR/$tag.audit" 2>/dev/null
    grep -E "TOTAL  Time|FINISH Time" "$work/run.log" >> "$RESDIR/$tag.audit" 2>/dev/null
    grep -m1 "final status" "$work/OUT.autotest/running_scf.log" 2>/dev/null | sed 's/^/      /'
done
echo "===== summary ====="
python3 "$CASEDIR/tools/extract_scan.py" "$RESDIR"
