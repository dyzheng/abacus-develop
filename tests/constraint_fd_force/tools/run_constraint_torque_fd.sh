#!/bin/bash
# Constraint torque FD (Task 2.6 Step 3, DeltaSpin cross-check).
#
#   base: PW spin-constraint SCF at R0 (delta mode, +0.1 uB on O from the
#         testcase target).  The outer loop re-converges mu so that
#         m = m_free(R0) + 0.1 uB.  Extract:
#           - m* = frozen absolute magnetization target (first-audit t, uB)
#           - mu* = converged multiplier (Ry/uB)
#   legs: same geometry, targets m* +/- dM (absolute mode, nspin=2), mu
#         re-converged; energy E' = E_tot - mu*m* (Lagrangian, see below).
#   FD:   T_FD = (E'_{m*+dM} - E'_{m*-dM}) / (2 dM)  [eV/uB]
#   ana:  T_ana = -mu* * Ry_to_eV  (framework mu = -DeltaSpin lambda;
#         dL/dM = -mu by the envelope theorem on L = E_KS + mu(m-M)).
#   criterion: |T_FD - T_ana| < 0.006 eV/uB ("capability equal to DeltaSpin").
#   Grid prerequisites (R7): ecutwfc=100, ecutrho=400, scf_thr=1e-8.
#   Anti-fake-convergence check: delta=0 run must converge at mu ~ 0
#   immediately (natural target, no spurious fixed point).
#
# Usage: bash run_constraint_torque_fd.sh [dM_uB] [nproc]
set -uo pipefail

DM="${1:-0.05}"
NPROC="${2:-4}"
ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
ECUTWFC="${ECUTWFC:-100}"
ECUTRHO="${ECUTRHO:-400}"
SCF_THR="${SCF_THR:-1e-8}"
TESTCASE="/root/abacus-develop/tests/01_PW/212_PW_constraint_h2o_spin"
WORK="$(mktemp -d /tmp/cft_XXXX)"
RYTOEV=13.605693
CRIT=0.006
export OMP_NUM_THREADS=1

echo "===== Constraint torque FD (spin channel, PW) dM=${DM} uB ====="
echo "  criterion |T_FD - T_ana| < ${CRIT} eV/uB  (work: ${WORK})"

write_input() { # out_dir suffix mode target_file
    local out="$1" suf="$2" mode="$3" tfile="$4"
    {
        echo "INPUT_PARAMETERS"
        echo "suffix      ${suf}"
        echo "calculation scf"
        echo "basis_type  pw"
        echo "ecutwfc     ${ECUTWFC}"
        echo "ecutrho     ${ECUTRHO}"
        echo "scf_thr     ${SCF_THR}"
        echo "scf_nmax    300"
        echo "nbands      8"
        echo "nspin       2"
        echo "symmetry    0"
        echo "init_wfc    atomic"
        echo "init_chg    atomic"
        echo "nelec       8"
        echo "smearing_method gauss"
        echo "smearing_sigma  0.002"
        echo "mixing_type     broyden"
        echo "mixing_beta     0.4"
        echo "pseudo_dir  /root/abacus-develop/tests/PP_ORB"
        echo "constraint        true"
        echo "constraint_type   spin"
        echo "constraint_weight_type becke"
        echo "constraint_target_file ${tfile}"
        echo "constraint_target_mode ${mode}"
        echo "constraint_mu_max  5.0"
        echo "constraint_thr     1e-4"
    } > "$out"
}

run_scf() { # dir suffix -> E (eV)
    local dir="$1" suf="$2"
    ( cd "$dir" && mpirun --allow-run-as-root -np ${NPROC} "$ABACUS" > run.log 2>&1 )
    grep -h '!FINAL_ETOT_IS' "$dir"/OUT.${suf}/running_*.log 2>/dev/null | tail -1 | awk '{print $(NF-1)}'
}

extract_tstar() {
    python3 - "$1" <<'PYEOF'
import sys, re
for l in open(sys.argv[1]):
    if 'CONSTRAINT_AUDIT' in l and ' c[' in l:
        mm = re.search(r'\bt=(-?[0-9.eE+-]+)', l)
        if mm: print(mm.group(1)); sys.exit(0)
sys.exit(1)
PYEOF
}

extract_mu() {
    python3 - "$1" <<'PYEOF'
import sys, re
m = None
for l in open(sys.argv[1]):
    if 'CONSTRAINT_AUDIT' in l and 'mu=' in l: m = l
if m is None: sys.exit(1)
mm = re.search(r'mu=(-?[0-9.eE+-]+)', m)
print(mm.group(1))
PYEOF
}

extract_steps() {
    python3 - "$1" <<'PYEOF'
import sys, re
n = 0
for l in open(sys.argv[1]):
    if '[constraint] outer step' in l: n += 1
print(n)
PYEOF
}

extract_status() {
    python3 - "$1" <<'PYEOF'
import sys
for l in open(sys.argv[1]):
    if '[constraint] final status:' in l:
        print(l.split(':')[-1].strip()); sys.exit(0)
sys.exit(1)
PYEOF
}

# ---- base (delta mode, +0.1 uB on O)
mkdir -p "${WORK}/base"
cp "${TESTCASE}/STRU" "${TESTCASE}/KPT" "${TESTCASE}/constraint_target.json" "${WORK}/base/"
write_input "${WORK}/base/INPUT" base delta constraint_target.json
E0=$(run_scf "${WORK}/base" base)
[ -n "$E0" ] || { echo "  [base] FAILED"; tail -5 "${WORK}/base/run.log"; exit 1; }
MSTAR=$(extract_tstar "${WORK}/base/OUT.base"/running_*.log) || { echo "  !! m* extraction failed"; exit 1; }
MU0=$(extract_mu "${WORK}/base/OUT.base"/running_*.log)
ST0=$(extract_steps "${WORK}/base/OUT.base"/running_*.log)
echo "  [base] E0=${E0} eV; frozen magnetization target m*=${MSTAR} uB; mu*=${MU0} Ry; outer steps=${ST0}"

# ---- torque legs: targets m* +/- dM (absolute mode)
run_leg() { # suffix target -> "E mu Eprime"
    local suf="$1" tgt="$2"
    local step="${WORK}/leg_${suf}"
    mkdir -p "$step"
    cp "${TESTCASE}/STRU" "${TESTCASE}/KPT" "$step/"
    printf '{"targets": [%s], "atoms": [[0]]}\n' "$tgt" > "$step/constraint_target.json"
    write_input "$step/INPUT" "$suf" absolute constraint_target.json
    E=$(run_scf "$step" "$suf")
    MU=$(extract_mu "$step/OUT.${suf}"/running_*.log 2>/dev/null || echo NA)
    if [ -n "$E" ] && [ -n "$MU" ] && [ "$MU" != "NA" ]; then
        EP=$(python3 -c "print(${E} - ${MU} * ${tgt} * ${RYTOEV})")
    else
        EP=NA
    fi
    echo "${E} ${MU} ${EP}"
}

read EP_hi MU_hi EPRIME_hi < <(run_leg hi "$(python3 -c "print(${MSTAR} + ${DM})")")
read EP_lo MU_lo EPRIME_lo < <(run_leg lo "$(python3 -c "print(${MSTAR} - ${DM})")")
echo "  [+dM] E=${EP_hi} mu=${MU_hi} E'=${EPRIME_hi}"
echo "  [-dM] E=${EP_lo} mu=${MU_lo} E'=${EPRIME_lo}"
if [ "$EPRIME_hi" = "NA" ] || [ "$EPRIME_lo" = "NA" ]; then echo "  !! leg FAILED"; exit 1; fi
TFD=$(python3 -c "print((${EPRIME_hi} - ${EPRIME_lo}) / (2 * ${DM}))")
TANA=$(python3 -c "print(-${MU0} * ${RYTOEV})")
RES=$(python3 -c "print(abs(${TFD} - ${TANA}))")
PAS=$(python3 -c "print('PASS' if ${RES} < ${CRIT} else 'FAIL')")
echo "  T_FD = ${TFD} eV/uB ; T_ana = -mu* = ${TANA} eV/uB ; |d| = ${RES} ${PAS}"

# ---- anti-fake-convergence: delta = 0 must converge at mu ~ 0
mkdir -p "${WORK}/delta0"
cp "${TESTCASE}/STRU" "${TESTCASE}/KPT" "$WORK/delta0/"
printf '{"targets": [0.0], "atoms": [[0]]}\n' > "$WORK/delta0/constraint_target.json"
write_input "$WORK/delta0/INPUT" d0 delta constraint_target.json
ED0=$(run_scf "$WORK/delta0" d0)
MUD0=$(extract_mu "$WORK/delta0/OUT.d0"/running_*.log 2>/dev/null || echo NA)
STD0=$(extract_steps "$WORK/delta0/OUT.d0"/running_*.log 2>/dev/null || echo NA)
STAD0=$(extract_status "$WORK/delta0/OUT.d0"/running_*.log 2>/dev/null || echo NA)
echo "  [delta=0] E=${ED0} mu=${MUD0} outer steps=${STD0} status=${STAD0}"

echo ""
echo "===== torque summary ====="
echo "  T_FD=${TFD} T_ana=${TANA} |d|=${RES} criterion=${CRIT} -> ${PAS}"
echo "  anti-fake delta=0: mu=${MUD0} steps=${STD0} status=${STAD0} (expect mu~0, 1 step, CONVERGED)"
echo "  work dir = ${WORK}"
