#!/bin/bash
# Constraint force FD (Task 2.6 Step 2, stationary4 protocol).
#
#   base: constrained SCF at R0 (delta mode, target delta from the testcase
#         constraint_target.json).  The outer loop re-converges mu so that
#         Q = Q_free(R0) + delta.  Extract:
#           - t* = the frozen absolute target = first-audit t (Q_free(R0)+delta)
#           - F_ana = analytic TOTAL-FORCE (eV/A) incl. the constraint force
#   legs: per atom x axis x +-delta (delta_bohr default 0.005), constrained
#         SCF with the SAME frozen target t* (absolute mode), mu re-converged;
#         start from the base's converged charge density (restart file).
#         Freezing t* is mandatory (stationary4): with delta-mode targets the
#         reference Q_ref would drift with R and the FD would be polluted by
#         mu*dQ_free/dR (O(eV/A) systematic error).
#   FD:   F_FD = -(E_+ - E_-) / (2 delta_A) with the RAW printed E_tot as the
#         observable (Task 2.6 attribution, 2026-09-07 spec section 1: the
#         printed E_tot = E_KS_phys(rho_mu) + cc_escon does NOT include mu*Q,
#         so E' = E_tot - mu*t* is WRONG here — it reintroduces the
#         t*dmu*/dR envelope pseudo-term, O(eV/A) at dmu/dR ~ 1 Ry/A).
#   criterion: |F_FD - F_ana| < CRIT (5e-4 Ry/Bohr = 0.0128555 eV/A).
#   Grid prerequisites (R7): ecutwfc=100, ecutrho=400, scf_thr=1e-8.
#
# Channels (V1): the case directory defaults to the historical charge cases
#   (PW 211 / LCAO 212_NAO) and is selected with CASE=/path/to/case.  nspin,
#   constraint_type and nelec are inherited from that case's INPUT, so the
#   spin case (tests/01_PW/212_PW_constraint_h2o_spin: nspin=2,
#   constraint_type spin, mag 0.5 on O) reuses the identical leg protocol.
#
# Usage: bash run_constraint_fd.sh <pw|lcao> [delta_bohr] [nproc] [max_jobs]
#   ONLY="iat_axis"  (e.g. ONLY="0_2") runs a single atom-axis pair (smoke).
#   CASE=<dir>       override the testcase directory (V1: spin case).
#   TEST_FORCE=1     add "test_force 1" so the run prints the per-term force
#                    decomposition (2.7 standard checks: net force before
#                    compensation + constraint-force sum).  Off by default.
set -uo pipefail

BASIS="${1:-lcao}"
DELTA_BOHR="${2:-0.005}"
NPROC="${3:-4}"
MAXJOBS="${4:-2}"
ONLY="${ONLY:-}"
ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
ECUTWFC="${ECUTWFC:-100}"
ECUTRHO="${ECUTRHO:-400}"
SCF_THR="${SCF_THR:-1e-8}"
SRCDIR="$(cd "$(dirname "$0")/../.." && pwd)/01_PW"
if [ "$BASIS" = "lcao" ]; then
  SRCDIR="$(cd "$(dirname "$0")/../.." && pwd)/02_NAO_Gamma"
fi
# Branch: the LCAO default is its own charge-channel case.
DEFAULT_CASE="$SRCDIR/211_PW_constraint_h2o"
if [ "$BASIS" = "lcao" ]; then
  DEFAULT_CASE="$SRCDIR/212_NAO_constraint_h2o"
fi
TESTCASE="${CASE:-$DEFAULT_CASE}"
if [ ! -f "${TESTCASE}/INPUT" ]; then
  echo "!! testcase INPUT not found: ${TESTCASE}/INPUT"; exit 1
fi
# Channel parameters are inherited from the case INPUT (nspin=2 +
# constraint_type spin for the spin case).  Absent tokens fall back to the
# historical charge-channel values, so the 211 / 212_NAO INPUT files stay
# byte-identical to the pre-V1 script.
CASE_NSPIN=$(awk '$1 == "nspin" {print $2; exit}' "${TESTCASE}/INPUT")
CASE_NSPIN="${CASE_NSPIN:-1}"
CASE_CTYPE=$(awk '$1 == "constraint_type" {print $2; exit}' "${TESTCASE}/INPUT")
CASE_CTYPE="${CASE_CTYPE:-charge}"
CASE_NELE=$(awk '$1 == "nelec" {print $2; exit}' "${TESTCASE}/INPUT")
CASE_NELE="${CASE_NELE:-8}"
TEST_FORCE="${TEST_FORCE:-0}"
WORK="$(mktemp -d /tmp/cfd_${BASIS}_XXXX)"
DELTA_A=$(python3 -c "print(${DELTA_BOHR} * 0.5291772109)")
CRIT=$(python3 -c "print(5e-4 * 13.605693 / 0.5291772109)")  # 0.0128555 eV/A
RYTOEV=13.605693
export OMP_NUM_THREADS=1

echo "===== Constraint FD: basis=${BASIS} delta=${DELTA_BOHR} Bohr ====="
echo "  criterion |F_FD - F_ana| < ${CRIT} eV/A  (work: ${WORK})"

# ---------- STRU parsing ----------
parse_stru() {
python3 - "$1" <<'PYEOF'
import sys
lines = open(sys.argv[1]).read().splitlines()
coords = []; in_pos = False; i = 0
while i < len(lines):
    l = lines[i].strip()
    if l.startswith('ATOMIC_POSITIONS'):
        in_pos = True; mode = lines[i+1].strip(); i += 2; continue
    if not in_pos: i += 1; continue
    if l == '' or l.startswith('#'): i += 1; continue
    toks = l.split()
    if len(toks) == 1 and toks[0][0].isalpha():
        sp = toks[0]; i += 1; i += 1  # mag
        n = int(lines[i].strip()); i += 1
        for _ in range(n):
            c = lines[i].split(); coords.append((sp, [float(x) for x in c[:3]])); i += 1
        continue
    i += 1
print(len(coords), mode)
for sp, c in coords: print(sp, ' '.join(f'{x:.10f}' for x in c))
PYEOF
}

displace_stru() {
python3 - "$1" "$2" "$3" "$4" "$5" "$6" <<'PYEOF'
import sys
src, dst, iat, axis, delta, mode = sys.argv[1:7]
iat, axis, delta = int(iat), int(axis), float(delta)
lines = open(src).read().splitlines(keepends=True)
out = []; in_pos = False; count = -1; i = 0
while i < len(lines):
    l = lines[i]
    if l.startswith('ATOMIC_POSITIONS'):
        in_pos = True; out.append(l); i += 1
        out.append(lines[i]); i += 1; continue
    if not in_pos: out.append(l); i += 1; continue
    if l.strip() == '' or l[0] == '#': out.append(l); i += 1; continue
    toks = l.split()
    if len(toks) == 1 and toks[0][0].isalpha():
        out.append(l); i += 1; out.append(lines[i]); i += 1
        n = int(lines[i].strip()); i += 1; out.append(f'{n}\n')
        for _ in range(n):
            c = [float(x) for x in lines[i].split()[:3]]; count += 1
            if count == iat:
                c[axis] += delta  # Cartesian_angstrom mode only
            extra = ' ' + ' '.join(lines[i].split()[3:]) if len(lines[i].split()) > 3 else ''
            out.append(' '.join(f'{x:.10f}' for x in c) + extra + '\n'); i += 1
        continue
    out.append(l); i += 1
open(dst, 'w').writelines(out)
PYEOF
}

write_input() { # out_dir suffix mode target_file read_from
    local out="$1" suf="$2" mode="$3" tfile="$4" read_from="$5"
    {
        echo "INPUT_PARAMETERS"
        echo "suffix      ${suf}"
        echo "calculation scf"
        echo "basis_type  ${BASIS}"
        echo "ecutwfc     ${ECUTWFC}"
        echo "ecutrho     ${ECUTRHO}"
        echo "scf_thr     ${SCF_THR}"
        echo "scf_nmax    300"
        echo "nbands      8"
        echo "symmetry    0"
        echo "init_wfc    atomic"
        if [ "$read_from" = "/dev/null" ]; then
            echo "init_chg    atomic"
        else
            echo "init_chg    file"
            echo "read_file_dir ${read_from}"
        fi
        if [ "$BASIS" = "lcao" ]; then
            echo "gamma_only  1"
            echo "orbital_dir /root/abacus-develop/tests/PP_ORB"
        fi
        # Branch: spin-polarized runs must declare nspin; the charge cases
        # keep the nspin=1 default and their INPUT stays unchanged.
        if [ "$CASE_NSPIN" != "1" ]; then
            echo "nspin       ${CASE_NSPIN}"
        fi
        echo "nelec       ${CASE_NELE}"
        echo "smearing_method gauss"
        echo "smearing_sigma  0.002"
        echo "mixing_type     broyden"
        echo "mixing_beta     0.4"
        echo "pseudo_dir  /root/abacus-develop/tests/PP_ORB"
        echo "constraint        true"
        echo "constraint_type   ${CASE_CTYPE}"
        echo "constraint_weight_type becke"
        echo "constraint_target_file ${tfile}"
        echo "constraint_target_mode ${mode}"
        echo "constraint_mu_max  5.0"
        echo "constraint_thr     1e-4"
        echo "cal_force       1"
        # Branch: optional per-term force dump, needed by the 2.7 standard
        # checks (pre-compensation net force, constraint-force sum).
        if [ "$TEST_FORCE" = "1" ]; then
            echo "test_force      1"
        fi
    } > "$out"
}

run_scf() { # dir suffix -> E (eV)
    local dir="$1" suf="$2"
    ( cd "$dir" && mpirun --allow-run-as-root -np ${NPROC} "$ABACUS" > run.log 2>&1 )
    grep -h '!FINAL_ETOT_IS' "$dir"/OUT.${suf}/running_*.log 2>/dev/null | tail -1 | awk '{print $(NF-1)}'
}

extract_force() {
    python3 - "$1" "$2" <<'PYEOF'
import sys, re
nat = int(sys.argv[2])
txt = open(sys.argv[1]).read()
m = None
for tag in ['#TOTAL-FORCE (eV/Angstrom)#', '#TOTAL-FORCE#']:
    m = re.search(re.escape(tag) + r'\n(.*?)\n\s*\n', txt, re.S)
    if m: break
if not m: sys.exit(1)
rows = []
for l in m.group(1).splitlines():
    t = l.split()
    if len(t) == 4 and t[0][0].isalpha() and t[0] != 'Atoms':
        rows += [float(t[1]), float(t[2]), float(t[3])]
if len(rows) != nat * 3: sys.exit(1)
print('\n'.join(f'{x:.10e}' for x in rows))
PYEOF
}

# First CONSTRAINT_AUDIT detail line: q = Q_ref, t = frozen target t*.
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

# 2.7 standard checks: sum the per-term force blocks of one running_*.log.
#   - Sigma CONSTRAINT force per axis (translation invariance: must be ~0)
#   - Sigma pre-compensation total force per axis (= the net force the code
#     removes via compen; a non-zero value here is the fingerprint of a
#     missing Pulay-type term)
#   - compen per axis and the consistency max|pre - compen - printed|
# Requires test_force=1 in the leg inputs (TEST_FORCE=1).
std_checks() { # log nat
    python3 - "$1" "$2" <<'PYEOF_CHECK'
import sys, re
log, nat = sys.argv[1], int(sys.argv[2])
txt = open(log).read()
RYBOHR2EVA = 13.605693 / 0.5291772109

def blk(name):
    m = re.search(re.escape('#' + name + '#') + r'\n(.*?)\n\s*\n', txt, re.S)
    if not m:
        return None
    rows = []
    for l in m.group(1).splitlines():
        t = l.split()
        if len(t) == 4 and t[0][0].isalpha() and t[0] != 'Atoms':
            rows.append([float(x) for x in t[1:4]])
    return rows if len(rows) == nat else None

terms = ['LOCAL    FORCE (eV/Angstrom)', 'NONLOCAL FORCE (eV/Angstrom)',
         'NLCC     FORCE (eV/Angstrom)', 'ION      FORCE (eV/Angstrom)',
         'SCC      FORCE (eV/Angstrom)']
pre = [[0.0] * 3 for _ in range(nat)]
cons = blk('CONSTRAINT  FORCE (Ry/Bohr)')
if cons is None:
    sys.exit('missing CONSTRAINT block (run with TEST_FORCE=1)')
for i in range(nat):
    for a in range(3):
        pre[i][a] += cons[i][a] * RYBOHR2EVA
for nm in terms:
    b = blk(nm)
    if b is None:
        sys.exit('missing block: ' + nm)
    for i in range(nat):
        for a in range(3):
            pre[i][a] += b[i][a]
tot = blk('TOTAL-FORCE (eV/Angstrom)')
if tot is None:
    sys.exit('missing TOTAL block')
sumcons = [sum(cons[i][a] for i in range(nat)) * RYBOHR2EVA for a in range(3)]
sumpre = [sum(pre[i][a] for i in range(nat)) for a in range(3)]
compen = [x / nat for x in sumpre]
sumtot = [sum(tot[i][a] for i in range(nat)) for a in range(3)]
dev = max(abs(pre[i][a] - compen[a] - tot[i][a]) for i in range(nat) for a in range(3))
print('  [std-check] Sigma CONSTRAINT force (eV/A)   : ' + ' '.join('%+.6f' % x for x in sumcons))
print('  [std-check] Sigma pre-compensation (eV/A)   : ' + ' '.join('%+.6f' % x for x in sumpre))
print('  [std-check] compen (eV/A)                   : ' + ' '.join('%+.6f' % x for x in compen))
print('  [std-check] Sigma printed TOTAL (eV/A)      : ' + ' '.join('%+.6f' % x for x in sumtot))
print('  [std-check] max|pre - compen - printed|     : %.3e eV/A' % dev)
PYEOF_CHECK
}

# Last CONSTRAINT_AUDIT detail line: final converged mu (Ry).
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

read NAT MODE < <(parse_stru "${TESTCASE}/STRU")
NAT_LEG="$NAT"
echo "  atoms=${NAT} (${MODE})"
mkdir -p "${WORK}/base"
cp "${TESTCASE}/STRU" "${TESTCASE}/KPT" "${TESTCASE}/constraint_target.json" "${WORK}/base/"
write_input "${WORK}/base/INPUT" base delta constraint_target.json /dev/null

E0=$(run_scf "${WORK}/base" base)
if [ -z "$E0" ]; then echo "  [base] FAILED"; tail -5 "${WORK}/base/run.log"; exit 1; fi
TSTAR=$(extract_tstar "${WORK}/base/OUT.base"/running_*.log) || { echo "  !! t* extraction failed"; exit 1; }
echo "  [base] E0=${E0} eV; frozen absolute target t*=${TSTAR} e"
MU0=$(extract_mu "${WORK}/base/OUT.base"/running_*.log 2>/dev/null || echo NA)
F0LOG=$(ls "${WORK}"/base/OUT.base/running_*.log | head -1)
mapfile -t F0 < <(extract_force "$F0LOG" "$NAT") || { echo "  !! force extraction failed"; exit 1; }
if [ "$TEST_FORCE" = "1" ]; then
    std_checks "$F0LOG" "$NAT"
fi
echo "  [base] mu*=${MU0} Ry; analytic forces (eV/A):"
for i in $(seq 0 $((NAT-1))); do
  printf '    atom %d: %s %s %s\n' "$i" "${F0[$((i*3))]}" "${F0[$((i*3+1))]}" "${F0[$((i*3+2))]}"
done

# Legs restart from the base's converged density.  init_chg file reads
# {read_file_dir}/{suffix}-CHARGE-DENSITY.restart, so stage the base restart
# under the leg suffix name in a dedicated restart dir.
mkdir -p "${WORK}/restart"
cp "${WORK}/base/OUT.base/base-CHARGE-DENSITY.restart" \
   "${WORK}/restart/s-CHARGE-DENSITY.restart"
cp "${TESTCASE}/KPT" "${WORK}/restart/"

# ---- stationary legs (one sign per leg; concurrent, max MAXJOBS at a time)
declare -a LEGS
for iat in $(seq 0 $((NAT-1))); do
  for axis in 0 1 2; do
    if [ -n "$ONLY" ] && [ "$ONLY" != "${iat}_${axis}" ]; then continue; fi
    for sign in plus minus; do
      LEGS+=("${iat}_${axis}_${sign}")
    done
  done
done
echo "  [legs] ${#LEGS[@]} legs, max ${MAXJOBS} concurrent..."

run_leg() { # leg -> "iat axis sign E mu"
    local leg="$1"
    local iat="${leg%%_*}"; local rest="${leg#*_}"
    local axis="${rest%%_*}"; local sign="${rest##*_}"
    local step="${WORK}/disp_s_${leg}"
    mkdir -p "$step"
    cp "${TESTCASE}/KPT" "$step/"
    # Frozen target: absolute mode with t* (same number for plus and minus).
    printf '{"targets": [%s], "atoms": [[0]]}\n' "$TSTAR" > "$step/constraint_target.json"
    local SD
    if [ "$sign" = plus ]; then SD="$DELTA_A"; else SD="-$DELTA_A"; fi
    displace_stru "${TESTCASE}/STRU" "$step/STRU" "$iat" "$axis" "$SD" "$MODE"
    write_input "$step/INPUT" s absolute constraint_target.json "${WORK}/restart"
    E=$(run_scf "$step" s)
    MU=$(extract_mu "$step/OUT.s"/running_*.log 2>/dev/null || echo NA)
    echo "${iat} ${axis} ${sign} ${E:-NA} ${MU}"
    # Branch: with TEST_FORCE=1 also dump the per-leg standard checks; the
    # extra lines do not match the aggregator's "iat axis sign E mu" pattern.
    if [ "$TEST_FORCE" = "1" ]; then
        std_checks "$step/OUT.s"/running_*.log "$NAT_LEG"
    fi
}

declare -a ROWS
running=0
for leg in "${LEGS[@]}"; do
  ( run_leg "$leg" > "${WORK}/leg_${leg}.out" 2>&1 ) &
  running=$((running + 1))
  if [ "$running" -ge "$MAXJOBS" ]; then
    wait -n
    running=$((running - 1))
  fi
done
while [ "$running" -gt 0 ]; do
  wait -n
  running=$((running - 1))
done

# ---- aggregate: FD of the raw FINAL_ETOT (see header note), central diff
for iat in $(seq 0 $((NAT-1))); do
  for axis in 0 1 2; do
    [ -n "$ONLY" ] && [ "$ONLY" != "${iat}_${axis}" ] && continue
    P=$(awk -v i="$iat" -v a="$axis" '$1==i && $2==a && $3=="plus" {print $4}' "${WORK}"/leg_${iat}_${axis}_plus.out)
    M=$(awk -v i="$iat" -v a="$axis" '$1==i && $2==a && $3=="minus" {print $4}' "${WORK}"/leg_${iat}_${axis}_minus.out)
    if [ -z "$P" ] || [ -z "$M" ] || [ "$P" = "NA" ] || [ "$M" = "NA" ]; then
      echo "  atom ${iat} axis ${axis}: RUN FAILED (E+=${P:-NA} E-=${M:-NA})"
      continue
    fi
    FFD=$(python3 -c "print(-(${P} - ${M}) / (2 * ${DELTA_A}))")
    FAN="${F0[$((iat*3+axis))]}"
    RES=$(python3 -c "print(abs(${FFD} - ${FAN}))")
    PAS=$(python3 -c "print('PASS' if ${RES} < ${CRIT} else 'FAIL')")
    ROWS+=("${iat} ${axis} ${FFD} ${FAN} ${RES} ${PAS}")
    echo "  atom ${iat} axis ${axis}: F_FD=${FFD} F_ana=${FAN} |d|=${RES} ${PAS}"
  done
done

echo ""
echo "===== ${BASIS} summary (raw FINAL_ETOT central diff, frozen t*=${TSTAR}) ====="
printf '%-4s %-4s %-16s %-16s %-16s %s\n' iat axis F_FD F_ana RES PASS
printf '%s\n' "${ROWS[@]}"
echo "criterion = ${CRIT} eV/A  (5e-4 Ry/Bohr);  work dir = ${WORK}"
