#!/bin/bash
# DeltaP FD force validation — two-group protocol (T7-b)
# ============================================================
# Differential object: E' = FINAL_ETOT_IS (already includes dp_escon,
# see source/source_estate/fp_energy.cpp calculate_etot()).
#
# Group 1 (frozen lambda):  displace atom ±δ, freeze λ at the base-run
#   converged value (deltap_lambda_init_file + deltap_lambda_step 0.0),
#   re-converge SCF.  FD force vs analytic TOTAL-FORCE.
#   Expected residual ≈ A2 (∂τ/∂R) + C (λ·dγ/dR) + B (H_HK force, missing).
# Group 2 (re-converged λ):  same displacements but let λ re-converge at
#   each geometry (deltap_lambda_step > 0).  This is the relax-usable force.
#
# Criterion: |F_FD − F_analytic| < 5e-4 Ry/Bohr = 0.0128555 eV/Å
#
# Usage: bash run_fd.sh [system=h2o1] [delta_bohr=0.005] [nproc=1] [group=both]
#   group: both | 1 | 2
# Prerequisites: ABACUS binary (ABACUS=... or build path), python3,
#   pseudopotentials/orbitals at /root/pporb/apns-* (paths in INPUT).

set -uo pipefail

SYSTEM="${1:-h2o1}"
DELTA_BOHR="${2:-0.005}"
NPROC="${3:-1}"
GROUP="${4:-both}"
ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
TESTDIR="${BASEDIR}/${SYSTEM}"

# δ in Å (STRU coordinates are Cartesian_angstrom; Direct converted inside)
DELTA_A=$(python3 -c "print(${DELTA_BOHR} * 0.5291772109)")
# criterion in eV/Å (5e-4 Ry/Bohr)
CRIT_EV_A=$(python3 -c "print(5e-4 * 13.605693 / 0.5291772109)")
BOHR_TO_A=0.5291772109
FAILED=0

# Serial LCAO runs hang in the FFTW OMP thread pool (recip2real barrier
# spin) on this environment; force single-threaded FFTW for all runs.
export OMP_NUM_THREADS=1

echo "===== DeltaP FD Force Validation (two-group) ====="
echo "  System:  ${SYSTEM}   Delta: ${DELTA_BOHR} Bohr = ${DELTA_A} Å"
echo "  Nproc:   ${NPROC}   Group: ${GROUP}"
echo "  Criterion: |F_FD - F_ana| < ${CRIT_EV_A} eV/Å (5e-4 Ry/Bohr)"
echo "  ABACUS:  ${ABACUS}"
echo ""

# ---------- STRU parsing helpers (python3) ----------
# parse_stru: print "nat" then "species count" lines and coordinate lines
parse_stru() {
python3 - "$1" <<'PYEOF'
import sys
lines = open(sys.argv[1]).readlines()
coords = []
cur = None
in_pos = False
mode = None
nat = 0
i = 0
while i < len(lines):
    l = lines[i].strip()
    if l.startswith('ATOMIC_POSITIONS'):
        in_pos = True
        mode = lines[i+1].strip()
        i += 2
        continue
    if not in_pos:
        i += 1
        continue
    if l == '':
        i += 1
        continue
    # species name line
    if len(l.split()) == 1 and l[0].isalpha():
        sp = l
        # magnetic moment line
        i += 1
        mag = lines[i].strip()
        i += 1
        n = int(lines[i].strip())
        i += 1
        for _ in range(n):
            c = lines[i].split()
            coords.append((sp, [float(x) for x in c[:3]], len(c) > 3))
            i += 1
        continue
    i += 1
print(len(coords), mode)
for sp, c, has_move in coords:
    print(sp, ' '.join(f'{x:.10f}' for x in c), '1' if has_move else '')
PYEOF
}

# displace_stru <stru_in> <stru_out> <iat> <axis> <delta_angstrom>
# Handles Cartesian_angstrom (h2o1) and Direct (bn, via lattice matrix).
displace_stru() {
python3 - "$1" "$2" "$3" "$4" "$5" <<'PYEOF'
import sys
src, dst, iat, axis, delta = sys.argv[1:6]
iat, axis, delta = int(iat), int(axis), float(delta)
text = open(src).read()
lines = text.splitlines(keepends=True)
# lattice: LATTICE_CONSTANT (Bohr per unit) and LATTICE_VECTORS (in lat0)
lat0 = 1.0
latv = None
for k, l in enumerate(lines):
    if l.startswith('LATTICE_CONSTANT'):
        lat0 = float(lines[k+1].split()[0])
    if l.startswith('LATTICE_VECTORS'):
        v = []
        for j in range(k+1, k+4):
            v.append([float(x) for x in lines[j].split()[:3]])
        latv = v
        break
out = []
in_pos = False
mode = None
count = -1
i = 0
while i < len(lines):
    l = lines[i]
    if l.startswith('ATOMIC_POSITIONS'):
        in_pos = True
        mode = lines[i+1].strip()
        out.append(l); i += 1
        out.append(lines[i]); i += 1
        continue
    if not in_pos:
        out.append(l); i += 1
        continue
    if l.strip() == '' or l[0] == '#':
        out.append(l); i += 1
        continue
    toks = l.split()
    if len(toks) == 1 and toks[0][0].isalpha():
        out.append(l); i += 1
        out.append(lines[i]); i += 1
        n = int(lines[i].strip()); i += 1
        out.append(f'{n}\n')
        for _ in range(n):
            c = [float(x) for x in lines[i].split()[:3]]
            count += 1
            if count == iat:
                if mode == 'Direct':
                    # cart_Å = frac · latv · lat0 ; displace; frac' = cart · latv^-1 / lat0
                    A = latv
                    cart = [sum(c[k]*A[k][j] for k in range(3))*lat0*0.5291772109 for j in range(3)]
                    cart[axis] += delta
                    # invert via numpy-free 3x3
                    def det3(m):
                        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                                -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                                +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
                    def inv3(m):
                        d = det3(m)
                        return [[(m[1][1]*m[2][2]-m[1][2]*m[2][1])/d,
                                 (m[0][2]*m[2][1]-m[0][1]*m[2][2])/d,
                                 (m[0][1]*m[1][2]-m[0][2]*m[1][1])/d],
                                [(m[1][2]*m[2][0]-m[1][0]*m[2][2])/d,
                                 (m[0][0]*m[2][2]-m[0][2]*m[2][0])/d,
                                 (m[0][2]*m[1][0]-m[0][0]*m[1][2])/d],
                                [(m[1][0]*m[2][1]-m[1][1]*m[2][0])/d,
                                 (m[0][1]*m[2][0]-m[0][0]*m[2][1])/d,
                                 (m[0][0]*m[1][1]-m[0][1]*m[1][0])/d]]
                    Ai = inv3(A)
                    c = [sum(cart[j]*Ai[j][k] for j in range(3))/(lat0*0.5291772109) for k in range(3)]
                else:
                    c[axis] += delta
            extra = ' ' + ' '.join(lines[i].split()[3:]) if len(lines[i].split()) > 3 else ''
            out.append(' '.join(f'{x:.10f}' for x in c) + extra + '\n')
            i += 1
        continue
    out.append(l); i += 1
open(dst, 'w').writelines(out)
PYEOF
}

# ---------- run one SCF and extract E', λ, forces ----------
# run_scf <workdir> <suffix> <nproc> -> prints "E LAM..." ; writes run.log
run_scf() {
    local dir="$1" suf="$2" np="$3"
    ( cd "$dir" && rm -rf "OUT.${suf}" run.log
      if [ "$np" -gt 1 ]; then
          mpirun -np "$np" "$ABACUS" > run.log 2>&1
      else
          "$ABACUS" > run.log 2>&1
      fi
    )
    local logf="$dir/OUT.${suf}/running_*.log"
    local E
    E=$(grep -h '!FINAL_ETOT_IS' $logf 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
    echo "$E"
}

# extract per-atom λ from last [DeltaP P3] line (12 atoms: 12 values)
extract_lambda() {
    python3 - "$1" <<'PYEOF'
import sys, re
line = None
for l in open(sys.argv[1]):
    if '[DeltaP P3]' in l and 'λ=' in l:
        line = l
if line is None:
    sys.exit(1)
m = re.search(r'λ=\(([^)]*)\)', line)
vals = [float(x) for x in m.group(1).split(',')]
for v in vals:
    print(f'{v:.9e}')
PYEOF
}

# extract TOTAL-FORCE table -> "fx fy fz" per atom
extract_force() {
    python3 - "$1" <<'PYEOF'
import sys, re
f = None
for path in sys.argv[1:]:
    try:
        txt = open(path).read()
    except OSError:
        continue
    m = re.search(r'#TOTAL-FORCE[^\n]*#\n(.*?)\n\s*\n', txt, re.S)
    if m:
        rows = []
        for l in m.group(1).splitlines():
            t = l.split()
            if len(t) == 4 and t[0][0].isalpha() and t[0] != 'Atoms':
                rows.append([float(t[1]), float(t[2]), float(t[3])])
        if rows:
            for r in rows:
                print(f'{r[0]:.9e} {r[1]:.9e} {r[2]:.9e}')
            sys.exit(0)
    break
sys.exit(1)
PYEOF
}

# ---------- build INPUT variants ----------
write_input() {  # $1=outfile $2=suffix $3=lambda_step $4=init_file $5=init_chg
    local out="$1" suf="$2" step="$3" initfile="$4" inchg="$5"
    {
        echo "INPUT_PARAMETERS"
        echo "suffix      ${suf}"
        echo "calculation scf"
        echo "basis_type  lcao"
        echo "ecutwfc     ${ECUTWFC:-50}"
        if [ -n "${ECUTRHO:-}" ]; then
            echo "ecutrho     ${ECUTRHO}"
        fi
        echo "gamma_only  0"
        echo "nspin       1"
        echo "scf_thr     ${SCF_THR:-1.0e-7}"
        echo "scf_nmax    100"
        echo "out_chg     0"
        echo "smearing_method gauss"
        echo "smearing_sigma  0.002"
        echo "mixing_type     broyden"
        echo "mixing_beta     0.4"
        echo "ks_solver       genelpa"
        echo "symmetry       0"
        echo "berry_phase     1"
        echo "deltap_switch   1"
        echo "deltap_method   wannier"
        echo "deltap_rm       3.0"
        echo "deltap_gdir     3"
        echo "deltap_corr     1"
        echo "deltap_lambda_step  ${step}"
        echo "deltap_lambda_init  0.0"
        if [ -n "$initfile" ]; then
            echo "deltap_lambda_init_file ${initfile}"
        fi
        echo "deltap_inner_thr    1.0e-3"
        echo "deltap_lambda_mixing 0.1"
        echo "deltap_target_file   target.dat"
        if [ -n "$inchg" ]; then
            echo "init_chg        file"
            echo "read_file_dir   ${inchg}"
        fi
        echo "cal_force       1"
        echo "pseudo_dir  /root/pporb/apns-pseudopotentials-v1"
        echo "orbital_dir /root/pporb/apns-orbitals-efficiency-v1"
    } > "$out"
}

# ---------- main ----------
read NAT MODE < <(parse_stru "${TESTDIR}/STRU")
echo "  Atoms: ${NAT}  ($MODE)"
echo ""

# ---- Step 0: base run (λ re-converged) ----
BASE_DIR="${TESTDIR}/base"
mkdir -p "${BASE_DIR}"
cp "${TESTDIR}/STRU" "${BASE_DIR}/STRU"
cp "${TESTDIR}/KPT" "${BASE_DIR}/KPT"
cp "${TESTDIR}/target.dat" "${BASE_DIR}/target.dat"
write_input "${BASE_DIR}/INPUT" base 0.01 "" ""
echo "[Step 0] Base SCF (λ re-converged)..."
E0=$(run_scf "${BASE_DIR}" base "${NPROC}")
if [ -z "$E0" ]; then echo "  BASE RUN FAILED"; exit 1; fi
echo "  E'(R0) = ${E0} eV"
grep -h "DeltaP P3" "${BASE_DIR}/run.log" | tail -1
# per-atom λ* (full precision) for group-1 freezing
if ! extract_lambda "${BASE_DIR}/run.log" > "${TESTDIR}/lambda_star.dat" 2>/dev/null; then
    echo "  !! no [DeltaP P3] line with λ — cannot freeze λ; check deltap_corr/SCF"
    exit 1
fi
echo "  λ* written to ${TESTDIR}/lambda_star.dat"

# analytic forces at R0
BASE_LOG=$(ls "${BASE_DIR}"/OUT.base/running_*.log | head -1)
F0=($(extract_force "${BASE_LOG}"))

# ---- FD loop ----
if [ "$GROUP" = "both" ] || [ "$GROUP" = "1" ]; then
    echo ""
    echo "===== Group 1: frozen λ (deltap_lambda_init_file = λ*) ====="
    for iat in $(seq 0 $((NAT - 1))); do
        for axis in 0 1 2; do
            for sign in plus minus; do
                STEP_DIR="${TESTDIR}/disp_g1_${iat}_${axis}_${sign}"
                mkdir -p "${STEP_DIR}"
                cp "${TESTDIR}/KPT" "${TESTDIR}/target.dat" "${STEP_DIR}/"
                cp "${TESTDIR}/lambda_star.dat" "${STEP_DIR}/"
                if [ "$sign" = plus ]; then SDELTA="$DELTA_A"; else SDELTA="-$DELTA_A"; fi
                displace_stru "${TESTDIR}/STRU" "${STEP_DIR}/STRU" "$iat" "$axis" "$SDELTA"
                write_input "${STEP_DIR}/INPUT" base 0.0 lambda_star.dat \
                    "${BASE_DIR}/OUT.base"
            done
        done
    done
    for iat in $(seq 0 $((NAT - 1))); do
        for axis in 0 1 2; do
            E_plus=$(run_scf "${TESTDIR}/disp_g1_${iat}_${axis}_plus" base "${NPROC}")
            E_minus=$(run_scf "${TESTDIR}/disp_g1_${iat}_${axis}_minus" base "${NPROC}")
            if [ -z "$E_plus" ] || [ -z "$E_minus" ]; then
                echo "  atom ${iat} axis ${axis}: RUN FAILED"
                FAILED=$((FAILED + 1)); continue
            fi
            F_FD=$(python3 -c "print(-(${E_plus} - ${E_minus}) / (2 * ${DELTA_A}))")
            F_ana=$(python3 -c "print(${F0[$((iat*3+axis))]})")
            RES=$(python3 -c "print(abs(${F_FD} - ${F_ana}))")
            PASS=$(python3 -c "print('PASS' if ${RES} < ${CRIT_EV_A} else 'FAIL')")
            if [ "$PASS" = FAIL ]; then FAILED=$((FAILED + 1)); fi
            echo "  atom ${iat} axis ${axis}: F_FD=${F_FD} F_ana=${F_ana} |Δ|=${RES} ${PASS}"
        done
    done
fi

if [ "$GROUP" = "both" ] || [ "$GROUP" = "2" ]; then
    echo ""
    echo "===== Group 2: re-converged λ at each displaced geometry ====="
    for iat in $(seq 0 $((NAT - 1))); do
        for axis in 0 1 2; do
            for sign in plus minus; do
                STEP_DIR="${TESTDIR}/disp_g2_${iat}_${axis}_${sign}"
                mkdir -p "${STEP_DIR}"
                cp "${TESTDIR}/KPT" "${TESTDIR}/target.dat" "${STEP_DIR}/"
                if [ "$sign" = plus ]; then SDELTA="$DELTA_A"; else SDELTA="-$DELTA_A"; fi
                displace_stru "${TESTDIR}/STRU" "${STEP_DIR}/STRU" "$iat" "$axis" "$SDELTA"
                write_input "${STEP_DIR}/INPUT" base 0.01 "" "${BASE_DIR}/OUT.base"
            done
        done
    done
    for iat in $(seq 0 $((NAT - 1))); do
        for axis in 0 1 2; do
            E_plus=$(run_scf "${TESTDIR}/disp_g2_${iat}_${axis}_plus" base "${NPROC}")
            E_minus=$(run_scf "${TESTDIR}/disp_g2_${iat}_${axis}_minus" base "${NPROC}")
            if [ -z "$E_plus" ] || [ -z "$E_minus" ]; then
                echo "  atom ${iat} axis ${axis}: RUN FAILED"
                FAILED=$((FAILED + 1)); continue
            fi
            F_FD=$(python3 -c "print(-(${E_plus} - ${E_minus}) / (2 * ${DELTA_A}))")
            F_ana=$(python3 -c "print(${F0[$((iat*3+axis))]})")
            RES=$(python3 -c "print(abs(${F_FD} - ${F_ana}))")
            PASS=$(python3 -c "print('PASS' if ${RES} < ${CRIT_EV_A} else 'FAIL')")
            if [ "$PASS" = FAIL ]; then FAILED=$((FAILED + 1)); fi
            echo "  atom ${iat} axis ${axis}: F_FD=${F_FD} F_ana=${F_ana} |Δ|=${RES} ${PASS}"
        done
    done
fi

echo ""
echo "===== Summary ====="
echo "  Criterion: |F_FD − F_ana| < ${CRIT_EV_A} eV/Å"
echo "  FAILED checks: ${FAILED}"
if [ "$FAILED" -gt 0 ]; then
    echo "  RESULT: FAIL — relax with deltap_corr is not force-consistent yet."
    echo "  Known gaps (T7-b): B (H_HK force, ~2.6 eV/Å @ λ*), A2 (∂τ/∂R),"
    echo "  τ-unit inconsistency (code uses lat0-unit position, spec says fractional)."
    exit 1
else
    echo "  RESULT: PASS"
    exit 0
fi
