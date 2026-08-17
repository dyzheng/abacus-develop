#!/bin/bash
# Stage 4.2 stationary-point FD (T3 protocol extended to hf/co/h2o_asym).
# Per system:
#   Step 0: natural run (lambda frozen at 0) -> natural Gamma -> t_Gamma*
#   Step 1: base + per-atom x/y/z +-delta legs, inner-loop lambda re-converged
#           against the frozen t_Gamma* (|Gamma-t*|<1e-3, deltap_secant off)
#   Step 2: FD force vs analytic force (base, lambda=0), residual report
#
# Serial only (OMP_NUM_THREADS=1), one task at a time. MPI not used.
# Usage: bash tools/run_stationary4.sh <system> [delta_bohr]
set -uo pipefail

SYSTEM="${1:-hf}"
DELTA_BOHR="${2:-0.005}"
ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
BASEDIR="$(cd "$(dirname "$0")/.." && pwd)"
TESTDIR="${BASEDIR}/${SYSTEM}"
DELTA_A=$(python3 -c "print(${DELTA_BOHR} * 0.5291772109)")
CRIT=$(python3 -c "print(5e-4 * 13.605693 / 0.5291772109)")
export OMP_NUM_THREADS=1

echo "===== Stage 4.2 stationary FD: ${SYSTEM}  delta=${DELTA_BOHR} Bohr ====="
echo "  criterion: |F_FD - F_ana| < ${CRIT} eV/Ag"

# ---------- STRU parsing (nat + species) ----------
parse_stru() {
python3 - "$1" <<'PYEOF'
import sys
lines = open(sys.argv[1]).readlines()
coords = []
in_pos = False; i = 0; nat = 0
while i < len(lines):
    l = lines[i].strip()
    if l.startswith('ATOMIC_POSITIONS'):
        in_pos = True; mode = lines[i+1].strip(); i += 2; continue
    if not in_pos: i += 1; continue
    if l == '' or l[0] == '#': i += 1; continue
    if len(l.split()) == 1 and l[0].isalpha():
        sp = l; i += 1; mag = lines[i].strip(); i += 1
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
python3 - "$1" "$2" "$3" "$4" "$5" <<'PYEOF'
import sys
src, dst, iat, axis, delta = sys.argv[1:6]
iat, axis, delta = int(iat), int(axis), float(delta)
lines = open(src).read().splitlines(keepends=True)
out = []; in_pos = False; mode = None; count = -1; i = 0
while i < len(lines):
    l = lines[i]
    if l.startswith('ATOMIC_POSITIONS'):
        in_pos = True; mode = lines[i+1].strip(); out.append(l); i += 1
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
                if mode != 'Direct': c[axis] += delta
                else: raise SystemExit('Direct mode unsupported here')
            extra = ' ' + ' '.join(lines[i].split()[3:]) if len(lines[i].split()) > 3 else ''
            out.append(' '.join(f'{x:.10f}' for x in c) + extra + '\n'); i += 1
        continue
    out.append(l); i += 1
open(dst, 'w').writelines(out)
PYEOF
}

write_input() { # out suffix lambda_step lambda_init_file extra
    local out="$1" suf="$2" step="$3" initfile="$4" proxy="$5"
    {
        echo "INPUT_PARAMETERS"
        echo "suffix      ${suf}"
        echo "calculation scf"
        echo "basis_type  lcao"
        echo "ecutwfc     100"
        echo "ecutrho     400"
        echo "gamma_only  0"
        echo "nspin       1"
        echo "scf_thr     1e-8"
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
        if [ -n "$initfile" ]; then echo "deltap_lambda_init_file ${initfile}"; fi
        echo "deltap_inner_thr    1.0e-3"
        echo "deltap_lambda_mixing 0.1"
        if [ -n "$proxy" ]; then
            echo "deltap_proxy_target_file ${proxy}"
            echo "deltap_secant        off"
            echo "deltap_inner_nmax    20"
            echo "deltap_conv_thr      1.0e-3"
        fi
        echo "cal_force       1"
        echo "pseudo_dir  /root/pporb/apns-pseudopotentials-v1"
        echo "orbital_dir /root/pporb/apns-orbitals-efficiency-v1"
    } > "$out"
}

run_scf() { # dir suffix -> E
    local dir="$1" suf="$2"
    ( cd "$dir" && rm -rf "OUT.${suf}" run.log && "$ABACUS" > run.log 2>&1 )
    grep -h '!FINAL_ETOT_IS' "$dir"/OUT.${suf}/running_*.log 2>/dev/null | tail -1 | awk '{print $(NF-1)}'
}

extract_gamma() { # log -> "g1 g2 ..."
    python3 - "$1" <<'PYEOF'
import sys, re
line=None
for l in open(sys.argv[1]):
    if '[DeltaP P3]' in l and 'Γ=' in l: line=l
if line is None: sys.exit(1)
m=re.search(r'Γ=\(([^)]*)\)',line)
print(' '.join(f'{float(x):.6f}' for x in m.group(1).split(',')))
PYEOF
}

extract_force() {
    python3 - "$1" <<'PYEOF'
import sys, re
txt=open(sys.argv[1]).read()
m=re.search(r'#TOTAL-FORCE[^\n]*#\n(.*?)\n\s*\n', txt, re.S)
if not m: sys.exit(1)
rows=[]
for l in m.group(1).splitlines():
    t=l.split()
    if len(t)==4 and t[0][0].isalpha() and t[0]!='Atoms':
        rows += [f'{float(t[1]):.9e}', f'{float(t[2]):.9e}', f'{float(t[3]):.9e}']
if not rows: sys.exit(1)
print('\n'.join(rows))
PYEOF
}

read NAT MODE < <(parse_stru "${TESTDIR}/STRU")
echo "  atoms=${NAT} (${MODE})"

NATDIR="${TESTDIR}/s4"
mkdir -p "${NATDIR}/base"
cp "${TESTDIR}/STRU" "${TESTDIR}/KPT" "${NATDIR}/base/"
# no target file: unconstrained natural run with lambda frozen at 0
write_input "${NATDIR}/base/INPUT" base 0.0 "" ""
E0=$(run_scf "${NATDIR}/base" base)
if [ -z "$E0" ]; then echo "  [Step 0] natural run FAILED"; tail -5 "${NATDIR}/base/run.log"; exit 1; fi
echo "  [Step 0] E_nat=${E0} eV"
if ! extract_gamma "${NATDIR}/base/run.log" > "${NATDIR}/gamma_nat.dat"; then
    echo "  !! no natural Gamma line"; exit 1
fi
echo "  natural Gamma: $(cat ${NATDIR}/gamma_nat.dat)"
# freeze t_Gamma* = natural Gamma
cp "${NATDIR}/gamma_nat.dat" "${NATDIR}/t_gamma_star.dat"

# analytic force at R0 (natural)
F0LOG=$(ls "${NATDIR}"/base/OUT.base/running_*.log | head -1)
mapfile -t F0 < <(extract_force "$F0LOG")
echo "  analytic forces (${#F0[@]} atoms):"
printf '    %s\n' "${F0[@]}"

# ---- stationary legs ----
declare -a ROWS
for iat in $(seq 0 $((NAT-1))); do
  for axis in 0 1 2; do
    for sign in plus minus; do
      STEP="${NATDIR}/disp_s_${iat}_${axis}_${sign}"
      mkdir -p "$STEP"
      cp "${TESTDIR}/KPT" "$STEP/"
      cp "${NATDIR}/t_gamma_star.dat" "$STEP/"
      if [ "$sign" = plus ]; then SD="$DELTA_A"; else SD="-$DELTA_A"; fi
      displace_stru "${TESTDIR}/STRU" "$STEP/STRU" "$iat" "$axis" "$SD"
      write_input "$STEP/INPUT" s 0.01 "" t_gamma_star.dat
    done
  done
done
echo "  [Step 1] stationary legs (${NAT}x3x2)..."
for iat in $(seq 0 $((NAT-1))); do
  for axis in 0 1 2; do
    Ep=$(run_scf "${NATDIR}/disp_s_${iat}_${axis}_plus" s)
    Em=$(run_scf "${NATDIR}/disp_s_${iat}_${axis}_minus" s)
    if [ -z "$Ep" ] || [ -z "$Em" ]; then
      echo "  atom ${iat} axis ${axis}: RUN FAILED (Ep=${Ep:-NA} Em=${Em:-NA})"; continue
    fi
    FFD=$(python3 -c "print(-(${Ep} - ${Em}) / (2 * ${DELTA_A}))")
    FAN=$(python3 -c "print(${F0[$((iat*3+axis))]})")
    RES=$(python3 -c "print(abs(${FFD} - ${FAN}))")
    PAS=$(python3 -c "print('PASS' if ${RES} < ${CRIT} else 'FAIL')")
    ROWS+=("${iat} ${axis} ${FFD} ${FAN} ${RES} ${PAS}")
    echo "  atom ${iat} axis ${axis}: F_FD=${FFD} F_ana=${FAN} |d|=${RES} ${PAS}"
  done
done

echo ""
echo "===== ${SYSTEM} summary ====="
printf '%-4s %-4s %-16s %-16s %-16s %s\n' iat axis F_FD F_ana RES PASS
printf '%s\n' "${ROWS[@]}"
echo "criterion = ${CRIT} eV/Ag (5e-4 Ry/Bohr)"
