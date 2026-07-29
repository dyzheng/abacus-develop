#!/bin/bash
# DeltaP FD force validation test
# ==================================
# Protocol: freeze lambda, displace atom ±δ, re-converge SCF,
# compare FD force = -(E_{+δ} - E_{-δ})/(2δ) with cal_force output.
#
# Usage: bash run_fd.sh [system=h2o] [delta=0.005] [nproc=1]
# Prerequisites:
#   1. abacus_basic_para built and in PATH or set ABACUS=
#   2. Pseudopotentials and orbitals at paths in INPUT
#   3. MPI (optional, for nproc > 1)

set -e

SYSTEM="${1:-h2o}"
DELTA="${2:-0.005}"
NPROC="${3:-1}"
ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
TESTDIR="${BASEDIR}/${SYSTEM}"

echo "===== DeltaP FD Force Validation ====="
echo "  System:  ${SYSTEM}"
echo "  Delta:   ${DELTA} Bohr"
echo "  Nproc:   ${NPROC}"
echo "  ABACUS:  ${ABACUS}"
echo ""

# ---- Step 0: base SCF + constraint convergence ----
STEP0_DIR="${TESTDIR}/base"
mkdir -p "${STEP0_DIR}"
cp "${TESTDIR}/STRU" "${STEP0_DIR}/"
cp "${TESTDIR}/KPT" "${STEP0_DIR}/"

# Build INPUT with frozen constraint
cat > "${STEP0_DIR}/INPUT" << 'EOF'
INPUT_PARAMETERS
suffix      base
calculation scf
basis_type  lcao
ecutwfc     100
gamma_only  0
nspin       1
scf_thr     1.0e-7
scf_nmax    100
out_chg     0
smearing_method gauss
smearing_sigma  0.002
mixing_type     broyden
mixing_beta     0.4
ks_solver       genelpa
symmetry       0
berry_phase     1
gdir            3
deltap_switch   1
deltap_method   wannier
deltap_rm       3.0
deltap_gdir     3
deltap_corr     1
deltap_lambda_step  0.5
deltap_lambda_init  0.0
deltap_inner_thr    1.0e-3
deltap_lambda_mixing 1.0
deltap_target_file   ../target.dat
cal_force       1
pseudo_dir  /root/pporb/apns-pseudopotentials-v1
orbital_dir /root/pporb/apns-orbitals-efficiency-v1
EOF

echo "[Step 0] Base SCF + constraint..."
cd "${STEP0_DIR}"
if [ $NPROC -gt 1 ]; then
    mpirun -np $NPROC $ABACUS > base.log 2>&1
else
    $ABACUS > base.log 2>&1
fi
echo "  Base energy (etot + dp_escon): $(grep '!FINAL_ETOT_IS' OUT.base/running*.log | tail -1 | awk '{print $NF}')"

# ---- Step 1: FD force for each atom ----
NAT=$(grep -c "^[A-Z]" "${TESTDIR}/STRU" | head -1 || echo 4)  # HACK: count species lines

echo "[Step 1] FD force loop (${NAT} atoms)..."
for iat in $(seq 0 $((NAT - 1))); do
    for sign in plus minus; do
        STEP_DIR="${TESTDIR}/disp_${iat}_${sign}"
        mkdir -p "${STEP_DIR}"
        cp "${TESTDIR}/STRU" "${STEP_DIR}/STRU"
        cp "${TESTDIR}/KPT" "${STEP_DIR}/KPT"

        # Displace atom iat by ±delta along gdir (z, index 2)
        h_delta=$(echo "$DELTA / 2.0" | bc -l 2>/dev/null || echo "0.0025")
        # Use python or awk to displace
        if [ "$sign" = "plus" ]; then
            python3 -c "
import sys; lines = open('${TESTDIR}/STRU').readlines();
iat, delta, gdir = ${iat}, ${DELTA}, 2;
in_xyz = False; count = -1;
out = [];
for l in lines:
    if l.startswith('Cartesian'):
        in_xyz = True; out.append(l); continue
    if in_xyz and l.strip() and not l.startswith('#') and not l.startswith('LATTICE') and not l.startswith('A') and not l.strip().startswith('Di'):
        toks = l.split()
        if len(toks) >= 4:
            count += 1
            if count == iat:
                toks[gdir] = str(float(toks[gdir]) + delta)
            l = ' '.join(toks) + '\n'
    out.append(l)
open('${STEP_DIR}/STRU', 'w').writelines(out)
" 2>/dev/null
        else
            python3 -c "
import sys; lines = open('${TESTDIR}/STRU').readlines();
iat, delta, gdir = ${iat}, ${DELTA}, 2;
in_xyz = False; count = -1;
for l in lines:
    if l.startswith('Cartesian'):
        in_xyz = True
    if in_xyz and l.strip() and not l.startswith('#') and not l.startswith('LATTICE') and not l.startswith('A') and not l.strip().startswith('Di'):
        toks = l.split()
        if len(toks) >= 4:
            count += 1
            if count == iat:
                toks[gdir] = str(float(toks[gdir]) - delta)
            l = ' '.join(toks) + '\n'
open('${STEP_DIR}/STRU', 'w').writelines(out)
" 2>/dev/null
        fi

        # Build INPUT (same as base but read charge from base for consistency)
        cat > "${STEP_DIR}/INPUT" << EOF
INPUT_PARAMETERS
suffix      disp_${iat}_${sign}
calculation scf
basis_type  lcao
ecutwfc     100
gamma_only  0
nspin       1
scf_thr     1.0e-7
scf_nmax    100
out_chg     0
smearing_method gauss
smearing_sigma  0.002
mixing_type     broyden
mixing_beta     0.4
ks_solver       genelpa
symmetry       0
berry_phase     1
gdir            3
deltap_switch   1
deltap_method   wannier
deltap_rm       3.0
deltap_gdir     3
deltap_corr     1
deltap_lambda_step  0.5
deltap_lambda_init  0.0
deltap_inner_thr    1.0e-3
deltap_lambda_mixing 1.0
deltap_target_file   ../target.dat
init_chg        file
read_file_dir   ${STEP0_DIR}/OUT.base
cal_force       1
pseudo_dir  /root/pporb/apns-pseudopotentials-v1
orbital_dir /root/pporb/apns-orbitals-efficiency-v1
EOF

        cd "${STEP_DIR}"
        if [ $NPROC -gt 1 ]; then
            mpirun -np $NPROC $ABACUS > disp.log 2>&1
        else
            $ABACUS > disp.log 2>&1
        fi
    done

    # Extract energies
    E_plus=$(grep '!FINAL_ETOT_IS' "${TESTDIR}/disp_${iat}_plus/OUT.disp_${iat}_plus/running"*.log 2>/dev/null | tail -1 | awk '{print $NF}')
    E_minus=$(grep '!FINAL_ETOT_IS' "${TESTDIR}/disp_${iat}_minus/OUT.disp_${iat}_minus/running"*.log 2>/dev/null | tail -1 | awk '{print $NF}')
    if [ -z "$E_plus" ] || [ -z "$E_minus" ]; then
        echo "  atom ${iat}: FD force = FAILED (check logs)"
        continue
    fi
    # FD force (Ry/Bohr), negative by definition F = -dE/dR
    F_FD=$(echo "scale=6; -($E_plus - $E_minus) / (2.0 * $DELTA)" | bc -l 2>/dev/null || echo "NAN")
    echo "  atom ${iat}: FD force = ${F_FD} Ry/Bohr"
done

echo ""
echo "===== FD validation complete ====="
echo "  Compare FD forces with cal_force output in base/OUT.base/"
echo "  Look for 'TOTAL-FORCE' in the running log."
echo ""
echo "  Expected: FD force ≈ cal_force (within 5% for H2O)"
