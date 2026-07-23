#!/bin/bash
# BN zinc blende DeltaP + Wannier90 validation test
# Usage: bash run.sh

set -e
ABACUS=/root/abacus-develop/build/abacus_basic_para
WANNIER90=$(which wannier90.x)
NPROC=2

BASEDIR=$(cd "$(dirname "$0")" && pwd)

echo "===== Step 1: SCF (equilibrium) ====="
cd "$BASEDIR/scf"
rm -rf OUT.autotest
mpirun -np $NPROC $ABACUS > scf.log 2>&1
echo "SCF done. Check OUT.autotest/ for charge density."

echo ""
echo "===== Step 2: NSCF berry+DeltaP (ref) ====="
cd "$BASEDIR/nscf_berry_ref"
rm -rf OUT.autotest
mpirun -np $NPROC $ABACUS > nscf_berry.log 2>&1
echo "NSCF berry ref done."

echo ""
echo "===== Step 3: NSCF berry+DeltaP (N displaced) ====="
cd "$BASEDIR/nscf_berry_disp"
rm -rf OUT.autotest
mpirun -np $NPROC $ABACUS > nscf_berry.log 2>&1
echo "NSCF berry disp done."

echo ""
echo "===== Step 4: Wannier90 pre-processing ====="
cd "$BASEDIR/wannier90"
rm -f bn.nnkp bn.amn mmn.dat eig.dat
$WANNIER90 -pp bn > w90_pp.log 2>&1
echo "Wannier90 -pp done. bn.nnkp generated."

echo ""
echo "===== Step 5: NSCF with wannier90 interface ====="
cd "$BASEDIR/nscf_wann"
cp "$BASEDIR/wannier90/bn.nnkp" .
rm -rf OUT.autotest
mpirun -np $NPROC $ABACUS > nscf_wann.log 2>&1
echo "NSCF wann done. .mmn/.amn/.eig generated."

echo ""
echo "===== Step 6: Wannier90 main run ====="
cd "$BASEDIR/wannier90"
cp "$BASEDIR/nscf_wann/bn.mmn" . 2>/dev/null || true
cp "$BASEDIR/nscf_wann/bn.amn" . 2>/dev/null || true
cp "$BASEDIR/nscf_wann/bn.eig" . 2>/dev/null || true
$WANNIER90 bn > w90_run.log 2>&1
echo "Wannier90 done."

echo ""
echo "===== All done ====="
echo "Results:"
echo "  Berry phase (ref):  $BASEDIR/nscf_berry_ref/OUT.autotest/"
echo "  Berry phase (disp): $BASEDIR/nscf_berry_disp/OUT.autotest/"
echo "  Wannier90 output:   $BASEDIR/wannier90/bn.wout"
