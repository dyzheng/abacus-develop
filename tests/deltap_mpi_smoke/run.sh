#!/bin/bash
# DeltaP MPI consistency smoke (D5).
#
# Runs the three validated KPAR=1 paths with the current build:
#   - PW  2-rank : tests/deltap_pw_h2o  (H2O, k-string along gdir=3)
#   - LCAO 4-rank: tests/deltap_bn_test (BN, 2x2x2 k-mesh, 2x2 square grid)
#   - LCAO 4-rank: tests/deltap_mpi_smoke/deltap_co_lcao (CO, odd NBANDS=15: regression guard
#     for the D_I Allreduce truncation / band-mixing bug; NBANDS=15 is not
#     divisible by the 2-column process grid)
#   - LCAO inner-loop 4-rank: tests/deltap_bn_sampling/test_stru_target
#     (deltap_inner_nmax=3; runs in a temp copy so its tracked
#     deltap_branch*.dat outputs are not overwritten in the repo)
#
# PASS criteria per case:
#   - run exits 0
#   - the DeltaP init / P2 / inner-loop-done marker line is printed
#   - no cross-rank gamma divergence WARNING (D5 regression guard)
#
# Usage: ./run.sh
# Env overrides: ABACUS (binary), MPIRUN, NPROC_PW (default 2), NPROC_LCAO (default 4).
# Prerequisites: MPI build of abacus_basic_para; pseudopotentials/orbitals at the
# paths in the case INPUTs (tests/PP_ORB for deltap_pw_h2o,
# /root/pporb/apns-* for the BN cases).
set -e

ABACUS="${ABACUS:-/root/abacus-develop/build/abacus_basic_para}"
MPIRUN="${MPIRUN:-mpirun}"
NPROC_PW="${NPROC_PW:-2}"
NPROC_LCAO="${NPROC_LCAO:-4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

fail=0

run_case()
{
    local dir="$1" nproc="$2" marker="$3" use_copy="$4"
    local workdir="$ROOT/$dir" tmp=""
    # Inner-loop cases write tracked deltap_branch*.dat files: run them in a
    # temp copy so the repo stays clean.
    if [ "$use_copy" = "1" ]; then
        tmp="$(mktemp -d)"
        cp -r "$ROOT/$dir"/. "$tmp/"
        rm -f "$tmp"/*.log "$tmp"/deltap_branch*.dat \
              "$tmp"/deltap_match.dat "$tmp"/deltap_zeta_debug.dat
        workdir="$tmp"
    fi
    cd "$workdir"
    rm -rf OUT.* run.log
    if ! timeout 900 "$MPIRUN" -np "$nproc" "$ABACUS" > run.log 2>&1; then
        echo "FAIL: $dir ($nproc ranks) exited non-zero"
        fail=1
        return 1
    fi
    if ! grep -Fq "$marker" run.log; then
        echo "FAIL: $dir ($nproc ranks) missing marker '$marker'"
        fail=1
        return 1
    fi
    if grep -Fq "diverges across MPI ranks" run.log; then
        echo "FAIL: $dir ($nproc ranks) cross-rank gamma divergence detected"
        fail=1
        return 1
    fi
    echo "PASS: $dir ($nproc ranks)"
    rm -rf OUT.* run.log
    rm -rf "$tmp"
}

# `|| true`: run_case records failures in $fail and returns 1; without it,
# set -e would abort the script and skip the remaining case and the summary.
run_case deltap_pw_h2o "$NPROC_PW" "[DeltaP-PW] Initialized" || true
run_case deltap_bn_test "$NPROC_LCAO" "[DeltaP P2]" || true
run_case deltap_mpi_smoke/deltap_co_lcao "$NPROC_LCAO" "[DeltaP P2]" 1 || true
run_case deltap_bn_sampling/test_stru_target "$NPROC_LCAO" "inner loop done" 1 || true

if [ "$fail" -ne 0 ]; then
    echo "DeltaP MPI smoke FAILED"
    exit 1
fi
echo "DeltaP MPI smoke PASSED"
