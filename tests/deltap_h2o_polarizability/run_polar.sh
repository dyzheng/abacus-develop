#!/bin/bash
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
BIN=/root/abacus-develop/build/abacus_basic_para
PP=/root/abacus-develop/tests/PP_ORB
cd "$DIR"

# Fix PP paths to absolute
sed -i "s|../PP_ORB|$PP|g" INPUT

for lam in -0.005 -0.0025 0.0 0.0025 0.005; do
    label=$(echo $lam | sed 's/-/m/;s/\./_/')
    echo "=== RUNNING λ=$lam ==="
    rm -rf OUT.autotest
    sed -i "s/deltap_lambda_init.*/deltap_lambda_init $lam/g" INPUT
    $BIN > run_${label}.log 2>&1
    gamma=$(grep "DeltaP P3.*iter=" run_${label}.log | tail -1 | grep -oP 'Σγ=-?[0-9]+\.[0-9]+(?=\))' | sed 's/Σγ=//')
    echo "λ=$lam  Σγ=$gamma"
done
