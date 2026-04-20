#!/bin/bash
BINARY="/root/abacus-dftu-pw-port/build/abacus_2p"
BASEDIR="/root/abacus-dftu-pw-port/tests/integrate"
ZDY_BINARY="/root/abacus-zdy-tmp/build/abacus"
RESULTS="$BASEDIR/all_results.tsv"

echo -e "TEST\tBASIS\tDFTU\tDS\tNSPIN\tMAGDIR\tPORT_ETOT\tPORT_ITER\tPORT_STATUS\tZDY_ETOT\tZDY_ITER\tZDY_STATUS\tDIFF(eV)" > "$RESULTS"

run_test() {
    local dir=$1
    local binary=$2
    local prefix=$3  # PORT or ZDY

    pushd "$dir" > /dev/null 2>&1
    # Don't delete OUT.autotest if it exists - keep logs for debugging
    # rm -rf OUT.autotest 2>/dev/null || true
    mkdir -p OUT.autotest 2>/dev/null || true

    export OMP_NUM_THREADS=1
    timeout 60 mpirun --allow-run-as-root -n 1 "$binary" > /dev/null 2>&1 || true

    # Extract ETOT - support both formats:
    # v3.9: #TOTAL ENERGY# -6807.7271408 eV
    # v3.7: !FINAL_ETOT_IS -6807.727140778096 eV
    local etot=$(grep -rh "#TOTAL ENERGY" OUT.autotest/ 2>/dev/null | tail -1 | sed 's/.*#TOTAL ENERGY# //' | sed 's/ eV$//' | tr -d ' ')
    if [ -z "$etot" ]; then
        etot=$(grep -rh "!FINAL_ETOT_IS" OUT.autotest/ 2>/dev/null | tail -1 | sed 's/.*!FINAL_ETOT_IS //' | sed 's/ eV$//' | tr -d ' ')
    fi

    # Extract max iteration - handle multiple running_scf* files safely
    local max_iter=0
    for f in OUT.autotest/running_scf*; do
        [ -f "$f" ] || continue
        local n
        n=$(grep -c "ELEC ITER" "$f" 2>/dev/null) || n=0
        if [ "$n" -gt "$max_iter" ] 2>/dev/null; then max_iter=$n; fi
    done

    # Check status
    if [ -n "$etot" ] && [ "$etot" != "" ]; then
        echo "${prefix}_ETOT=$etot ${prefix}_ITER=$max_iter ${prefix}_STATUS=PASS"
    else
        echo "${prefix}_ETOT=NA ${prefix}_ITER=$max_iter ${prefix}_STATUS=FAIL"
    fi
    popd > /dev/null 2>&1
}

# Get list of tests
tests=($(ls -d "$BASEDIR"/[23]*_*/ 2>/dev/null | sort))
total=${#tests[@]}
echo "Running $total tests..."

for i in "${!tests[@]}"; do
    dir="${tests[$i]}"
    name=$(basename "$dir")

    # Parse test info from name
    basis=$(echo "$name" | grep -oP "LCAO|PW" || echo "NA")
    dftu=$(echo "$name" | grep -q "DFTU" && echo "Y" || echo "N")
    ds=$(echo "$name" | grep -q "_DS_" && echo "Y" || echo "N")
    nspin=$(echo "$name" | grep -oP "S[0-9]+" | sed 's/S//')
    magdir=$(echo "$name" | grep -oP "Z|XY|XYZ" || echo "NA")

    # Run port test
    port_out=$(run_test "$dir" "$BINARY" "PORT")
    eval "$port_out"

    # Run zdy-tmp test (for comparison)
    zdy_out=$(run_test "$dir" "$ZDY_BINARY" "ZDY")
    eval "$zdy_out"

    # Compute difference
    if [ "$PORT_ETOT" != "NA" ] && [ "$ZDY_ETOT" != "NA" ]; then
        diff=$(python3 -c "import sys; print(f'{abs(float(sys.argv[1]) - float(sys.argv[2])):.6e}')" "$PORT_ETOT" "$ZDY_ETOT" 2>/dev/null || echo "ERR")
    else
        diff="NA"
    fi

    echo -e "${name}\t${basis}\t${dftu}\t${ds}\t${nspin}\t${magdir}\t${PORT_ETOT}\t${PORT_ITER}\t${PORT_STATUS}\t${ZDY_ETOT}\t${ZDY_ITER}\t${ZDY_STATUS}\t${diff}" >> "$RESULTS"

    # Print progress
    printf "[%3d/%3d] %-35s PORT=%-15s (%3s iters)  ZDY=%-15s (%3s iters)  diff=%s\n" \
        $((i+1)) $total "$name" "$PORT_ETOT" "$PORT_ITER" "$ZDY_ETOT" "$ZDY_ITER" "$diff"
done

echo ""
echo "=== PASSED ==="
grep "PASS" "$RESULTS" | grep -v "FAIL" | wc -l
echo "=== FAILED ==="
grep "FAIL" "$RESULTS" | wc -l
echo "=== Results saved to $RESULTS ==="
cat "$RESULTS"
