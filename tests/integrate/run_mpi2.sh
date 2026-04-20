#!/bin/bash
BINARY="/root/abacus-dftu-pw-port/build/abacus_2p"
BASEDIR="/root/abacus-dftu-pw-port/tests/integrate"
RESULTS="$BASEDIR/mpi2_results.tsv"

echo -e "TEST\tBASIS\tDFTU\tDS\tNSPIN\tMAGDIR\tPORT_ETOT\tPORT_ITER\tPORT_STATUS\tTIME(s)" > "$RESULTS"

run_test_mpi2() {
    local dir=$1
    local prefix=$2  # PORT

    pushd "$dir" > /dev/null 2>&1
    rm -rf OUT.autotest 2>/dev/null || true
    mkdir -p OUT.autotest 2>/dev/null || true

    export OMP_NUM_THREADS=1
    local start_time=$(date +%s)
    timeout 120 mpirun --allow-run-as-root -n 2 "$BINARY" > /dev/null 2>&1 || true
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Extract ETOT - support both formats
    local etot=$(grep -rh "#TOTAL ENERGY" OUT.autotest/ 2>/dev/null | tail -1 | sed 's/.*#TOTAL ENERGY# //' | sed 's/ eV$//' | tr -d ' ')
    if [ -z "$etot" ]; then
        etot=$(grep -rh "!FINAL_ETOT_IS" OUT.autotest/ 2>/dev/null | tail -1 | sed 's/.*!FINAL_ETOT_IS //' | sed 's/ eV$//' | tr -d ' ')
    fi

    # Extract max iteration
    local max_iter=0
    for f in OUT.autotest/running_scf*; do
        [ -f "$f" ] || continue
        local n
        n=$(grep -c "ELEC ITER" "$f" 2>/dev/null) || n=0
        if [ "$n" -gt "$max_iter" ] 2>/dev/null; then max_iter=$n; fi
    done

    # Check convergence
    if grep -rh "SCF IS CONVERGED" OUT.autotest/ > /dev/null 2>&1; then
        local converged="YES"
    else
        local converged="NO"
    fi

    if [ -n "$etot" ] && [ "$etot" != "" ]; then
        echo "${prefix}_ETOT=$etot ${prefix}_ITER=$max_iter ${prefix}_STATUS=PASS ${prefix}_CONVERGED=$converged ${prefix}_TIME=$elapsed"
    else
        echo "${prefix}_ETOT=NA ${prefix}_ITER=$max_iter ${prefix}_STATUS=FAIL ${prefix}_CONVERGED=$converged ${prefix}_TIME=$elapsed"
    fi
    popd > /dev/null 2>&1
}

# Get list of PW tests with kpar=2
tests=($(ls -d "$BASEDIR"/[123]*_PW*/ 2>/dev/null | sort))
total=${#tests[@]}
echo "Running $total PW tests with MPI n=2, kpar=2..."
echo ""

pass=0
fail=0

for i in "${!tests[@]}"; do
    dir="${tests[$i]}"
    name=$(basename "$dir")

    basis="PW"
    dftu=$(echo "$name" | grep -q "DFTU" && echo "Y" || echo "N")
    ds=$(echo "$name" | grep -q "_DS_" && echo "Y" || echo "N")
    nspin=$(echo "$name" | grep -oP "S[0-9]+" | sed 's/S//')
    magdir=$(echo "$name" | grep -oP "Z|XY|XYZ" || echo "NA")

    port_out=$(run_test_mpi2 "$dir" "PORT")
    eval "$port_out"

    if [ "$PORT_STATUS" = "PASS" ]; then
        ((pass++))
    else
        ((fail++))
    fi

    conv_mark=""
    if [ "$PORT_CONVERGED" = "NO" ]; then
        conv_mark=" ⚠️NOT_CONV"
    fi

    printf "[%3d/%3d] %-35s E=%-18s iter=%-3s time=%3ds%s\n" \
        $((i+1)) $total "$name" "$PORT_ETOT" "$PORT_ITER" "$PORT_TIME" "$conv_mark"

    echo -e "${name}\t${basis}\t${dftu}\t${ds}\t${nspin}\t${magdir}\t${PORT_ETOT}\t${PORT_ITER}\t${PORT_STATUS}\t${PORT_TIME}" >> "$RESULTS"
done

echo ""
echo "========================================="
echo "PASSED: $pass / $total"
echo "FAILED: $fail / $total"
echo "========================================="
echo "Results saved to $RESULTS"
echo ""
cat "$RESULTS"
