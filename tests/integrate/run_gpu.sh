#!/bin/bash
BINARY="/root/abacus-dftu-pw-port/build_gpu/abacus_2g"
BASEDIR="/root/abacus-dftu-pw-port/tests/integrate"
PP_ORB="/root/abacus-dftu-pw-port/tests/PP_ORB"
RESULTS="$BASEDIR/gpu_results.tsv"
CPU_RESULTS="$BASEDIR/all_results.tsv"

echo -e "TEST\tBASIS\tDFTU\tDS\tNSPIN\tMAGDIR\tGPU_ETOT\tGPU_ITER\tGPU_STATUS\tDIFF(eV)" > "$RESULTS"

run_gpu_test() {
    local dir=$1
    pushd "$dir" > /dev/null 2>&1
    rm -rf OUT.gpu 2>/dev/null
    mkdir -p OUT.gpu 2>/dev/null

    # Copy INPUT/STRU/KPT and fix paths
    cp INPUT STRU KPT OUT.gpu/ 2>/dev/null || true
    cd OUT.gpu

    # Fix paths in INPUT
    sed -i "s|pseudo_dir.*|pseudo_dir    $PP_ORB|" INPUT 2>/dev/null
    sed -i "s|orbital_dir.*|orbital_dir    $PP_ORB|" INPUT 2>/dev/null
    # Force kpar=1 for single-process GPU
    if grep -q "^kpar" INPUT 2>/dev/null; then
        sed -i "s/^kpar.*/kpar    1/" INPUT
    fi
    # Force gamma_only=0 for spin-polarized (GPU may not support gamma_only=1 with spin)
    # Don't change gamma_only as it might be needed

    export OMP_NUM_THREADS=1
    timeout 60 "$BINARY" > /dev/null 2>&1 || true

    local etot=$(grep -rh "!FINAL_ETOT_IS" OUT.autotest/ 2>/dev/null | tail -1 | sed 's/.*!FINAL_ETOT_IS //' | sed 's/ eV$//' | tr -d ' ')
    if [ -z "$etot" ]; then
        etot=$(grep -rh "#TOTAL ENERGY" OUT.autotest/ 2>/dev/null | tail -1 | sed 's/.*#TOTAL ENERGY# //' | sed 's/ eV$//' | tr -d ' ')
    fi

    local max_iter=0
    for f in OUT.autotest/running_scf*; do
        [ -f "$f" ] || continue
        local n
        n=$(grep -c "ELEC ITER" "$f" 2>/dev/null) || n=0
        if [ "$n" -gt "$max_iter" ] 2>/dev/null; then max_iter=$n; fi
    done

    if [ -n "$etot" ] && [ "$etot" != "" ]; then
        echo "${etot} ${max_iter} PASS"
    else
        echo "NA ${max_iter} FAIL"
    fi
    popd > /dev/null 2>&1
}

# Only PW tests (220+)
tests=($(ls -d "$BASEDIR"/22[0-9]*/ "$BASEDIR"/25[0-9]*/ "$BASEDIR"/26[0-9]*/ "$BASEDIR"/32[0-9]*/ "$BASEDIR"/33[0-9]*/ "$BASEDIR"/34[0-9]*/ "$BASEDIR"/35[0-9]*/ 2>/dev/null | sort))
total=${#tests[@]}
echo "Running $total GPU tests (PW only)..."

for i in "${!tests[@]}"; do
    dir="${tests[$i]}"
    name=$(basename "$dir")

    basis=$(echo "$name" | grep -oP "LCAO|PW" || echo "NA")
    dftu=$(echo "$name" | grep -q "DFTU" && echo "Y" || echo "N")
    ds=$(echo "$name" | grep -q "_DS_\|_DS_S" && echo "Y" || echo "N")
    nspin=$(echo "$name" | grep -oP "S[0-9]+" | head -1 | sed 's/S//')
    magdir=$(echo "$name" | grep -oP "Z|XY|XYZ" | head -1 || echo "NA")

    gpu_out=$(run_gpu_test "$dir")
    read GPU_ETOT GPU_ITER GPU_STATUS <<< "$gpu_out"

    # Get CPU ETOT from all_results.tsv for comparison
    cpu_etot="NA"
    if [ -f "$CPU_RESULTS" ]; then
        cpu_etot=$(grep "^${name}	" "$CPU_RESULTS" 2>/dev/null | cut -f7)
        [ -z "$cpu_etot" ] && cpu_etot="NA"
    fi

    if [ "$GPU_ETOT" != "NA" ] && [ "$cpu_etot" != "NA" ]; then
        diff=$(python3 -c "import sys; print(f'{abs(float(sys.argv[1]) - float(sys.argv[2])):.6e}')" "$GPU_ETOT" "$cpu_etot" 2>/dev/null || echo "ERR")
    else
        diff="NA"
    fi

    echo -e "${name}\t${basis}\t${dftu}\t${ds}\t${nspin}\t${magdir}\t${GPU_ETOT}\t${GPU_ITER}\t${GPU_STATUS}\t${diff}" >> "$RESULTS"

    printf "[%3d/%3d] %-40s GPU=%-20s (%3s iters)  CPU=%-20s  diff=%s\n" \
        $((i+1)) $total "$name" "$GPU_ETOT" "$GPU_ITER" "$cpu_etot" "$diff"
done

echo ""
echo "=== PASSED ==="
grep "PASS" "$RESULTS" | wc -l
echo "=== FAILED ==="
grep "FAIL" "$RESULTS" | wc -l
echo "=== Results saved to $RESULTS ==="
