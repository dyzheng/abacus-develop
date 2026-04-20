#!/bin/bash
BINARY="/root/abacus-dftu-pw-port/build/abacus_2p"
BASEDIR="/root/abacus-dftu-pw-port/tests/integrate"

tests=($(ls -d "$BASEDIR"/3[0-9][0-9]_PW*/ 2>/dev/null | sort))
total=${#tests[@]}
echo "Running $total new tests with MPI n=2, timeout 90s..."
echo ""

pass=0; fail=0

for i in "${!tests[@]}"; do
    dir="${tests[$i]}"
    name=$(basename "$dir")
    
    rm -rf "$dir/OUT.autotest" 2>/dev/null
    mkdir -p "$dir/OUT.autotest"
    
    export OMP_NUM_THREADS=1
    start_time=$(date +%s)
    timeout 90 mpirun --allow-run-as-root -n 2 "$BINARY" > /dev/null 2>&1 || true
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    
    # Check results
    etot=$(grep -rh "#TOTAL ENERGY" "$dir/OUT.autotest/" 2>/dev/null | tail -1 | sed 's/.*#TOTAL ENERGY# //' | sed 's/ eV$//' | tr -d ' ')
    
    converged="NO"
    grep -rh "SCF IS CONVERGED" "$dir/OUT.autotest/" > /dev/null 2>&1 && converged="YES"
    
    max_iter=0
    for f in "$dir/OUT.autotest/running_"*; do
        [ -f "$f" ] || continue
        n=$(grep -c "ELEC ITER" "$f" 2>/dev/null) || n=0
        [ "$n" -gt "$max_iter" ] 2>/dev/null && max_iter=$n
    done
    
    if [ -n "$etot" ] && [ "$etot" != "" ] && [ "$converged" = "YES" ]; then
        status="PASS"; ((pass++))
    else
        status="FAIL"; ((fail++))
    fi
    
    printf "[%3d/%3d] %-45s E=%-22s iter=%-3d time=%3ds %s\n" \
        $((i+1)) $total "$name" "${etot:-NA}" "$max_iter" "$elapsed" "$status"
done

echo ""
echo "========================================="
echo "PASSED: $pass / $total"
echo "FAILED: $fail / $total"
echo "========================================="
