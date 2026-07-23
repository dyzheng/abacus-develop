#!/bin/bash
set -e

SAMPLING_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="/root/abacus-develop/build/abacus_basic_para"
MAX_JOBS=3

labels=(center x_plus x_minus y_plus y_minus diag_plus diag_minus anti_plus anti_minus)

run_one() {
    local label=$1
    local dir="${SAMPLING_DIR}/${label}"
    echo "[$(date '+%H:%M:%S')] Starting ${label}..."
    cd "$dir"
    rm -rf OUT.bn
    OMP_NUM_THREADS=1 mpirun -np 1 "$BINARY" > "${label}.log" 2>&1
    local rc=$?
    echo "[$(date '+%H:%M:%S')] Finished ${label} (rc=${rc})"
    cd "$SAMPLING_DIR"
}

count=0
pids=()
for label in "${labels[@]}"; do
    run_one "$label" &
    pids+=($!)
    count=$((count + 1))
    if [ $count -ge $MAX_JOBS ]; then
        wait -n
        count=$((count - 1))
    fi
done

for pid in "${pids[@]}"; do
    wait $pid
done

echo "[$(date '+%H:%M:%S')] All 9 jobs completed."
