#!/bin/bash
BASEDIR="/root/abacus-dftu-pw-port/tests/integrate"
BINARY="/root/abacus-dftu-pw-port/build/abacus_2p"
RESULTS="/root/abacus-dftu-pw-port/tests/integrate/test_results.tsv"

echo -e "TEST\tBASIS\tFEATURES\tETOT(eV)\tITER\tEXIT_CODE\tSTATUS" > "$RESULTS"

for dir in "$BASEDIR"/[23]*_*/; do
  name=$(basename "$dir")
  
  # Clean old output
  rm -rf "${dir}OUT.autotest" 2>/dev/null
  
  # Run test
  cd "$dir"
  timeout 120 mpirun --allow-run-as-root -n 2 "$BINARY" > "${dir}run.log" 2>&1
  exit_code=$?
  
  # Extract results
  etot=$(grep -rh "#TOTAL ENERGY" "${dir}OUT.autotest/" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
  niter=$(grep -rh "#ELEC ITER" "${dir}OUT.autotest/" 2>/dev/null | tail -1 | awk '{print $NF}')
  
  # Determine status
  if [ "$exit_code" -eq 0 ] && [ -n "$etot" ] && [ "$etot" != "" ]; then
    status="PASS"
  elif [ "$exit_code" -eq 124 ]; then
    status="TIMEOUT"
  elif [ -n "$etot" ]; then
    status="PASS(exit_code=$exit_code)"
  else
    status="FAIL"
  fi
  
  # Parse test name
  basis=$(echo "$name" | grep -oP "LCAO|PW")
  features=$(echo "$name" | sed 's/^[0-9]*_//')
  
  echo -e "${name}\t${basis}\t${features}\t${etot}\t${niter}\t${exit_code}\t${status}" >> "$RESULTS"
  echo "[$name] ETOT=$etot ITERS=$niter STATUS=$status (exit=$exit_code)"
done

echo ""
echo "=== Summary ==="
cat "$RESULTS"
