#!/bin/bash
set -e

ABACUS="/root/abacus-dftu-pw-port/build/abacus_2p"
TESTDIR="/root/abacus-dftu-pw-port/tests/integrate"
RESULTS="$TESTDIR/batch1_results.tsv"
TIMEOUT=120

# Test list
TESTS=(
  "200_LCAO_SPIN_S2_Z"
  "201_LCAO_SPIN_S4_XYZ"
  "202_LCAO_DFTU_S2_Z"
  "203_LCAO_DFTU_S2_XY"
  "204_LCAO_DFTU_S4_XYZ"
  "220_PW_SPIN_S2_Z"
  "221_PW_SPIN_S4_XYZ"
  "222_PW_DFTU_S2_Z"
  "223_PW_DFTU_S2_XY"
  "224_PW_DFTU_S4_XYZ"
  "225_PW_DFTU_S2_FeO"
)

# Header
printf "TEST\tETOT\tITER\tEXIT_CODE\tSTATUS\n" > "$RESULTS"

for t in "${TESTS[@]}"; do
  echo "=== Running $t ==="
  td="$TESTDIR/$t"
  outdir="$td/OUT.autotest"

  # Clean previous output
  rm -rf "$outdir"
  mkdir -p "$outdir"

  # Run with timeout
  exit_code=0
  cd "$td"
  timeout $TIMEOUT mpirun --allow-run-as-root -n 2 "$ABACUS" > /dev/null 2>&1 || exit_code=$?
  cd "$TESTDIR"

  # Determine status
  if [ $exit_code -eq 0 ]; then
    status="PASS"
  elif [ $exit_code -eq 124 ]; then
    status="TIMEOUT"
  else
    status="FAIL"
  fi

  # Extract ETOT from running_scf file
  etot="N/A"
  if [ -d "$outdir" ]; then
    running_file=$(ls "$outdir"/running_scf* 2>/dev/null | head -1)
    if [ -n "$running_file" ]; then
      etot_line=$(grep "#TOTAL ENERGY" "$running_file" | tail -1)
      if [ -n "$etot_line" ]; then
        etot=$(echo "$etot_line" | awk '{print $NF}')
      fi
    fi
  fi

  # Extract iteration count from #ELEC ITER lines
  iter="N/A"
  if [ -d "$outdir" ]; then
    running_file=$(ls "$outdir"/running_scf* 2>/dev/null | head -1)
    if [ -n "$running_file" ]; then
      iter_count=$(grep -c "#ELEC ITER" "$running_file" 2>/dev/null || echo "0")
      iter="$iter_count"
    fi
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" "$t" "$etot" "$iter" "$exit_code" "$status" >> "$RESULTS"
  echo "  ETOT=$etot  ITER=$iter  EXIT=$exit_code  STATUS=$status"
done

echo ""
echo "=== Results Summary ==="
cat "$RESULTS"
