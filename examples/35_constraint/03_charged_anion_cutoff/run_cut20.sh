#!/bin/bash

# Run the same anion example with constraint_mbis_cutoff = 20 Bohr.
# The tracked INPUT is restored after the run.

cp INPUT INPUT.cut10.keep
cp INPUT.cut20 INPUT
trap 'mv -f INPUT.cut10.keep INPUT' EXIT

bash run.sh
