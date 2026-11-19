#!/bin/bash

ABACUS_PATH=$(awk -F "=" '$1=="ABACUS_PATH"{print $2}' ../../SETENV)
ABACUS_NPROCS=$(awk -F "=" '$1=="ABACUS_NPROCS"{print $2}' ../../SETENV)
ABACUS_THREADS=$(awk -F "=" '$1=="ABACUS_THREADS"{print $2}' ../../SETENV)

OMP_NUM_THREADS=${ABACUS_THREADS} mpirun -np ${ABACUS_NPROCS} ${ABACUS_PATH} | tee scf.output
status=${PIPESTATUS[0]}

rm -f KPT

# Suffix-agnostic completion check: the newest OUT.*/running_scf.log must end
# with the "Total  Time" trailer.
log=$(ls -td OUT.*/running_scf.log 2>/dev/null | head -1)
if [ ${status} -ne 0 ] || [ -z "${log}" ] ||
   ! tail -1 "${log}" | grep -q "Total  Time"
then
	echo "job failed!"
	exit 1
else
	echo "job succeeded!"
	exit 0
fi
