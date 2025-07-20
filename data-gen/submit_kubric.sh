#!/bin/bash
# submit_kubric.sh

CHUNK_SIZE=1000
TOTAL=3000
STEP=$CHUNK_SIZE
while [ $STEP -lt $TOTAL ]; do
  OFFSET=$(( STEP - CHUNK_SIZE ))
  END=$(( STEP - 1 ))
  echo "Submitting scenes ${OFFSET}–${END}"
  sbatch --array=${OFFSET}-${END}%38 compute_dataset.sbatch
  STEP=$(( STEP + CHUNK_SIZE ))
done
# last chunk: 2000–2999
OFFSET=$(( STEP - CHUNK_SIZE ))
END=$(( TOTAL - 1 ))
echo "Submitting scenes ${OFFSET}–${END}"
sbatch --array=${OFFSET}-${END}%38 compute_dataset.sbatch