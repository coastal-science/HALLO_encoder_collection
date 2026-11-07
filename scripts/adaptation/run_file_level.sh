#!/bin/bash
# File-level scoring of the deployed detectors on the leak-free SoG holdout (score_file_level.py), sharded over CPU workers,
# then merge + a 20-file check against the unmodified backend detect(). Resumable (finished files are skipped).
#   tmux new -d -s file_level "bash scripts/adaptation/run_file_level.sh"
cd /home/noah/HALLO_encoder_collection
PY=${PY:-/home/noah/hallo-tuned-detectors/.venv/bin/python}
OUT=scripts/adaptation/results/file_level
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
declare -A SHARDS=([birdnet]=6 [sequence]=4)
mkdir -p $OUT/logs
for d in birdnet sequence; do
  for ((i=0; i<${SHARDS[$d]}; i++)); do
    nice -n 10 $PY scripts/adaptation/score_file_level.py --detector $d --shard $i --n-shards ${SHARDS[$d]} > $OUT/logs/${d}_shard$i.txt 2>&1 &
  done
done
wait
echo "SCORING DONE $(date)"; grep -h "done:\|FAILED" $OUT/logs/*_shard*.txt
for d in birdnet sequence; do
  $PY scripts/adaptation/score_file_level.py --detector $d --merge 2>&1 | grep -v "Warning\|super(Adam\|XNNPACK"
  $PY scripts/adaptation/score_file_level.py --detector $d --verify 20 2>&1 | grep -v "Warning\|super(Adam\|XNNPACK" | tail -1
done
echo "ALL DONE $(date)"
