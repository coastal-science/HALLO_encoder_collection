#!/bin/bash
# Scores the three hallo-tuned-detectors models on one site's comparison clips, in parallel (score_tuned_detectors.py),
# each logged as an MLflow run in experiment hallo_tuned_detectors_comparison; then runs compare_tuned_detectors.py.
#   hallo_v1 on the GPU (embed_venv); birdnet + sequence on CPU (hallo-tuned-detectors/.venv: py3.10, TF 2.8, ketos).
# Niced and thread-capped so it can run next to a training run. Run inside tmux:
#   tmux new -d -s tuned_det "bash scripts/adaptation/run_tuned_detectors.sh"
# Elsewhere (audio not under /data/DORI): PATH_MAP=/data/DORI=/new/root DETECTORS_PY=... bash run_tuned_detectors.sh
cd /home/noah/HALLO_encoder_collection
SITE=${SITE:-strait_of_georgia}
OUT=scripts/adaptation/results/tuned_detectors
EMBED_PY=${EMBED_PY:-/home/noah/embed_venv/bin/python}
DETECTORS_PY=${DETECTORS_PY:-/home/noah/hallo-tuned-detectors/.venv/bin/python}
EXTRA=(); [ -n "$PATH_MAP" ] && EXTRA=(--path-map "$PATH_MAP")
mkdir -p $OUT
export MLFLOW_DISABLE_AGENT_HINT=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
declare -A PY=([hallo_v1]=$EMBED_PY [birdnet]=$DETECTORS_PY [sequence]=$DETECTORS_PY)
status=0
for d in hallo_v1 birdnet sequence; do
  nice -n 10 ${PY[$d]} scripts/adaptation/score_tuned_detectors.py --detector $d --site $SITE "${EXTRA[@]}" \
    > $OUT/${d}_${SITE}_log.txt 2>&1 &
done
for job in $(jobs -p); do wait $job || status=1; done
echo "SCORING DONE status=$status $(date)"; tail -n 3 $OUT/*_${SITE}_log.txt
[ $status -eq 0 ] || { echo "a detector failed -- not running the comparison"; exit 1; }
$EMBED_PY scripts/adaptation/compare_tuned_detectors.py --site $SITE 2>&1 | grep -v "Downloading artifacts" | tee $OUT/compare_${SITE}_log.txt
echo "ALL DONE $(date)"
