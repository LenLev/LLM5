#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
for run in rl1 rl2 rl3; do
  echo "Training $run"
  python3 train.py "$run" > "logs/$run.log" 2>&1
  echo "Merging $run"
  python3 merge.py "$run" > "logs/${run}_merge.log" 2>&1
  echo "Evaluating $run"
  python3 evaluate.py "checkpoints/$run" --output "results/$run" > "logs/${run}_eval.log" 2>&1
done
echo ALL_RUNS_COMPLETE
