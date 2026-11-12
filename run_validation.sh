#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
for run in base rl1 rl2 rl3; do
  echo "Validation $run"
  python3 validate.py "$run" > "logs/${run}_validation.log" 2>&1
done
echo VALIDATION_COMPLETE
