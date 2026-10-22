#!/bin/bash
# Evaluate the three published models on test_sim and test_heldout, then compare with the reference results.
#   bash scripts/test_models.sh [cuda|cpu]
# Writes models/eval/<run>/eval_<split>.json. GPU: ~10 min in total; CPU: a few hours.
set -euo pipefail
cd "$(dirname "$0")/.."
DEV=${1:-cuda}
PY=.venv/bin/python
declare -A FRONT=([B_logf]=logf [A_mel]=mel [C_small]=logf) ARCH=([B_logf]=cnn14 [A_mel]=cnn14 [C_small]=small)
for r in B_logf A_mel C_small; do
  mkdir -p models/eval/$r
  cp models/${r}_best.pt models/eval/$r/best.pt
  echo "== evaluating $r on $DEV"
  $PY -m cnn.evaluate --data data/sim_v1 --split test_sim test_heldout --model models/eval/$r/best.pt \
    --frontend ${FRONT[$r]} --arch ${ARCH[$r]} --device "$DEV" > models/eval/$r/evaluate.out
done
$PY scripts/compare_results.py models/eval results
