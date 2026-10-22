#!/bin/bash
# Fast end-to-end check (a few minutes, CPU is fine): render 4 test scenes, check them against the manifest,
# evaluate the small model on them and run field-recording inference on one clip.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=data/smoke
rm -rf $OUT
$PY -m sim.render --out $OUT --start 0 --count 80 --workers 4 --prefix p --only-split test_sim
$PY scripts/manifest.py verify $OUT results/sim_v1_manifest.csv
$PY -m cnn.evaluate --data $OUT --split test_sim --model models/C_small_best.pt --frontend logf --arch small > $OUT/eval.out
$PY -c "import json; d=json.load(open('models/eval_test_sim.json')); print('smoke F1', {k: round(v['f1'], 3) for k, v in d['frame'].items()})"
rm models/eval_test_sim.json
$PY -m cnn.infer $OUT/test_sim/p0000001.flac --model models/C_small_best.pt --frontend logf --arch small
ls $OUT/test_sim/p0000001.detections.*
echo "smoke test OK"
