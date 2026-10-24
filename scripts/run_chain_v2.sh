#!/usr/bin/env bash
# Chain v2 evaluation on the flat-cluster scenes (frequency tracker, tracker-guided beamformer, Doppler).
#   scripts/run_chain_v2.sh [data dir, default data/loc_ft] [config, default flat_ground_100]
# Steps: whole-band direction finding, oracle comb, tracker comb (+ .freq.npz), tracker scoring, tracking.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-data/loc_ft}
CFG=${2:-flat_ground_100}
PY=.venv/bin/python
$PY -m loc.run_doa --config "$CFG" --data "$DATA"
$PY -m loc.run_doa --config "$CFG" --data "$DATA" --comb
$PY -m loc.run_doa --config "$CFG" --data "$DATA" --trkcomb
$PY -m loc.eval_freq --scenes "$DATA/$CFG" --out "results/freqtrack/$CFG" --workers 4
$PY -m loc.evaluate --configs "$CFG" --data "$DATA" --out "results/chain_v2" --workers 4 \
    --track-methods bf bf_comb bf_trk bf_trk:real bf_trk:dop bf_trk:real:dop bf_comb:dop
echo CHAIN_V2_DONE
