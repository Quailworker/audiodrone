#!/bin/bash
# Second half of run_all.sh, used after the first pull of Anna's scenes failed (B and A were started on Anna by hand):
# unpack Anna's scenes, check labels, train and evaluate C here, then pull Anna's results.
set -u
cd /home/quail/audiodrone
PY=~/venvs/audiodrone/bin/python
ANNA='C:\Users\pc\Joseph AI\audiodrone'
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for the scene tar"
until grep -q PULLED "$1" 2>/dev/null; do sleep 30; done
tar -xf data/sim_v1_anna.tar -C data && rm data/sim_v1_anna.tar || exit 1
ssh anna "Remove-Item \"$ANNA\\data\\sim_v1_anna.tar\""
$PY scripts/check_labels.py

log "train C_small here"
$PY -m cnn.train --data data/sim_v1 --out runs/C_small --frontend logf --arch small --batch 16 --workers 6 --lr2 1e-3 \
  > runs/C_small.out 2>&1 || log "train C_small FAILED"
log "evaluate C_small"
$PY -m cnn.evaluate --data data/sim_v1 --split test_sim test_heldout --model runs/C_small/best.pt --frontend logf --arch small \
  > runs/C_small.eval.out 2>&1 || log "evaluate C_small FAILED"

log "waiting for Anna's runs"
until ssh -o ConnectTimeout=10 anna "Test-Path \"$ANNA\\runs\\ANNA_DONE\"" 2>/dev/null | grep -q True; do sleep 300; done
for r in B_logf A_mel; do
  mkdir -p runs/$r
  scp -q -T "anna:\"Joseph AI/audiodrone/runs/$r/*\"" runs/$r/ && scp -q -T "anna:\"Joseph AI/audiodrone/runs/$r.*\"" runs/ || log "pull $r FAILED"
done
log "all done"
