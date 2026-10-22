#!/bin/bash
# Second half of run_all.sh, used after the first pull of the Windows PC's scenes failed (B and A were started on the Windows PC by hand):
# unpack the Windows PC's scenes, check labels, train and evaluate C here, then pull the Windows PC's results.
set -u
cd $HOME/audiodrone
PY=~/venvs/audiodrone/bin/python
PC2='C:\work\audiodrone'
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for the scene tar"
until grep -q PULLED "$1" 2>/dev/null; do sleep 30; done
tar -xf data/sim_v1_pc2.tar -C data && rm data/sim_v1_pc2.tar || exit 1
ssh pc2 "Remove-Item \"$PC2\\data\\sim_v1_pc2.tar\""
$PY scripts/check_labels.py

log "train C_small here"
$PY -m cnn.train --data data/sim_v1 --out runs/C_small --frontend logf --arch small --batch 16 --workers 6 --lr2 1e-3 \
  > runs/C_small.out 2>&1 || log "train C_small FAILED"
log "evaluate C_small"
$PY -m cnn.evaluate --data data/sim_v1 --split test_sim test_heldout --model runs/C_small/best.pt --frontend logf --arch small \
  > runs/C_small.eval.out 2>&1 || log "evaluate C_small FAILED"

log "waiting for the Windows PC's runs"
until ssh -o ConnectTimeout=10 pc2 "Test-Path \"$PC2\\runs\\PC2_DONE\"" 2>/dev/null | grep -q True; do sleep 300; done
for r in B_logf A_mel; do
  mkdir -p runs/$r
  scp -q -T "pc2:\"work/audiodrone/runs/$r/*\"" runs/$r/ && scp -q -T "pc2:\"work/audiodrone/runs/$r.*\"" runs/ || log "pull $r FAILED"
done
log "all done"
