#!/bin/bash
# Wait for both PCs to finish rendering, pull Anna's scenes, then train and evaluate runs B, A, C.
set -u
cd /home/quail/audiodrone
PY=~/venvs/audiodrone/bin/python
log() { echo "[$(date '+%F %T')] $*"; }

anna_done() {
  ssh -o ConnectTimeout=10 anna 'Get-Content "C:\Users\pc\Joseph AI\audiodrone\data\render_ah.out" -Tail 1 -ErrorAction SilentlyContinue; Get-Content "C:\Users\pc\Joseph AI\audiodrone\data\render_p2.out" -Tail 1 -ErrorAction SilentlyContinue' 2>/dev/null \
    | grep -c -E "^(400/400|1866/1866) " | grep -q 2
}

log "waiting for renders"
until anna_done && ! systemctl --user is-active -q audiodrone-render; do sleep 120; done

log "pulling Anna's scenes"
ssh anna 'cd "C:\Users\pc\Joseph AI\audiodrone\data"; tar -cf sim_v1_anna.tar sim_v1' || exit 1
scp -q 'anna:"Joseph AI/audiodrone/data/sim_v1_anna.tar"' data/ || exit 1
tar -xf data/sim_v1_anna.tar -C data && rm data/sim_v1_anna.tar
$PY - <<'EOF' || exit 1
import glob, json, collections
c = collections.Counter(); stale = 0
for js in glob.glob("data/sim_v1/*/*.json"):
    m = json.load(open(js)); c[js.split("/")[-2]] += 1
    stale += m.get("labels") != "snr_lines2"
print("scenes per split", dict(c), "stale labels", stale)
assert stale == 0, "some scenes still have old labels"
EOF
grep -h -c "error" data/sim_v1/*.log | paste -sd+ | bc | xargs echo "render errors:"

train() {  # name frontend arch [extra train args]
  local name=$1 fe=$2 arch=$3; shift 3
  log "train $name"
  $PY -m cnn.train --data data/sim_v1 --out runs/$name --frontend $fe --arch $arch --batch 16 --workers 6 "$@" \
    > runs/$name.out 2>&1 || log "train $name FAILED"
  log "evaluate $name"
  $PY -m cnn.evaluate --data data/sim_v1 --split test_sim test_heldout --model runs/$name/best.pt \
    --frontend $fe --arch $arch > runs/$name.eval.out 2>&1 || log "evaluate $name FAILED"
}
train B_logf logf cnn14
train A_mel mel cnn14
train C_small logf small --lr2 1e-3
log "all done"
