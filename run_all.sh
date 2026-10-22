#!/bin/bash
# Wait for both PCs to finish rendering, pull the Windows PC's scenes, then train and evaluate runs B, A, C.
set -u
cd $HOME/audiodrone
PY=~/venvs/audiodrone/bin/python
log() { echo "[$(date '+%F %T')] $*"; }

pc2_done() {
  ssh -o ConnectTimeout=10 pc2 'Get-Content "C:\work\audiodrone\data\render_ah.out" -Tail 1 -ErrorAction SilentlyContinue; Get-Content "C:\work\audiodrone\data\render_p2.out" -Tail 1 -ErrorAction SilentlyContinue' 2>/dev/null \
    | grep -c -E "^(400/400|1866/1866) " | grep -q 2
}

log "waiting for renders"
until pc2_done && ! systemctl --user is-active -q audiodrone-render; do sleep 120; done

log "pulling the Windows PC's scenes"
ssh pc2 'cd "C:\work\audiodrone\data"; tar -cf sim_v1_pc2.tar sim_v1' || exit 1
scp -q 'pc2:"work/audiodrone/data/sim_v1_pc2.tar"' data/ || exit 1
tar -xf data/sim_v1_pc2.tar -C data && rm data/sim_v1_pc2.tar
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
