#!/bin/bash
# Wait for both PCs to finish rendering, give each PC the full dataset, then train:
#   Anna's PC (RTX 3080): B (CNN14 + log-frequency) then A (CNN14 + stock mel)  -- train_anna.cmd
#   this PC (4060 Ti):    C (small from-scratch CNN)
# and finally pull Anna's results back here.
set -u
cd /home/quail/audiodrone
PY=~/venvs/audiodrone/bin/python
ANNA='C:\Users\pc\Joseph AI\audiodrone'
log() { echo "[$(date '+%F %T')] $*"; }

anna_done() {  # both Anna jobs print a final DONE line (held-out runs after the main job in render_anna.cmd)
  ssh -o ConnectTimeout=10 anna "cd \"$ANNA\\data\"; foreach (\$f in 'render_ah.out','render_p2.out') { if (Select-String -Path \$f -Pattern '^DONE' -Quiet -ErrorAction SilentlyContinue) { 'ok' } }" 2>/dev/null \
    | grep -c ok | grep -q 2
}

check_labels() {
  $PY - <<'EOF'
import glob, json, collections
c = collections.Counter(); stale = 0
for js in glob.glob("data/sim_v1/*/*.json"):
    m = json.load(open(js)); c[js.split("/")[-2]] += 1
    stale += m.get("labels") != "snr_lines2"
print("scenes per split", dict(c), "stale labels", stale)
assert stale == 0, "some scenes still have old labels"
EOF
}

log "waiting for renders"
until anna_done && ! systemctl --user is-active -q audiodrone-render; do sleep 120; done

log "sending this PC's scenes to Anna"
tar -cf data/sim_v1_local.tar -C data sim_v1 || exit 1
scp -q data/sim_v1_local.tar "anna:\"Joseph AI/audiodrone/data/\"" || exit 1
ssh anna "cd \"$ANNA\\data\"; tar -xf sim_v1_local.tar; Remove-Item sim_v1_local.tar" || exit 1
rm data/sim_v1_local.tar

log "pulling Anna's scenes"
ssh anna "cd \"$ANNA\\data\"; tar -cf sim_v1_anna.tar sim_v1" || exit 1
scp -q "anna:\"Joseph AI/audiodrone/data/sim_v1_anna.tar\"" data/ || exit 1
tar -xf data/sim_v1_anna.tar -C data && rm data/sim_v1_anna.tar
ssh anna "Remove-Item \"$ANNA\\data\\sim_v1_anna.tar\""
check_labels || exit 1
ssh anna "(Get-ChildItem \"$ANNA\\data\\sim_v1\" -Recurse -Filter *.json).Count" | xargs echo "Anna scene count:"

log "starting B and A on Anna"
ssh anna "\$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{PriorityClass=[uint32]16384; ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c \"$ANNA\\train_anna.cmd\"'; CurrentDirectory='$ANNA'; ProcessStartupInformation=\$si} | Select ReturnValue" || exit 1

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
  scp -q "anna:\"Joseph AI/audiodrone/runs/$r/*\"" runs/$r/ && scp -q "anna:\"Joseph AI/audiodrone/runs/$r.*\"" runs/ || log "pull $r FAILED"
done
log "all done"
