#!/usr/bin/env bash
# Check (and where it is safe, start) everything the Border watch demo needs, then say what is missing.
#
#   bash scripts/demo_check.sh            # check only
#   bash scripts/demo_check.sh --start    # also start the terra server and the stopped containers
#
# Parts: terra server (:8790), Zetaris (Docker, ~/zetaris/zetaris-platform), the border-ops-db and sensor-db
# containers, the SSH tunnel to the GPU PC (:18088 Nightshift), and a Nemotron reply through Nightshift.
# Reads settings from ~/.config/audiodrone/alerting.env; never prints a secret.
set -uo pipefail
cd "$(dirname "$0")/.."
START=0; [ "${1:-}" = "--start" ] && START=1
ZDIR=${ZETARIS_DIR:-$HOME/zetaris/zetaris-platform}
PY=.venv/bin/python
ok=0; bad=0
pass() { printf '  OK    %s\n' "$1"; ok=$((ok+1)); }
fail() { printf '  FAIL  %s\n        -> %s\n' "$1" "$2"; bad=$((bad+1)); }
dk() { if id -nG | grep -qw docker; then docker "$@"; else sg docker -c "docker $(printf '%q ' "$@")"; fi; }

echo "Border watch demo check"
echo "======================="

# terra server
if curl -s -o /dev/null -m 5 http://127.0.0.1:8790/border; then pass "terra server on http://127.0.0.1:8790/border"
elif [ $START = 1 ]; then
    systemd-run --user --unit ad-terra -p MemoryMax=3G -p MemorySwapMax=0 --working-directory="$PWD" $PY -m terra.server >/dev/null 2>&1
    sleep 6
    curl -s -o /dev/null -m 5 http://127.0.0.1:8790/border && pass "terra server started" || fail "terra server" "see: journalctl --user -u ad-terra"
else fail "terra server not running" "bash scripts/demo_check.sh --start   (or: $PY -m terra.server)"; fi

# Zetaris and the two databases
if [ -d "$ZDIR" ]; then
    [ $START = 1 ] && (cd "$ZDIR" && dk compose up -d >/dev/null 2>&1)
    if (cd "$ZDIR" && dk compose ps --status running --format '{{.Service}}' 2>/dev/null) | grep -q driver; then
        pass "Zetaris running ($ZDIR)"
    else fail "Zetaris not running" "cd $ZDIR && docker compose up -d && ./verify.sh"; fi
else fail "Zetaris bundle not found at $ZDIR" "README section 10, 'Set up Zetaris locally'"; fi
for c in border-ops-db sensor-db; do
    [ $START = 1 ] && dk start "$c" >/dev/null 2>&1
    if [ "$(dk inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" = "true" ]; then pass "container $c"
    else fail "container $c not running" "docker start $c   (created in README section 10)"; fi
done
if $PY scripts/sensor_stores.py check 2>/dev/null | grep -q "fusion.incidents: 6"; then pass "Zetaris answers: 8 stores, fusion.incidents = 6"
else fail "Zetaris federation check" "$PY scripts/sensor_stores.py check   (then register / load if empty)"; fi

# Meterless Nightshift and Nemotron
if systemctl --user is-active -q nightshift-tunnel; then pass "SSH tunnel to the GPU PC (nightshift-tunnel)"
else fail "SSH tunnel down" "systemctl --user start nightshift-tunnel"; fi
reply=$($PY - <<'EOF' 2>/dev/null
from alerting.config import Config, _load_env_file
_load_env_file()
from openai import OpenAI
c = Config()
r = OpenAI(base_url=c.llm_base_url, api_key=c.llm_api_key, timeout=30, max_retries=0).chat.completions.create(
    model=c.llm_model, max_tokens=40, extra_body=c.llm_extra_body or None,
    messages=[{"role": "user", "content": "Reply with one word: ready"}])
print(r.model, "|", (r.choices[0].message.content or "").strip()[:40])
EOF
)
if [ -n "$reply" ]; then pass "Nemotron through Nightshift: $reply"
else fail "no Nemotron reply through Nightshift" "on the GPU PC: open Nightshift (its API must be running); check LLM_* in ~/.config/audiodrone/alerting.env"; fi
if curl -s -m 3 http://127.0.0.1:11434/api/tags | grep -q nemotron-mini; then pass "fallback: Nemotron Mini on local Ollama"
else fail "fallback model missing" "ollama pull nemotron-mini   (optional; the Ask box falls back to it)"; fi

echo
if [ $bad = 0 ]; then echo "READY: open http://127.0.0.1:8790/border"; else echo "$ok ok, $bad to fix (see above)"; fi
[ $bad = 0 ]
