#!/bin/zsh
# Speculative-decoding variations on other runtimes, each against its own no-MTP baseline (same quiz, greedy).
# Runs MTPLX (off, MTP default depth, depth 1) and oMLX (off, adaptive, fixed 1) one after another.
#   MTPLX_VENV / OMLX_VENV: folders holding each tool's own Python venv (installed separately, see README)
#   MTPLX_MODEL: local path of Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed
#   OMLX_MODEL:  local path of Jundot/Qwen3.8-27B-oQ4e-mtp
cd "${0:A:h}"; mkdir -p results
MT=${MTPLX_VENV:-$HOME/mtplx-test}; OM=${OMLX_VENV:-$HOME/omlx-test}
: ${MTPLX_MODEL:?set MTPLX_MODEL} ${OMLX_MODEL:?set OMLX_MODEL}
PY=${PY:-.venv/bin/python}

up() {  # wait for an OpenAI server on port $1, print its first model id
  for i in $(seq 1 600); do
    id=$(curl -s -m 3 http://127.0.0.1:$1/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    [ -n "$id" ] && { echo $id; return; }; sleep 2
  done
}
quiz() {  # port tag
  id=$(up $1); echo "== $2 (model $id) $(date +%H:%M)"
  $PY api_quiz.py --url http://127.0.0.1:$1/v1 --model "$id" --tag $2 2>&1 | grep -E 'SCORE|Error|error'
}
stop() {  # oMLX renames its process, so stop whatever holds the port
  kill $1 2>/dev/null; for i in $(seq 1 60); do lsof -ti :$2 >/dev/null || break; sleep 1; done
  lsof -ti :$2 | xargs kill -9 2>/dev/null; sleep 2
}

for v in "off --no-mtp" "mtp --mtp" "mtp_d1 --mtp --depth 1"; do
  tag=${v%% *}; flags=${v#* }
  $MT/.venv/bin/mtplx serve --model $MTPLX_MODEL --port 8100 --no-auth --ssd-session-cache off ${=flags} > results/mtplx_$tag.log 2>&1 & srv=$!
  quiz 8100 mtplx_$tag; stop $srv 8100
done

N=Jundot--Qwen3.8-27B-oQ4e-mtp
mkdir -p $OM/models $OM/base; ln -sfn "$OMLX_MODEL" $OM/models/$N
for v in 'off|{"mtp_enabled": false}' 'mtp|{"mtp_enabled": true}' 'mtp_d1|{"mtp_enabled": true, "mtp_fixed_depth": 1}'; do
  tag=${v%%|*}; cfg=${v#*|}
  echo "{\"version\": \"1.0\", \"models\": {\"$N\": $cfg}}" > $OM/base/model_settings.json
  OMLX_BASE_PATH=$OM/base $OM/.venv/bin/omlx serve --model-dir $OM/models --port 8200 > results/omlx_$tag.log 2>&1 & srv=$!
  quiz 8200 omlx_$tag; stop $srv 8200
done

python3 - <<'EOF'
import json
for tool in ("mtplx", "omlx"):
    try: base = json.load(open(f"results/spec_{tool}_off.json"))
    except Exception: print(tool, "baseline missing"); continue
    for t in ("off", "mtp", "mtp_d1"):
        try: x = json.load(open(f"results/spec_{tool}_{t}.json"))
        except Exception: print(f"{tool}_{t}: missing"); continue
        same = sum(a["text"] == b["text"] for a, b in zip(base["rows"], x["rows"]))
        print(f"{tool}_{t:7s} {x['score']}/10  {x['avg_tps']:5.2f} tok/s  identical to its baseline {same}/10")
EOF
echo "all done $(date +%H:%M)"
