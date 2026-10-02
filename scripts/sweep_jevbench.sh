#!/usr/bin/env bash
# run jevbench's public items through verdict's jev endpoint, one model at a time.
#
# jevbench is a third party's benchmark and is fetched, not vendored (NFR4):
# pinned to one commit and checked out beside the other caches. its own
# `typesafe` adapter sends the requests unchanged, so the client is theirs too.
#
# usage: scripts/sweep_jevbench.sh OUT_DIR MODEL [MODEL...]
#   VERDICT_ENDPOINT=http://host:8477   sweep through a verdict endpoint that is
#                         already running, which routes each request to the model
#                         it names. without it a server is started per model here
#   VERDICT_API_KEY=...   the key that endpoint requires, if it requires one
#   LAYOUT=semif          read every model with a named layout (started servers)
#   SERVER_ARGS="--order-averaging 2 --prior-correction"   any other server flags
#   LIMIT=n               a smoke run over the first n items
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
JEVBENCH_REPO=https://github.com/fstandhartinger/jevbench
JEVBENCH_COMMIT=bb05a335bc809e61b20c0f745d25499a82b326fc
JEVBENCH=${JEVBENCH:-${XDG_CACHE_HOME:-$HOME/.cache}/verdict/jevbench}
ENDPOINT=${VERDICT_ENDPOINT:-}
PORT=${VERDICT_SWEEP_PORT:-8495}
# the key a request carries, for curl here and by name for jevbench
AUTH=(); KEY_ENV=
[ -n "${VERDICT_API_KEY:-}" ] && { AUTH=(-H "authorization: Bearer $VERDICT_API_KEY"); KEY_ENV=VERDICT_API_KEY; }
WARM_UP='"state":"warm up","questions":{"q":{"type":"noul","instructions":"is this a test?"}}'
OUT=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift

if [ -z "$ENDPOINT" ]; then
  # the backend comes from the environment or the repo's .env, as for every
  # other script; no eval server's address is written down here
  BASE=$(cd "$ROOT" && PYTHONPATH=$ROOT/python python3 -c \
    'from llama_verdict import config; print(config.backend_url() or "")')
  [ -n "$BASE" ] || { echo "name the backend in the environment or $ROOT/.env, or set VERDICT_ENDPOINT" >&2; exit 2; }
fi

if [ ! -d "$JEVBENCH/.git" ]; then
  git clone --quiet "$JEVBENCH_REPO" "$JEVBENCH" || exit 1
fi
git -C "$JEVBENCH" checkout --quiet "$JEVBENCH_COMMIT" || exit 1
TASKS=$JEVBENCH/datasets/public/easy.jsonl,$JEVBENCH/datasets/public/original.jsonl,$JEVBENCH/datasets/public/hard.jsonl

# start a verdict server bound to one model and wait for it to answer /health
start_server() {
  local model=$1 log=$2 extra=()
  [ -n "${LAYOUT:-}" ] && extra+=(--layout "$LAYOUT")
  # SERVER_ARGS holds several flags, split on whitespace
  [ -n "${SERVER_ARGS:-}" ] && { read -ra flags <<< "$SERVER_ARGS"; extra+=("${flags[@]}"); }
  PYTHONPATH=$ROOT/python python3 -m llama_verdict.server --base-url "$BASE" --model "$model" \
    --port "$PORT" --quiet "${extra[@]}" > "$log" 2>&1 &
  srv=$!
  for _ in $(seq 150); do
    curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null && return 0
    kill -0 $srv 2> /dev/null || return 1
    sleep 2
  done
  return 1
}

for model in "$@"; do
  slug=${model//:/_}
  dir=$OUT/$slug
  mkdir -p "$dir"
  echo "=== $model $(date +%T)"
  srv=; target=$ENDPOINT; ready=1
  if [ -z "$ENDPOINT" ]; then
    target=http://127.0.0.1:$PORT
    start_server "$model" "$dir/server.log" || ready=0
  fi
  # jevbench never retries, so load the model before the first scored item. on
  # an endpoint that routes, this is also what derives and verifies the model
  if [ $ready = 1 ]; then
    ready=0
    for _ in $(seq 60); do
      curl -sf -m 600 "$target/v1/systemone" -H 'content-type: application/json' "${AUTH[@]}" \
        -d "{\"model\":\"$model\",$WARM_UP}" > /dev/null && { ready=1; break; }
      sleep 5
    done
  fi
  if [ $ready = 0 ]; then
    echo "!!! $model did not come up${srv:+; see $dir/server.log}"
    [ -n "$srv" ] && { kill $srv 2> /dev/null; wait $srv 2> /dev/null; }
    continue
  fi
  # an endpoint that was already running has no startup log for this model, so
  # ask it how it read the model: the layout, weights and build a result is
  # evidence beside
  if [ -n "$ENDPOINT" ]; then
    curl -sf -m 60 -G "$target/v1/banner" --data-urlencode "model=$model" "${AUTH[@]}" \
      | python3 -c 'import json, sys; print(json.load(sys.stdin)["banner"])' > "$dir/server.log" \
      || echo "!!! $model: the endpoint did not say how it read the model"
  fi
  # raw responses carry the item text and jevbench's console echoes absolute
  # paths, so both stay outside the repo; one directory per run, since jevbench
  # refuses to overwrite evidence
  scratch=/tmp/verdict-jevbench-raw/$(basename "$OUT")/$slug-$(date +%s)
  mkdir -p "$scratch"
  (cd "$JEVBENCH" && python3 -m jevbench.cli run --tasks "$TASKS" --adapter typesafe \
    --endpoint "$target" --key-env "$KEY_ENV" --model "$model" \
    --cost-basis local_llama_swap_no_tariff --reserve-usd 0 \
    --results "$dir/results.jsonl" --raw-dir "$scratch/raw" \
    --ledger "$dir/ledger.jsonl" --manifest "$dir/manifest.json" \
    ${LIMIT:+--limit $LIMIT}) > "$scratch/run.log" 2>&1
  status=$?
  # jevbench records where it sent the requests, and a running endpoint's
  # address is the eval server's, which no published result may name
  [ -n "$ENDPOINT" ] && [ -f "$dir/manifest.json" ] && python3 - "$dir/manifest.json" <<'EOF'
import json, sys
manifest = json.load(open(sys.argv[1]))
manifest["endpoint"] = "a running verdict endpoint, address withheld"
with open(sys.argv[1], "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
EOF
  echo "run exit=$status $(grep -E 'done:' "$scratch/run.log")"
  [ $status = 0 ] || tail -n 5 "$scratch/run.log"
  (cd "$JEVBENCH" && python3 -m jevbench.cli summarize --tasks "$TASKS" \
    --results "$dir/results.jsonl" --public-export "$dir/summary.json") > "$scratch/summarize.log" 2>&1
  [ -n "$srv" ] && { kill $srv 2> /dev/null; wait $srv 2> /dev/null; }
done
echo "=== all done $(date +%T)"
