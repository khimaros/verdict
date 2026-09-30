#!/usr/bin/env bash
# run jevbench's public items through verdict's jev endpoint, one model at a time.
#
# jevbench is a third party's benchmark and is fetched, not vendored (NFR4):
# pinned to one commit and checked out beside the other caches. its own
# `typesafe` adapter sends the requests unchanged, so the client is theirs too.
#
# usage: scripts/sweep_jevbench.sh OUT_DIR MODEL [MODEL...]
#   LAYOUT=semif          read every model with a named layout
#   SERVER_ARGS="--order-averaging 2 --prior-correction"   any other server flags
#   LIMIT=n               a smoke run over the first n items
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
JEVBENCH_REPO=https://github.com/fstandhartinger/jevbench
JEVBENCH_COMMIT=bb05a335bc809e61b20c0f745d25499a82b326fc
JEVBENCH=${JEVBENCH:-${XDG_CACHE_HOME:-$HOME/.cache}/verdict/jevbench}
# the backend comes from the environment or the repo's .env, as for every other
# script; no eval server's address is written down here
BASE=${LLAMA_VERDICT_URL:-$(sed -n 's/^LLAMA_VERDICT_URL=//p' "$ROOT/.env" 2>/dev/null)}
[ -n "$BASE" ] || { echo "set LLAMA_VERDICT_URL, or add it to $ROOT/.env" >&2; exit 2; }
PORT=${VERDICT_SWEEP_PORT:-8495}
OUT=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift

if [ ! -d "$JEVBENCH/.git" ]; then
  git clone --quiet "$JEVBENCH_REPO" "$JEVBENCH" || exit 1
fi
git -C "$JEVBENCH" checkout --quiet "$JEVBENCH_COMMIT" || exit 1
TASKS=$JEVBENCH/datasets/public/easy.jsonl,$JEVBENCH/datasets/public/original.jsonl,$JEVBENCH/datasets/public/hard.jsonl

for model in "$@"; do
  slug=${model//:/_}
  dir=$OUT/$slug
  mkdir -p "$dir"
  extra=()
  case $model in gpt-oss-*) extra=(--assistant-open '<|start|>assistant<|channel|>final<|message|>');; esac
  [ -n "${LAYOUT:-}" ] && extra+=(--layout "$LAYOUT")
  # SERVER_ARGS holds several flags, split on whitespace
  [ -n "${SERVER_ARGS:-}" ] && { read -ra flags <<< "$SERVER_ARGS"; extra+=("${flags[@]}"); }
  echo "=== $model $(date +%T)"
  PYTHONPATH=$ROOT/python python3 -m llama_verdict.server --base-url "$BASE" --model "$model" \
    --port "$PORT" --quiet "${extra[@]}" > "$dir/server.log" 2>&1 &
  srv=$!
  ready=0
  for _ in $(seq 150); do
    curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null && { ready=1; break; }
    kill -0 $srv 2> /dev/null || break
    sleep 2
  done
  # jevbench never retries, so load the model before the first scored item
  if [ $ready = 1 ]; then
    ready=0
    for _ in $(seq 60); do
      curl -sf -m 300 "http://127.0.0.1:$PORT/v1/systemone" -H 'content-type: application/json' \
        -d '{"state":"warm up","questions":{"q":{"type":"noul","instructions":"is this a test?"}}}' \
        > /dev/null && { ready=1; break; }
      sleep 5
    done
  fi
  if [ $ready = 0 ]; then
    echo "!!! $model did not come up; see $dir/server.log"
    kill $srv 2> /dev/null; wait $srv 2> /dev/null
    continue
  fi
  # raw responses carry the item text and jevbench's console echoes absolute
  # paths, so both stay outside the repo; one directory per run, since jevbench
  # refuses to overwrite evidence
  scratch=/tmp/verdict-jevbench-raw/$(basename "$OUT")/$slug-$(date +%s)
  mkdir -p "$scratch"
  (cd "$JEVBENCH" && python3 -m jevbench.cli run --tasks "$TASKS" --adapter typesafe \
    --endpoint "http://127.0.0.1:$PORT" --key-env '' --model "$model" \
    --cost-basis local_llama_swap_no_tariff --reserve-usd 0 \
    --results "$dir/results.jsonl" --raw-dir "$scratch/raw" \
    --ledger "$dir/ledger.jsonl" --manifest "$dir/manifest.json" \
    ${LIMIT:+--limit $LIMIT}) > "$scratch/run.log" 2>&1
  status=$?
  echo "run exit=$status $(grep -E 'done:' "$scratch/run.log")"
  [ $status = 0 ] || tail -n 5 "$scratch/run.log"
  (cd "$JEVBENCH" && python3 -m jevbench.cli summarize --tasks "$TASKS" \
    --results "$dir/results.jsonl" --public-export "$dir/summary.json") > "$scratch/summarize.log" 2>&1
  kill $srv 2> /dev/null; wait $srv 2> /dev/null
done
echo "=== all done $(date +%T)"
