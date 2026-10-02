#!/bin/sh
# repeat the android task with and without the planner.
#
# every run starts from the home screen, because a run that begins on the
# answer screen is already finished and would score as a success.
#
# usage: demos/mimic_reliability.sh RUNS

set -u
RUNS="${1:-3}"
# device-localhost via `adb forward`, not a lan bind. on an emulator a lan bind
# lands on the guest's NAT address and is unreachable from the host; the
# forward also survives a reinstall, where the bind preference does not.
#
# the HOST side is 18473 and the device side is mimic's own 8473. host ports
# are GLOBAL to the adb server and nothing warns you: forwarding host 8473 here
# took the port mimic's husky suite forwards to its own phone, and 46 of their
# tests failed looking exactly like product defects. pick a host port nobody
# else would choose, and never the same number as the device port.
MIMIC="${MIMIC_HOST:-127.0.0.1:18473}"
HOST_PORT="${MIMIC##*:}"
DEVICE_PORT="${MIMIC_DEVICE_PORT:-8473}"
VERDICT="${VERDICT_URL:-http://127.0.0.1:8477}"
# the planner defaults to the backend the .env already names for verdict
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLANNER="${PLANNER_URL:-$(cd "$ROOT" && PYTHONPATH="$ROOT/python" python3 -c \
    'from llama_verdict import config; print(config.backend_url() or "")')}"
[ -n "$PLANNER" ] || { echo "set PLANNER_URL, or name the backend in the environment or .env" >&2; exit 2; }
MODEL="${PLANNER_MODEL:-qwen3.5-9b:Q8_0}"
# a single-step goal cannot measure the planner: `step_done` only becomes true
# when the whole goal is satisfied, so the planner is never invoked and both
# arms run identically. measured, 4 steps each, 0 planning calls.
#
# the two halves must also live on different screens. an earlier attempt asked
# for the android version and then free storage, and storage is printed in the
# summary line of the Settings home row -- so half the goal was answered before
# any navigation happened, and the run ended in one step. About phone and
# Display are two screens with a BACK between them.
#
# "read-only" has to mean the SCREENS are informational, not merely that the
# goal is phrased as a question. asking whether dark theme is on pointed the
# agent at a screen whose main control is the dark theme switch, and it tapped
# it seven times: night mode was on afterwards and the emulator default is off.
# an agent with tap authority will operate whatever it is looking at, so the
# benchmark has to choose screens with nothing on them worth operating.
#
# About phone and Storage are informational. the storage figure must be one
# that only appears on the Storage screen, because the free-space summary is
# printed on the Settings home row and a goal satisfied there never navigates.
GOAL="${GOAL:-First open About phone and find the Android version. Once you have seen it, go back and open the Storage screen and find how much space Apps are using.}"
# the requirements are the score and must match the goal. overriding GOAL and
# leaving these behind measures the wrong thing: a single-screen goal was once
# scored against a storage requirement it could never satisfy.
# no colon: `${VAR:-default}` substitutes when VAR is empty OR unset, so
# REQUIRE2="" would silently fall back to the storage pattern and score a
# single-screen goal against a fact it can never reach. `${VAR-default}`
# substitutes only when unset, which is what lets REQUIRE2="" mean "one
# requirement, not two".
REQUIRE1="${REQUIRE1-Android version}"
REQUIRE2="${REQUIRE2-Apps / [0-9]}"
OUT="${OUT_DIR:-eval/results}"
TOKEN=$(cat "$HOME/.config/mimic/token")

mkdir -p "$OUT"

# going home is not enough: Settings stays in the back stack, so the next
# launch resumes whatever screen it was left on. measured -- a run "succeeded"
# in one step because it opened directly onto Android version.
#
# `am force-stop` is not enough either, and that is the same bug wearing a
# hat: it kills the process and leaves the task's saved activity state, so the
# app comes back to the screen it was on. `pm clear` drops the task and the
# app data and gives a genuine first-run screen.
#
# `pm clear` is destructive, so it is used only on an emulator lane, where the
# device exists to be reset. on a real phone this falls back to force-stop and
# leans on the start-screen guard below, which detects a bad start rather than
# preventing one. clear the app UNDER TEST, never mimic itself: clearing mimic
# wipes its tokens and surface toggles and locks the agent out of the device.
ADB="$(mise where android-sdk)/platform-tools/adb"
LOCK="$HOME/.claude/skills/android-adb/device_lock.py"
# every one of these is overridable, because the device this runs against is
# not verdict's to assume: husky is shared and an emulator lane may replace it
SERIAL="${ANDROID_SERIAL:-emulator-5576}"
PORT="${ANDROID_ADB_PORT:-5054}"
LANE="${ANDROID_LANE:-mimic-emulator}"
SETTINGS_PKG="${SETTINGS_PKG:-com.android.settings}"

dump_ids() {
    curl -s -m 30 -X POST "http://$MIMIC/v1/dump" \
        -H "x-mimic-token: $TOKEN" -H 'content-type: application/json' \
        -d '{"filter":"interactive","format":"compact"}'
}

case "$LANE" in
    *emulator*) RESET_CMD="pm clear $SETTINGS_PKG" ;;
    *)          RESET_CMD="am force-stop $SETTINGS_PKG" ;;
esac

# an unknown lane name reports as FREE rather than as an error: `device_lock.py
# status nosuchlane` exits 0 saying "nosuchlane is FREE". so a typo in a lane
# does not fail, it silently means no lock at all while looking like a held
# one. check the name against the allocation before relying on it.
LANES="$HOME/.claude/skills/android-adb/lanes.json"
python3 -c "
import json, sys
devices = json.load(open(sys.argv[1])).get('devices', [])
sys.exit(0 if any(d.get('name') == sys.argv[2] for d in devices) else 1)
" "$LANES" "$LANE" || {
    echo "!! lane '$LANE' is not in $LANES. an unallocated lane locks nothing." >&2
    exit 1
}

# `adb forward` is per adb-server and does not survive a server restart, so it
# is re-asserted rather than assumed. idempotent, and it must come after the
# variables it uses are set.
ANDROID_LOCK_WHO=verdict python3 "$LOCK" run "$LANE" -- \
    "$ADB" -P "$PORT" -s "$SERIAL" forward "tcp:$HOST_PORT" "tcp:$DEVICE_PORT" \
    >/dev/null 2>&1 \
    || { echo "!! could not forward $HOST_PORT; is $LANE up and free?" >&2; exit 1; }

# and then CHECK WHERE IT POINTS, matching on the serial. `adb forward` on a
# host port another device already holds does not fail loudly, so a successful
# call is not evidence the port is ours -- the run would drive somebody else's
# phone while reporting on this one. the serial is the only part of that line
# that distinguishes the two.
ANDROID_LOCK_WHO=verdict python3 "$LOCK" run "$LANE" -- \
    "$ADB" -P "$PORT" forward --list > /tmp/verdict-forwards.txt 2>&1
grep -q "^$SERIAL tcp:$HOST_PORT tcp:$DEVICE_PORT\$" /tmp/verdict-forwards.txt || {
    echo "!! host port $HOST_PORT is not forwarded to $SERIAL. current mappings:" >&2
    cat /tmp/verdict-forwards.txt >&2
    exit 1
}

# the lock serialises access; the token does not live under it. another
# lock-holder that mints a token revokes ours, and we find out as a 401 partway
# through the next run -- which looks exactly like agent failure in the
# results. observed twice. so re-pair whenever the token has stopped working,
# after the lock is ours and before anything is scored.
probe_token() {
    curl -s -m 15 -o /dev/null -w '%{http_code}' -X POST "http://$MIMIC/v1/dump" \
        -H "x-mimic-token: $(cat "$HOME/.config/mimic/token")" \
        -H 'content-type: application/json' -d '{"filter":"interactive","format":"compact"}'
}

ensure_token() {
    # "cannot connect" and "not allowed" are different faults and only one of
    # them is fixed by re-pairing. curl reports a connection failure as 000,
    # and treating that as a bad token re-pairs for nothing -- which revokes
    # every other client and takes the lock, to cure a blip that cures itself.
    code=$(probe_token)
    attempt=1
    while [ "$code" = "000" ] && [ "$attempt" -le 3 ]; do
        echo "  mimic unreachable (attempt $attempt); waiting" >&2
        sleep 3
        code=$(probe_token)
        attempt=$((attempt + 1))
    done

    case "$code" in
        200) return 0 ;;
        401|403) ;;
        *) echo "  !! mimic returned $code and is not merely unauthorised" >&2
           return 1 ;;
    esac

    echo "  token rejected ($code); re-pairing" >&2
    python3 demos/mimic_pair.py >/dev/null 2>&1 || {
        echo "  !! could not re-pair" >&2; return 1; }
    TOKEN=$(cat "$HOME/.config/mimic/token")
    return 0
}

reset_home() {
    ensure_token || return 1
    # shellcheck disable=SC2086
    ANDROID_LOCK_WHO=verdict python3 "$LOCK" run "$LANE" -- \
        "$ADB" -P "$PORT" -s "$SERIAL" shell $RESET_CMD >/dev/null 2>&1
    curl -s -m 30 -X POST "http://$MIMIC/v1/global" \
        -H "x-mimic-token: $TOKEN" -H 'content-type: application/json' \
        -d '{"nav":"home"}' -o /dev/null
    # a start screen that already mentions the answer invalidates the run
    if dump_ids | grep -qi 'android version'; then
        echo "  !! still showing the answer after reset; refusing to score" >&2
        return 1
    fi
    return 0
}

for mode in planner noplanner; do
    echo "=== $mode"
    i=1
    while [ "$i" -le "$RUNS" ]; do
        if ! reset_home; then
            printf '  run %s: SKIPPED, bad start state\n' "$i"
            i=$((i + 1))
            continue
        fi
        summary="$OUT/mimic-$mode-run$i.json"
        # the score is what reached the screen, not what the agent claims. a
        # run reported `done` having answered only the second half of this
        # goal, never opening the screen that holds the first.
        set -- --mimic "$MIMIC" --verdict "$VERDICT" --goal "$GOAL" \
               --require "$REQUIRE1" ${REQUIRE2:+--require "$REQUIRE2"} \
               --launch --min-confidence 0.5 --max-steps 14 --summary "$summary"
        [ "$mode" = planner ] && set -- "$@" --planner-url "$PLANNER" --planner-model "$MODEL"
        # the agent drives the device over mimic's http surface, not over adb,
        # so it is invisible to device_lock unless the whole run is wrapped in
        # it. it was invisible, and another agent took the same phone while a
        # run was in flight. holding the lock around the run is what makes
        # `device_lock.py list` tell the truth about this kind of client.
        ANDROID_LOCK_WHO=verdict timeout 900 python3 "$LOCK" run "$LANE" -- \
            python3 demos/mimic_agent.py "$@" > "/tmp/mimic-$mode-$i.log" 2>&1
        [ $? -eq 75 ] && printf '  run %s: device busy, skipped\n' "$i"
        printf '  run %s: %s | %s\n' "$i" \
            "$(grep -E '^status' "/tmp/mimic-$mode-$i.log" || echo 'no status')" \
            "$(grep -E '^witnessed' "/tmp/mimic-$mode-$i.log" || echo 'witnessed none')"
        i=$((i + 1))
    done
done
