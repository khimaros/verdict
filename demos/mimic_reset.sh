#!/bin/sh
# put the device back to a state that has not been handed the answer.
#
# clearing the app under test is not enough, and that gap cost a sweep: a run
# tapped 'Phone', and every later run started inside the dialler with a popup
# focused, reading two rows. `pm clear` verified ITS OWN ACTION rather than the
# state wanted, which is this project's recurring bug.
#
# so: clear the app, go home, and CHECK we got there. `mimic_reliability.sh`
# already did all three; only the first was wired into the model sweep.
#
# usage: demos/mimic_reset.sh          # env overrides below

set -u
MIMIC="${MIMIC_HOST:-127.0.0.1:18473}"
LANE="${ANDROID_LANE:-mimic-emulator}"
SERIAL="${ANDROID_SERIAL:-emulator-5576}"
PORT="${ANDROID_ADB_PORT:-5054}"
PKG="${SETTINGS_PKG:-com.android.settings}"
TOKEN=$(cat "$HOME/.config/mimic/token")
ADB="$(mise where android-sdk)/platform-tools/adb"
LOCK="$HOME/.claude/skills/android-adb/device_lock.py"

# clear the app UNDER TEST, never mimic itself: clearing mimic wipes its tokens
# and surface toggles and locks the agent out of the device.
ANDROID_LOCK_WHO=verdict python3 "$LOCK" run "$LANE" -- \
    "$ADB" -P "$PORT" -s "$SERIAL" shell pm clear "$PKG" >/dev/null 2>&1

# and leave wherever we are. an app the agent wandered into stays focused
# through a pm clear of something else.
curl -s -m 20 -X POST "http://$MIMIC/v1/global" \
    -H "x-mimic-token: $TOKEN" -H 'content-type: application/json' \
    -d '{"nav":"home"}' -o /dev/null

dump() {
    curl -s -m 20 -X POST "http://$MIMIC/v1/dump" \
        -H "x-mimic-token: $TOKEN" -H 'content-type: application/json' \
        -d '{"filter":"interactive,visible","format":"compact"}'
}

# a home press can land on a popup or an animation, so confirm rather than
# assume. a screen with almost nothing on it is the signature of the bad start
# this exists to prevent.
i=0
while [ "$i" -lt 8 ]; do
    ROWS=$(dump | tr ',' '\n' | grep -c . || echo 0)
    [ "$ROWS" -ge 5 ] && break
    sleep 1
    i=$((i + 1))
done

if [ "${ROWS:-0}" -lt 5 ]; then
    echo "!! reset left only $ROWS rows on screen; the next run starts blind" >&2
    exit 1
fi
exit 0
