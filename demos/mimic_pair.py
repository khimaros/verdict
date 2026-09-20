#!/usr/bin/env python3
"""take sole ownership of a mimic device and save a token.

mimic deliberately has no remote minting path: minting outside pairing, and
all revocation, are gui-only so that a client holding a token cannot mint
itself more or revoke its peers. that property is worth keeping, so this does
not work around it. it drives the app's own gui over adb, which is the
physical-access path automated by something that already has adb on the
device, and reuses mimic's own e2e helper rather than reimplementing it.

**this revokes every existing paired client.** that is how the helper keeps
the onboarding screen predictable, and it means this is a "take sole ownership"
call rather than a "get me a token alongside yours" one. pointed at a shared
phone it would break whoever else is paired, so it refuses anything that is
not an emulator lane unless told otherwise in as many words.

usage:
  ./demos/mimic_pair.py                      # the mimic emulator lane
  ./demos/mimic_pair.py --lane husky --yes-revoke-other-clients
"""

import argparse
import os
import pathlib
import subprocess
import sys

MIMIC_TESTS = pathlib.Path.home() / "src/github.com/khimaros/mimic/tests"
LOCK = pathlib.Path.home() / ".claude/skills/android-adb/device_lock.py"
TOKEN_FILE = pathlib.Path.home() / ".config/mimic/token"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane", default="mimic-emulator")
    ap.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL", "emulator-5576"))
    ap.add_argument("--port", default=os.environ.get("ANDROID_ADB_SERVER_PORT", "5054"))
    ap.add_argument("--yes-revoke-other-clients", action="store_true",
                    help="required on any lane that is not an emulator")
    args = ap.parse_args()

    if "emulator" not in args.lane and not args.yes_revoke_other_clients:
        sys.exit(f"refusing: pairing revokes every client on {args.lane!r}, which is "
                 f"not an emulator lane. someone else is probably paired to it. "
                 f"pass --yes-revoke-other-clients if you are certain.")

    # hold the lane for the whole thing: this drives the gui, and a concurrent
    # driver would fight it tap for tap
    status = subprocess.run([sys.executable, str(LOCK), "status", args.lane],
                            capture_output=True, text=True, check=False)
    if status.returncode == 75:
        sys.exit(f"{args.lane} is held:\n{status.stdout.strip()}")

    env = {**os.environ, "ANDROID_SERIAL": args.serial,
           "ANDROID_ADB_SERVER_PORT": str(args.port), "ANDROID_LOCK_WHO": "verdict"}
    script = (
        f"import sys; sys.path.insert(0, {str(MIMIC_TESTS)!r});\n"
        "import adb;\n"
        "adb.forward();\n"
        "print(adb.reveal_token())\n")
    proc = subprocess.run(
        [sys.executable, str(LOCK), "run", args.lane, "--", sys.executable, "-c", script],
        env=env, capture_output=True, text=True, check=False)

    if proc.returncode == 75:
        sys.exit(f"{args.lane} became busy; nothing was changed")
    if proc.returncode != 0:
        sys.exit(f"pairing failed ({proc.returncode}):\n{proc.stderr.strip()}")

    token = proc.stdout.strip().splitlines()[-1].strip()
    if not token:
        sys.exit("no token came back; the app may not be on the clients tab")

    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)
    print(f"token saved to {TOKEN_FILE} ({len(token)} chars)")
    # mimic's own `adb.forward()` puts the device on host 8473, and host ports
    # are global to the adb server -- that number is also what mimic's suite
    # forwards to its phone. so verdict's runs use host 18473, set up and
    # checked against the serial by demos/mimic_reliability.sh, and this line
    # reports what was actually asked for rather than a port to rely on.
    print("paired. mimic's helper forwarded host 8473; verdict's runs use "
          "18473 and assert it themselves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
