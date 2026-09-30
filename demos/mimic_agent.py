#!/usr/bin/env python3
"""an android agent: a generative planner, a typed decider, and nothing else.

the split is the point. planning is generative and infrequent; deciding is
typed and happens every step. a planner invoked per step would reinstate
exactly the cost verdict exists to remove, so the planner runs only when a
typed boolean says the current sub-goal is satisfied.

    every step      one prefill, three typed questions, no tokens generated
    per sub-goal    one generative call

decisions go over the jev wire protocol, so this is a jev client in the same
sense browser-use/jev-ultrafast is. see docs/JEV_API.md.

mimic (../mimic) supplies the device surface: an accessibility tree filtered on
the device, and back/home/recents as first-class operations. neither of the two
walls the browser demo hit applies here -- see docs/DEMOS.md.

usage:
  ./demos/mimic_agent.py --mimic <device-host>:8473 \
      --verdict http://127.0.0.1:8477 --goal "Open Settings and find the Android version"
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shortlist import restore_answers, shortlist_request

TOKEN_FILE = os.path.expanduser("~/.config/mimic/token")

# what the device can be asked to do. every one of these is a real mimic verb,
# so an operation the model picks can always be carried out.
OPERATIONS = {
    "TAP": "Tap an element on the screen.",
    "LAUNCH": "Open an app directly by name, without finding its icon first.",
    "SCROLL_DOWN": "Scroll the screen down to reveal what is below.",
    "SCROLL_UP": "Scroll the screen up to reveal what is above.",
    "BACK": "Go back to the previous screen. Always available.",
    "HOME": "Go to the device home screen. Always available.",
    "DONE": "The goal is visibly satisfied on this screen.",
    "BLOCKED": "No available operation can make progress.",
}

# operations that need a target element; the rest act on the whole device
NEEDS_TARGET = {"TAP"}

MAX_STEPS = 40


class Mimic:
    """mimic's http surface. every call is POST /v1/<verb> with a flat body."""

    def __init__(self, host, token):
        self.base = f"http://{host}/v1"
        self.token = token

    def call(self, verb, **body):
        """one mimic call, reloading the token if it has been revoked.

        mimic's token surface grants sole ownership: anyone who mints one
        revokes every other client, and that revocation outlives the device
        lock, so serialising access does not prevent it. checking the token
        before a run is not enough either -- a run died on its first screen
        read, inside the seconds between the check passing and the agent
        starting.

        so recover where it happens. the token file is the shared source of
        truth, and re-reading it costs nothing next to another lost run.
        """
        for attempt in (1, 2):
            req = urllib.request.Request(
                f"{self.base}/{verb.lower()}",
                data=json.dumps({k: str(v) for k, v in body.items()}).encode(),
                headers={"x-mimic-token": self.token,
                         "content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    out = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code != 401 or attempt == 2 or not self._reload_token():
                    raise
        if not out.get("ok"):
            raise RuntimeError(f"mimic {verb}: {out.get('error')}")
        return out.get("data")

    def _reload_token(self):
        """pick up a token minted by something else since we started."""
        try:
            with open(TOKEN_FILE) as f:
                fresh = f.read().strip()
        except OSError:
            return False
        if not fresh or fresh == self.token:
            return False
        print("  token was revoked; reloaded from disk", file=sys.stderr)
        self.token = fresh
        return True

    def screen(self):
        # interactive AND visible: reachable rows only, no negative centres.
        # `label` carries a name borrowed from descendants, `text` the node's
        # own, and mimic omits `label` when `text` already says it.
        return parse_screen(self.call(
            "DUMP", filter="interactive,visible", format="flat",
            fields="class,text,label,id,center,actions"))

    def settled_screen(self, tries=6, pause=0.4):
        """the screen once a window actually exists to read.

        "the app has been started" and "a window is up and belongs to it" are
        different claims, and querying between them gets `no active window` or,
        worse, whatever the previous step left on screen.
        """
        for attempt in range(tries):
            try:
                screen = self.screen()
            except RuntimeError:
                screen = []
            if screen:
                return screen
            if attempt < tries - 1:
                time.sleep(pause)
        return []

    def screen_after(self, previous, tries=8, pause=0.4):
        """the screen once the action has actually landed.

        "a screen exists" and "the screen changed" are different questions and
        settled_screen only answers the first. read immediately after a tap it
        returns the OLD screen, because the transition has not happened yet:

            +0.0s  unchanged, still the Settings list
            +0.5s  changed, now the About screen

        so an action that worked was reported as doing nothing, the operation
        that did it was retired, and the next decision was taken against a
        stale reading. waits for a change and gives up honestly if none comes,
        which is the only way to tell "it did nothing" from "it has not
        happened yet".
        """
        for attempt in range(tries):
            screen = self.settled_screen()
            if signature(screen) != previous:
                return screen, True
            if attempt < tries - 1:
                time.sleep(pause)
        return screen, False

    def packages(self):
        """launchable apps with human labels.

        an app's label is far more discriminative than the id of whatever icon
        happens to be on screen, and launching by package skips icon hunting
        entirely: on this device Settings lives inside a launcher folder, which
        is two correct taps before anything useful happens.
        """
        out = self.call("PACKAGES")
        return out if isinstance(out, list) else []

    def act(self, operation, element, package=None):
        """carry out one chosen operation. mimic has a verb for each."""
        if operation == "TAP":
            return self.call("TAP", x=element["x"], y=element["y"])
        if operation == "LAUNCH":
            return self.call("LAUNCH", package=package, wait="true")
        if operation in ("SCROLL_DOWN", "SCROLL_UP"):
            return self.call("SCROLL", direction="down" if "DOWN" in operation else "up")
        if operation in ("BACK", "HOME"):
            return self.call("GLOBAL", nav=operation.lower())
        raise ValueError(f"no device verb for {operation}")


def parse_screen(nodes):
    """the nodes this agent can tap, named.

    mimic labels a row from the text it contains, stopping at any clickable
    descendant. that is done on the device, so there is no join to do here: a
    borrowed name arrives as `label` and a node's own text as `text`, and the
    two stay distinguishable because `label` is omitted when `text` already
    says it.

    the remaining filter is this agent's own and is a different question from
    mimic's. mimic answers "what can be acted on at all", which rightly
    includes a ScrollView, since `scroll` is a real action on one. a tap
    target additionally needs `click`, because a tap on a scroll container is
    delivered to its centre point and lands on whichever row is mid-screen.
    """
    elements = []
    for node in nodes if isinstance(nodes, list) else []:
        if "click" not in (node.get("actions") or []):
            continue
        cx, cy = (node.get("center") or [-1, -1])[:2]
        text = (node.get("text") or node.get("label") or "").strip()
        node_id = (node.get("id") or "").rpartition("/")[2]
        if not text and not node_id:
            continue
        elements.append({"index": str(len(elements) + 1), "x": int(cx), "y": int(cy),
                         "class": (node.get("class") or "").rpartition(".")[2],
                         "text": text, "id": node_id,
                         # whether the name is the node's own or borrowed from
                         # what it contains, so a report can say which
                         "borrowed": not node.get("text") and bool(node.get("label"))})
    return elements


def signature(screen):
    """what the screen looks like, for deciding whether an action did anything."""
    return tuple(sorted((e["text"] or e["id"]) for e in screen))


class Witnessed:
    """the answers the run actually found, and where it found them.

    two separate jobs live here, and conflating them is how an agent ends up
    "answering" a question it never read.

    **navigation is a decision and extraction is not.** this agent emits only
    typed choices and never generates a token, so it can reach the screen that
    holds the android version but it cannot say "17". that value is not
    something a model needs to produce: it is already structured data in the
    accessibility tree, sitting in the row the agent navigated to. reading it
    out is a string operation, not an inference, and doing it with a model
    would be slower, more expensive and less reliable than a substring match.

    so the agent decides where to go, and this records what was there. the
    agent's own `status` is scored nowhere: a run reported `done` having
    answered the second half of a two-part goal and never opened the screen
    holding the first half.
    """

    def __init__(self, patterns):
        # regex, because a plain substring is not selective enough to name a
        # fact: `Apps` matched `Apps / Recent apps, default apps` on the
        # settings home row and scored the storage half of a goal that had
        # never been reached. `Apps / [0-9]` names the row that carries a size.
        self.patterns = [p for p in patterns]
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]
        self.found = {}

    def observe(self, screen, step):
        for pattern, rx in zip(self.patterns, self._compiled, strict=True):
            if pattern in self.found:
                continue
            for element in screen:
                row = element["text"] or element["id"]
                if rx.search(row):
                    # the whole row, because the value travels with its label:
                    # mimic renders these as "Android version / 17"
                    self.found[pattern] = {"value": row, "step": step}
                    break

    def report(self):
        return {"required": len(self.patterns), "witnessed": len(self.found),
                "complete": len(self.found) == len(self.patterns),
                "answers": {k: v["value"] for k, v in self.found.items()},
                "at_step": {k: v["step"] for k, v in self.found.items()},
                "missing": [p for p in self.patterns if p not in self.found]}


def live_operations(dead, acted=True):
    """the operations still worth offering on this screen.

    an operation that left the screen unchanged cannot be the way forward, and
    offering it again invites the model to choose it again: measured, eleven
    consecutive SCROLL_DOWNs at 0.81 to 0.96 confidence. the model is not
    confused, it is answering the question it was asked. removing the dead
    option changes the question.

    DONE is withheld until something has been done. a run that starts from a
    reset home screen cannot already satisfy its goal, and offering the claim
    anyway got it taken: twice, at 0.95 confidence, on the launcher, after zero
    actions. a confidence gate cannot catch that, because the model is not
    hesitant. removing the option it cannot truthfully pick is the only thing
    that does.
    """
    live = {k: v for k, v in OPERATIONS.items() if k not in dead}
    if not acted:
        live.pop("DONE", None)
    return live


def tappable_now(screen):
    """the rows this agent can actually tap.

    named or identified, and on the screen. an element with a negative
    coordinate is not merely a poor candidate: android's gesture dispatcher
    refuses it outright, and granite-4.2-3b lost a run to
    `mimic TAP: Path bounds must not be negative` on its first action. a
    MISSING coordinate is not a negative one and is left alone.
    """
    kept = []
    for e in screen:
        if not (e.get("text") or e.get("id")):
            continue
        if any(e.get(k, 0) < 0 for k in ("x", "y")):
            continue
        kept.append(e)
    return kept


class Exhausted:
    """targets whose destination is a screen the agent has already left.

    `DeadEnds` retires an operation that changed nothing. this retires a
    TARGET that changes the screen to one already finished with, which is the
    two-state oscillation DeadEnds cannot see: measured, TAP 'Model' / BACK
    four times until the step budget ran out, every one of those actions
    genuinely changing the screen so none was ever a no-op.

    it is the android form of the browser agent's `--retire-read`, and it
    narrows the ACTION SPACE without touching the state -- the row stays
    visible as context and merely stops being offered as a destination, which
    is the distinction the browser's trimming a/b showed matters.

    going somewhere is not finishing with it: only a screen the agent came
    back from counts as left.
    """

    def __init__(self):
        self._destination = {}
        self._left = set()

    def record(self, before, index, after):
        if index is not None and after is not None:
            self._destination[(before, index)] = after

    def leaving(self, signature):
        self._left.add(signature)

    def offer(self, signature, elements):
        kept = [e for e in elements
                if self._destination.get((signature, e["index"])) not in self._left]
        # an empty candidate list is worse than a stale one: the agent cannot
        # answer at all, and BLOCKED on no options says nothing about the screen
        return kept or elements


class DeadEnds:
    """which operations have already done nothing, per screen.

    keyed by screen rather than reset whenever the screen changes. the reset
    version looked right and did not work: the check after an action and the
    read at the top of the next step are two reads of a settling ui, and any
    difference between them wiped the record. measured, six consecutive
    identical taps each correctly retired and then immediately offered again.

    keying by signature also means a screen revisited later still remembers
    what was useless on it, which the reset version threw away.
    """

    def __init__(self):
        self._by_screen = {}

    def on(self, sig):
        return frozenset(self._by_screen.get(sig, ()))

    def add(self, sig, operation):
        self._by_screen.setdefault(sig, set()).add(operation)
        return self.on(sig)


def as_target(subgoal):
    """frame a sub-goal as outstanding, not as a fact.

    a sub-goal phrased as an outcome -- "The Settings app is open and the
    screen showing the Android version is visible" -- sitting in a field
    called `current_step` reads as an assertion that it IS open and IS
    visible. the model then answers DONE, correctly, from a false premise.

    measured on the emulator home screen, same screen, same everything, only
    the phrasing varying:

        declarative                 LAUNCH 0.006   DONE 0.779
        "NOT YET DONE. Target: ..."  LAUNCH 0.467   DONE 0.019
        imperative                   LAUNCH 0.102   DONE 0.008

    done here rather than in the planner prompt so that it holds whatever
    wording the planner returns.
    """
    return f"NOT YET DONE. Target: {subgoal.rstrip('.')}."


def build_state(goal, subgoal, screen, history, progress):
    """the shared prefix: prefilled once, reused by every question about it."""
    return {
        "goal": goal,
        "current_step": as_target(subgoal),
        "progress": progress,
        "screen": [{k: e[k] for k in ("index", "class", "text", "id") if e[k]}
                   for e in screen],
        "recent_actions": history[-8:],
    }


def target_request(model, state, goal, subgoal, tappable, apps):
    """the targets, and whether the step is finished. none need the operation."""
    questions = {
        "tap_target": {
            "type": "choice",
            "criteria": {e["index"]: {"element": e["text"] or e["id"],
                                      "kind": e["class"]} for e in tappable},
            "instructions": {"goal": goal, "current_step": as_target(subgoal),
                             "rules": ["Choose the element most likely to advance "
                                       "the current step if tapped.",
                                       "Choose only an offered element."]},
        },
        "step_done": {
            "type": "boolean",
            "instructions": {"current_step": as_target(subgoal),
                             "rules": ["Is the current step already satisfied by "
                                       "what is on this screen?"]},
        },
    }
    if apps:
        questions["launch_target"] = {
            "type": "choice",
            "criteria": {a["package"]: a["label"] for a in apps},
            "instructions": {"goal": goal, "current_step": as_target(subgoal),
                             "rules": ["Which app would you open to advance the "
                                       "current step, if you were to open one?"]},
        }
    return {"model": model, "state": state, "questions": questions}


def operation_request(model, state, goal, subgoal, best, best_confidence,
                      app=None, app_confidence=0.0, dead=frozenset(),
                      acted=True):
    """which operation, told what tapping would actually do.

    asked separately and second on purpose. the speculative fan-out the jev
    clients use asks operation and target independently against one state, so
    the operation head cannot see that the target head is certain. measured on
    a launcher screen: the target head put 0.9991 on 'Folder: Settings' while
    the operation head spread 0.44/0.16/0.13 and picked SCROLL_DOWN. naming the
    winning element in this question is what closes that gap.
    """
    described = f"{best['element']} ({best['kind']})" if best else "nothing useful"
    instructions = {
        "goal": goal, "current_step": as_target(subgoal),
        "if_you_tap": f"Tapping would hit: {described}",
        "tap_confidence": round(best_confidence, 3),
        # two rewrites of these rules were tried against a launcher where the
        # destination app has no icon, to move the operation head off
        # SCROLL_DOWN. neither helped: LAUNCH went 0.0068 -> 0.0021, and
        # naming the candidate in the option itself pushed BACK to 0.85
        # instead. that matches what every other measurement on this project
        # says -- prompt-layer changes do nothing here, and the action space is
        # where the wins are. left as it was rather than accumulating churn.
        "rules": ["Choose the one operation that advances the current step.",
                  "If the element named above would advance the step, answer TAP.",
                  "Prefer LAUNCH over hunting for an app icon on the screen.",
                  "Scroll only when what you need is plainly not on this screen.",
                  "Only answer DONE when the goal is visible right now."],
    }
    if app:
        instructions["if_you_launch"] = f"Launching would open: {app}"
        instructions["launch_confidence"] = round(app_confidence, 3)
    if dead:
        instructions["already_tried_and_changed_nothing"] = sorted(dead)
    return {"model": model, "state": state, "questions": {"operation": {
        "type": "choice", "criteria": live_operations(dead, acted),
        "instructions": instructions}}}


def post(url, body, timeout, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"content-type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def plan(planner_url, planner_model, goal, progress, history, screen):
    """the generative half, and the only place tokens are produced.

    called when the typed step_done question says the current step is finished,
    so its cost scales with sub-goals rather than with actions.
    """
    prompt = (
        f"Goal: {goal}\n"
        f"Sub-goals already finished: {progress or 'none yet'}\n"
        f"Recent actions: {json.dumps(history[-6:])}\n"
        f"Visible now: {json.dumps([e['text'] or e['id'] for e in screen][:25])}\n\n"
        "Name the next SUB-GOAL: what must be true next for the goal to "
        "progress. Describe a destination or an outcome.\n"
        "Do NOT name an action. Do not say tap, click, scroll or press, and do "
        "not name a button. Something else decides which action gets there; "
        "naming one here only competes with it.\n"
        "DO name the app the destination lives in, every time. 'The About "
        "phone screen is open' sent an agent hunting the launcher for "
        "something called Phone and it tapped the dialler; 'The About phone "
        "screen inside the Settings app is open' does not.\n"
        "If the goal states an order, respect it: the first unfinished "
        "sub-goal is the answer, even if a later one looks easier from here.\n"
        "Reply with ONE short sentence. No preamble, no numbering.")
    body = {"model": planner_model, "max_tokens": 60, "temperature": 0.0,
            # these are reasoning models, and with thinking left on the whole
            # budget goes into it: 60 tokens spent, content came back empty
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    out = post(f"{planner_url.rstrip('/')}/chat/completions", body, 300,
               {"authorization": "Bearer local"})
    text = (out["choices"][0]["message"].get("content") or "").strip()
    # a planner that returns nothing must not erase the step that was working
    return text.splitlines()[0] if text else None


def ask(verdict_url, request, keep):
    dropped = {}
    if keep:
        request, _, dropped = shortlist_request(request, keep)
    result = post(verdict_url.rstrip("/") + "/v1/systemone", request, 600)
    return restore_answers(result, dropped)["answers"]


def decide(verdict_url, model, state, goal, subgoal, tappable, apps, keep,
           min_confidence, dead=frozenset(), acted=True, done_confidence=None):
    """targets first, then the operation that knows what each would do.

    two calls rather than one, but the state is prefilled once and cached, so
    the second pays only for its own suffix.
    """
    first = ask(verdict_url,
                target_request(model, state, goal, subgoal, tappable, apps), keep)

    probs = first["tap_target"].get("probabilities") or {}
    target = first["tap_target"].get("choice")
    target_confidence = probs.get(target, 0.0)
    best = next((e for e in tappable if e["index"] == target), None)
    described = ({"element": best["text"] or best["id"], "kind": best["class"]}
                 if best else None)

    package, app_label, app_confidence = None, None, 0.0
    if "launch_target" in first:
        app_probs = first["launch_target"].get("probabilities") or {}
        package = first["launch_target"].get("choice")
        app_confidence = app_probs.get(package, 0.0)
        app_label = next((a["label"] for a in apps if a["package"] == package), package)

    second = ask(verdict_url, operation_request(
        model, state, goal, subgoal, described, target_confidence,
        app_label, app_confidence, dead, acted), 0)
    op_probs = second["operation"].get("probabilities") or {}
    operation = second["operation"]["choice"]
    confidence = op_probs.get(operation, 0.0)

    # DONE is the most consequential answer available: it ends the run and
    # claims the goal is met, and nothing downstream checks it. a run of this
    # benchmark ended in one step on a DONE held at 0.44, which is the same
    # false completion the browser demo produced. it has to clear at least the
    # same bar as a decision to touch something, and when it does not the agent
    # keeps working rather than stopping on a claim it does not believe.
    #
    # `done_confidence` raises that bar for stopping alone; see browser_agent
    # for the measurements and why the default leaves it where it was.
    done_floor = min_confidence if done_confidence is None else done_confidence
    gated = False
    if operation == "DONE" and confidence < done_floor:
        alternatives = {k: v for k, v in op_probs.items() if k != "DONE"}
        if alternatives:
            operation = max(alternatives, key=alternatives.get)
            confidence = alternatives[operation]
            gated = True

    # the scorer reports when it cannot tell the candidates apart; acting anyway
    # is what sent the browser agent into an unrelated page
    chosen_confidence = app_confidence if operation == "LAUNCH" else target_confidence
    if operation in (NEEDS_TARGET | {"LAUNCH"}) and chosen_confidence < min_confidence:
        operation, gated = "BLOCKED", True
    return {"operation": operation, "target": target, "package": package,
            "app_label": app_label, "confidence": confidence,
            "target_confidence": chosen_confidence, "gated": gated,
            # the whole distribution, not just the winner: a DONE at 0.99
            # and a DONE at 0.34 that merely beat five worse options are
            # different problems with different fixes
            "probabilities": {k: round(v, 4) for k, v in op_probs.items()},
            "step_done": first["step_done"]["noul"] > 0.5}


def run(args, mimic):
    goal, subgoal = args.goal, args.goal
    history, progress, plans = [], [], 0
    # the planner must name the FIRST sub-goal too, not only later ones.
    # starting with the whole goal as the current step meant the agent chased
    # whichever half was easiest to reach: measured, it went straight to the
    # Storage row visible on the Settings home screen and never opened About
    # phone, while the ordering sat unread in the goal's prose.
    first_screen = mimic.settled_screen()
    if args.planner_model and first_screen:
        opening = plan(args.planner_url, args.planner_model, goal, [], [],
                       first_screen)
        plans += 1
        if opening:
            subgoal = opening
            print(f"planned first step: {subgoal!r}", file=sys.stderr)
    # operations already shown to do nothing, remembered per screen
    dead = DeadEnds()
    # DONE is withheld until something has been done, and "done"
    # has to mean something CHANGED. minicpm5-2b tapped a search
    # icon that did nothing and was then allowed to claim the goal
    # was met, three runs out of three.
    effective = False
    # targets that lead back to a screen already finished with
    exhausted = Exhausted()
    # ground truth: what the run actually put on screen, not what it claims
    witness = Witnessed(args.require or [])
    # launchable apps change rarely; fetched once and shortlisted per request
    apps = mimic.packages() if args.launch else []
    print(f"apps    {len(apps)} launchable", file=sys.stderr)
    # latency and the screen each step ran on. both were computed and thrown
    # away, which is why a false completion could not be diagnosed from the
    # summary afterwards: the history said what was chosen and never where.
    outcome = {"goal": goal, "status": "running", "steps": 0, "plans": 0,
               "decision_ms": [], "gated": 0}

    for step in range(1, args.max_steps + 1):
        screen = mimic.settled_screen()
        tappable = tappable_now(screen)
        if not tappable:
            outcome["status"] = "no-elements"
            break

        state = build_state(goal, subgoal, screen, history, progress)
        started = time.monotonic()
        witness.observe(screen, step)
        now = signature(screen)
        # the state keeps every row as context; only the OFFER is narrowed
        if args.retire_visited:
            tappable = exhausted.offer(now, tappable)
        d = decide(args.verdict, args.model, state, goal, subgoal, tappable,
                   apps, args.shortlist, args.min_confidence, dead.on(now),
                   acted=effective, done_confidence=args.done_confidence)
        elapsed = (time.monotonic() - started) * 1000
        outcome["decision_ms"].append(round(elapsed))
        outcome["gated"] += int(d["gated"])

        label = d["app_label"] or "" if d["operation"] == "LAUNCH" else ""
        if d["operation"] != "LAUNCH" and d["target"]:
            match = next((e for e in tappable if e["index"] == d["target"]), None)
            label = (match or {}).get("text") or (match or {}).get("id") or "?"
        print(f"  {step:>3} {elapsed:>6.0f}ms  {d['operation']:<12} "
              f"conf={d['confidence']:.2f} target={label[:34]!r} "
              f"tconf={d['target_confidence']:.2f}", flush=True)

        if d["operation"] in ("DONE", "BLOCKED"):
            # the decision that ENDS a run was the only one not recorded,
            # because the loop breaks before the history append below
            outcome["status"] = d["operation"].lower()
            outcome["final_confidence"] = round(d["confidence"], 4)
            outcome["final_probabilities"] = d.get("probabilities")
            break

        if d["step_done"] and args.planner_model:
            nxt = plan(args.planner_url, args.planner_model, goal, progress,
                       history, screen)
            plans += 1
            if nxt:
                progress.append(subgoal)
                subgoal = nxt
                print(f"      planned next step: {subgoal!r}", flush=True)
            else:
                print("      planner returned nothing, keeping current step",
                      flush=True)

        element = next((e for e in tappable if e["index"] == d["target"]), None)
        # going BACK is what makes the screen behind us finished with
        if d["operation"] == "BACK":
            exhausted.leaving(now)
        try:
            mimic.act(d["operation"], element, d["package"])
        except Exception as e:
            # keep WHY, and which verb. the browser agent recorded only that a
            # run had ended here, and a harness fault was indistinguishable
            # from a model that could not decide until the cause was saved.
            print(f"      action failed: {type(e).__name__}: {e}", flush=True)
            outcome["status"] = "action-failed"
            outcome["error"] = f"{type(e).__name__}: {e}"
            outcome["failed_operation"] = d["operation"]
            break
        history.append({"operation": d["operation"], "target": label,
                        "screen": now, "confidence": round(d["confidence"], 4),
                        "target_confidence": round(d["target_confidence"], 4)})
        outcome["steps"] = step
        landed, changed = mimic.screen_after(now)
        effective = effective or changed
        history[-1]["landed"] = signature(landed)
        history[-1]["changed"] = changed
        if d["operation"] == "TAP":
            exhausted.record(now, d["target"], signature(landed))
        if not changed:
            retired = dead.add(now, d["operation"])
            print(f"      screen unchanged; {d['operation']} retired on this "
                  f"screen (dead here: {sorted(retired)})", flush=True)
    else:
        outcome["status"] = "out-of-steps"

    outcome["plans"] = plans
    outcome["history"] = history
    outcome["witness"] = witness.report()
    return outcome


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # host 18473, not 8473: an `adb forward` host port is global to the adb
    # server, and 8473 is the one mimic's own suite forwards to its phone
    ap.add_argument("--mimic", default=os.environ.get("MIMIC_HOST", "127.0.0.1:18473"))
    ap.add_argument("--verdict", default="http://127.0.0.1:8477")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--goal", required=True)
    ap.add_argument("--planner-url", default=None,
                    help="openai-compatible base url for the planner, e.g. .../v1")
    ap.add_argument("--planner-model", default=None,
                    help="omit to run with no planner at all, for comparison")
    ap.add_argument("--retire-visited", action="store_true",
                    help="stop offering a target whose destination is a screen "
                         "already returned from. the android form of the browser "
                         "agent's --retire-read, which took that demo 0/3 to 3/3. "
                         "UNMEASURED HERE: no device was available when it was "
                         "written, so it is opt-in until a run says otherwise")
    ap.add_argument("--launch", action="store_true",
                    help="offer LAUNCH, so an app can be opened by name instead "
                         "of by finding its icon")
    ap.add_argument("--shortlist", type=int, default=26)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--done-confidence", type=float, default=None,
                    help="a separate, usually higher floor for DONE. stopping "
                         "is the one decision nothing downstream checks. "
                         "measured: true completions run 0.862-0.984 and false "
                         "ones 0.522-0.757, and probabilities drift by up to "
                         "0.032, so 0.81 clears both sides. defaults to "
                         "--min-confidence, which is what every measurement in "
                         "docs/EVALS.md was taken under")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--require", action="append", default=None, metavar="TEXT",
                    help="regex that must actually match a row on a screen the "
                         "agent reaches. repeat per requirement. this is the "
                         "score; the agent's own DONE is only its opinion")
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    with open(TOKEN_FILE) as f:
        token = f.read().strip()
    mimic = Mimic(args.mimic, token)

    print(f"goal    {args.goal}", file=sys.stderr)
    print(f"device  {args.mimic}", file=sys.stderr)
    print(f"planner {args.planner_model or 'none'}", file=sys.stderr)
    outcome = run(args, mimic)
    w = outcome["witness"]
    print(f"\nstatus {outcome['status']} after {outcome['steps']} steps, "
          f"{outcome['plans']} planning calls", file=sys.stderr)
    if w["required"]:
        print(f"witnessed {w['witnessed']}/{w['required']} required facts"
              + (f", missing {w['missing']}" if w["missing"] else ""), file=sys.stderr)
        for pattern, value in w["answers"].items():
            print(f"  answer  {pattern!r}: {value!r} (step {w['at_step'][pattern]})",
                  file=sys.stderr)

    if args.summary:
        with open(args.summary, "w") as f:
            json.dump(outcome, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
