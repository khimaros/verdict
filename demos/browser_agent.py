#!/usr/bin/env python3
"""a browser agent we own, so the action space is ours to get right.

browser-use/jev-ultrafast remains the conformance test: it is an unmodified
third-party jev client, and it working against verdict is real evidence the
wire protocol is right. it is a poor capability demo, for two reasons that are
properties of its harness rather than of the model:

- **no way back.** its snapshot emits only scroll_up, scroll_down and wait as
  controls. no back, no history, no navigate. a goal that says "return to the
  front page" asks for an operation that does not exist.
- **the way home scrolls away.** its snapshot drops any element whose centre
  is outside the viewport, so reading comments scrolls the site logo off the
  page and deletes the only remaining route back.

measured over ten runs across two configurations: never completed, 0/5 with no
variance at all. that is a wall, not a tuning problem.

this harness fixes both by construction and applies what the android agent
learned the hard way -- target-first ordering, retiring operations that change
nothing, withholding DONE until something has been done, framing the objective
as outstanding, and scoring on what actually reached the screen. see
docs/DEMOS.md.

usage:
  ./demos/browser_agent.py --url https://news.ycombinator.com/ \
      --goal "Read the comments on the top story" --require "comments"
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import websocket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimic_agent import (DeadEnds, Witnessed, as_target, ask, gate_stopping, post,
                         signature)
from shortlist import restore_answers, shortlist, shortlist_request, shortlist_state

# every one of these is a real thing this driver can do, so an operation the
# model picks can always be carried out. BACK and HOME are the two the
# jev-ultrafast action space lacks, and their absence is what made its task
# unreachable.
OPERATIONS = {
    "CLICK": "Click an element on the page.",
    "SCROLL_DOWN": "Scroll down to reveal what is below.",
    "SCROLL_UP": "Scroll up to reveal what is above.",
    "BACK": "Go back to the previous page. Always available, never scrolls away.",
    "HOME": "Return to the page this run started from. Always available.",
    "DONE": "The goal is visibly satisfied on this page.",
    "BLOCKED": "No available operation can make progress.",
}

NEEDS_TARGET = {"CLICK"}

# how a move is worded when it competes with the links in one question: each
# has to say when it is the answer, since no second question does
MOVES = {
    "SCROLL_DOWN": "Scroll down: what is needed is not among the links offered "
                   "and the page continues below.",
    "SCROLL_UP": "Scroll up: what is needed is above this part of the page.",
    "BACK": "Go back to the previous page: this one has nothing more the goal needs.",
}

# what CDP says when the page is navigating out from under the call
NAVIGATING = "Inspected target navigated or closed"

# a match spanning a line break is normal in page text -- a hacker news byline
# and the comment under it are two lines -- so collected items are squeezed to
# one line each rather than printed with the page's own wrapping
WHITESPACE = re.compile(r"\s+")

# the whole page, not the viewport. an element scrolled out of view is still on
# the page and still reachable; dropping it is what deleted the route home.
SNAPSHOT_JS = r"""
(() => {
  const sel = 'a[href],button,input,textarea,select,summary,[role=button],[onclick]';
  const out = [];
  for (const e of document.querySelectorAll(sel)) {
    const r = e.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) continue;
    const label = (e.innerText || e.value || e.getAttribute('aria-label') ||
                   e.getAttribute('title') || '').trim().replace(/\s+/g, ' ');
    if (!label) continue;
    // the row the link sits in, and the row before it. "238 comments" is not a
    // choice a model can make -- a front page offers twenty of them and they
    // differ only in a number. hacker news puts the title in one <tr> and the
    // comments link in the next, so the previous sibling is where the meaning
    // is. this is the browser form of labelling android rows from contained
    // text, which is what let that demo finish its task.
    const row = e.closest('tr, li, article');
    const prev = row && row.previousElementSibling;
    const context = ((prev && prev.innerText ? prev.innerText + ' | ' : '') +
                     (row ? row.innerText : '')).trim().replace(/\s+/g, ' ');
    out.push({
      label: label.slice(0, 120),
      context: context.slice(0, 160),
      tag: e.tagName.toLowerCase(),
      href: (e.getAttribute('href') || '').slice(0, 200),
      x: Math.round(r.left + r.width / 2 + scrollX),
      y: Math.round(r.top + r.height / 2 + scrollY),
      onscreen: r.top < innerHeight && r.bottom > 0,
    });
  }
  return JSON.stringify({
    url: location.href,
    title: document.title,
    text: document.body.innerText.slice(0, 8000),
    elements: out.slice(0, 400),
  });
})()
"""


class Chrome:
    """the bit of CDP this needs, over one websocket."""

    def __init__(self, cdp_url, start_url):
        # /json/new is PUT-only on current chrome; a GET gets 405
        request = urllib.request.Request(f"{cdp_url}/json/new?about:blank",
                                         method="PUT")
        with urllib.request.urlopen(request, timeout=20) as r:
            target = json.loads(r.read())
        # chrome rejects a devtools websocket that carries an Origin header
        # unless it was launched with --remote-allow-origins. suppressing the
        # header is the client-side half of that and does not loosen chrome.
        self.ws = websocket.create_connection(target["webSocketDebuggerUrl"],
                                              timeout=60, suppress_origin=True)
        self.target_id = target["id"]
        self.cdp_url = cdp_url
        self.start_url = start_url
        self.next_id = 0
        self.send("Page.enable")
        self.send("Runtime.enable")

    def send(self, method, **params):
        self.next_id += 1
        want = self.next_id
        self.ws.send(json.dumps({"id": want, "method": method, "params": params}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") == want:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def evaluate(self, expression):
        out = self.send("Runtime.evaluate", expression=expression, returnByValue=True)
        return out.get("result", {}).get("value")

    def navigate(self, url):
        self.send("Page.navigate", url=url)
        self.settle(expect=url)

    def settle(self, tries=20, pause=0.3, expect=None):
        """wait for the document to be usable, not merely for a call to return.

        `readyState` describes whatever is loaded RIGHT NOW, so on a fresh tab
        it reports about:blank as 'complete' and a navigate that trusts it
        returns before the target page exists. that does not crash: the agent
        finds a handful of elements, decides against them at low confidence and
        blocks on step one, which reads as a model failure. measured -- 0/3
        runs, 0 actions, 4.6 s decisions where a loaded page takes 8.1 s.
        """
        for _ in range(tries):
            try:
                ready, href = self.evaluate(
                    "[document.readyState, location.href]") or [None, None]
            except RuntimeError as e:
                # `history.back()` destroys the execution context, so the very
                # settle that follows a navigation asks a target that is
                # navigating. this is the condition settle exists to wait for,
                # not a fault: raising here killed 17 of 33 runs in a model
                # sweep and recorded them as the agent failing to act.
                if NAVIGATING not in str(e):
                    raise
                time.sleep(pause)
                continue
            if ready in ("interactive", "complete") and not (
                    expect and href in ("about:blank", "", None)):
                return
            time.sleep(pause)

    def snapshot(self):
        raw = self.evaluate(SNAPSHOT_JS)
        return json.loads(raw) if raw else {"elements": [], "text": "", "url": "",
                                            "title": ""}

    def screen(self):
        page = self.snapshot()
        elements = []
        for element in page["elements"]:
            elements.append({"index": str(len(elements) + 1), "x": element["x"],
                             "y": element["y"], "class": element["tag"],
                             "text": element["label"], "id": "",
                             "href": element["href"],
                             "context": element.get("context", ""),
                             "onscreen": element["onscreen"]})
        return elements, page

    def screen_after(self, previous, tries=10, pause=0.4):
        """the page once the action has landed, not merely once a call returned.

        the android agent lost six runs to reading immediately after a tap and
        seeing the previous screen. the same applies here and more so, because
        a click can start a navigation that takes longer than a repaint.
        """
        for attempt in range(tries):
            elements, page = self.screen()
            if signature(elements) != previous:
                return elements, page, True
            if attempt < tries - 1:
                time.sleep(pause)
        return elements, page, False

    def act(self, operation, element):
        if operation == "CLICK":
            # scroll it into view first: a click is delivered at a viewport
            # coordinate, and an element below the fold is not where it sits
            # in page coordinates
            self.evaluate(
                f"window.scrollTo(0, Math.max(0, {element['y']} - innerHeight/2))")
            time.sleep(0.2)
            viewport_y = self.evaluate(f"{element['y']} - scrollY")
            for kind in ("mousePressed", "mouseReleased"):
                self.send("Input.dispatchMouseEvent", type=kind, button="left",
                          clickCount=1, x=element["x"], y=int(viewport_y))
            self.settle()
            return
        if operation in ("SCROLL_DOWN", "SCROLL_UP"):
            sign = 1 if operation == "SCROLL_DOWN" else -1
            self.evaluate(f"window.scrollBy(0, {sign} * innerHeight * 0.8)")
            return
        if operation == "BACK":
            # the call races the navigation it starts, and CDP reports that as
            # an error even though the navigation happened. it cannot be
            # retried -- going back twice is a different page -- so it is
            # swallowed here and settle() waits for where we landed.
            try:
                self.evaluate("history.back()")
            except RuntimeError as e:
                if NAVIGATING not in str(e):
                    raise
            self.settle()
            return
        if operation == "HOME":
            self.navigate(self.start_url)
            return
        raise ValueError(f"no browser verb for {operation}")

    def close(self):
        # two steps, two try blocks. sharing one meant a websocket that refused
        # to close skipped the tab close entirely and said nothing: a 33-run
        # sweep left 13 about:blank tabs behind.
        try:
            self.ws.close()
        except Exception:
            pass
        try:
            urllib.request.urlopen(
                f"{self.cdp_url}/json/close/{self.target_id}", timeout=10)
        except Exception as e:
            print(f"warning: leaked target {self.target_id}: {e}", file=sys.stderr)


def build_state(goal, subgoal, page, elements, history, progress):
    return {
        "goal": goal,
        "current_step": as_target(subgoal),
        "progress": progress,
        "page": {"url": page.get("url", ""), "title": page.get("title", "")},
        "visible_text": (page.get("text") or "")[:2500],
        "links": [{"index": e["index"], "text": e["text"],
                   "onscreen": e["onscreen"]} for e in elements],
        "recent_actions": history[-8:],
    }


def target_request(model, state, goal, subgoal, elements):
    return {"model": model, "state": state, "questions": {
        "click_target": {
            "type": "choice",
            "criteria": {e["index"]: {"link": e["text"], "kind": e["class"],
                                      **({"in": e["context"]} if e.get("context")
                                         else {})}
                         for e in elements},
            "instructions": {"goal": goal, "current_step": as_target(subgoal),
                             "rules": ["Choose the link most likely to advance the "
                                       "current step if clicked.",
                                       "Choose only an offered link."]},
        },
        "step_done": {
            "type": "boolean",
            "instructions": {"current_step": as_target(subgoal),
                             "rules": ["Is the current step already satisfied by "
                                       "what is on this page?"]},
        },
    }}


def operation_request(model, state, goal, subgoal, best, confidence, dead, acted,
                      at_start=False):
    described = f"{best['link']} ({best['kind']})" if best else "nothing useful"
    instructions = {
        "goal": goal, "current_step": as_target(subgoal),
        "if_you_click": f"Clicking would open: {described}",
        "click_confidence": round(confidence, 3),
        "rules": ["Choose the one operation that advances the current step.",
                  "If the link named above is the destination, answer CLICK.",
                  "BACK returns to the previous page and is always available.",
                  "Only answer DONE when the goal is visible right now."],
    }
    criteria = {k: v for k, v in OPERATIONS.items() if k not in dead}
    if not acted:
        criteria.pop("DONE", None)
    if at_start:
        # the tab is created at about:blank and then navigated, so the start
        # page has a history entry behind it and BACK genuinely works -- it
        # lands on a blank page with no elements and the run is over. gating on
        # "has acted yet" was not enough: CLICK, BACK, BACK walked front page
        # -> item -> front page -> about:blank. on the start page there is
        # nowhere useful behind us, whatever we did to get here.
        criteria.pop("BACK", None)
    if dead:
        instructions["already_tried_and_changed_nothing"] = sorted(dead)
    return {"model": model, "state": state,
            "questions": {"operation": dict(type="choice", criteria=criteria,
                                            instructions=instructions)}}


def decide(verdict_url, model, state, goal, subgoal, elements, keep, min_confidence,
           dead, acted, trim_state=False, at_start=False, done_confidence=None):
    """target first, then the operation that knows what clicking would do."""
    request = target_request(model, state, goal, subgoal, elements)
    if keep:
        request, _, dropped = shortlist_request(request, keep)
        # and optionally drop the same rows from the state. measured: a hacker
        # news front page put 190 rows in a 23,042 character prompt where 26
        # were choosable, and the state is prefilled on every one of the three
        # scoring passes in a step. trimming cuts the prompt 69% and the call
        # 68% -- and costs context the model was using, so it is a flag and not
        # a default until the reliability a/b says otherwise.
        if trim_state:
            request, _ = shortlist_state(request)
            state = request["state"]
        first = restore_answers({"answers": ask(verdict_url, request, 0)},
                                dropped)["answers"]
    else:
        first = ask(verdict_url, request, 0)

    probs = first["click_target"].get("probabilities") or {}
    target = first["click_target"].get("choice")
    confidence = probs.get(target, 0.0)
    match = next((e for e in elements if e["index"] == target), None)
    described = {"link": match["text"], "kind": match["class"]} if match else None

    second = ask(verdict_url, operation_request(
        model, state, goal, subgoal, described, confidence, dead, acted,
        at_start), 0)
    op_probs = second["operation"].get("probabilities") or {}
    operation = second["operation"]["choice"]
    op_confidence = op_probs.get(operation, 0.0)

    # DONE ends the run and claims the goal is met, and nothing downstream
    # checks it -- so it has to clear at least the same bar as a decision to
    # touch something. observed at 0.32 on a three-story goal with two stories
    # read: the model was not claiming completion, it was merely least unsure.
    #
    # `done_confidence` raises that bar for stopping alone. the two decisions
    # separate in the measurements: true completions at 0.862-0.984 against
    # false ones at 0.522-0.757, across three models and two tasks. 0.81
    # divides them with more than the 0.032 drift on each side; 0.85 also
    # divides them and does NOT clear the drift, which is how it was corrected.
    # the default stays where it was, because a dozen observations set a knob
    # rather than a default.
    #
    # BLOCKED is held to the same floor, and a DONE reached with operations
    # withheld is asked again with none: both measured on the android agent.
    done_floor = min_confidence if done_confidence is None else done_confidence
    operation, op_confidence, gated = gate_stopping(operation, op_probs, done_floor)
    if operation == "DONE" and (dead or at_start):
        full = ask(verdict_url, operation_request(
            model, state, goal, subgoal, described, confidence, frozenset(), acted),
            0)["operation"]
        confirmed = (full.get("probabilities") or {}).get("DONE", 0.0)
        moves = {k: v for k, v in op_probs.items() if k != "DONE"}
        op_probs = {**op_probs, "DONE": confirmed}
        if moves and (full["choice"] != "DONE" or confirmed < done_floor):
            operation = max(moves, key=moves.get)
            op_confidence, gated = moves[operation], True
        else:
            op_confidence = confirmed

    if operation in NEEDS_TARGET and confidence < min_confidence:
        operation, gated = "BLOCKED", True
    return {"operation": operation, "target": target, "match": match,
            "confidence": op_confidence, "target_confidence": confidence,
            "gated": gated, "probabilities": {k: round(v, 4) for k, v in op_probs.items()},
            "step_done": first["step_done"]["noul"] > 0.5}


def together_request(model, state, goal, subgoal, elements, dead, acted, at_start):
    """one question whose options are every offered link and every move, and
    beside it whether to stop: DONE against the operations, read for DONE."""
    live = {k: v for k, v in OPERATIONS.items() if k not in dead and k != "DONE"}
    if at_start:
        live.pop("BACK", None)
    criteria = {}
    if "CLICK" in live:
        criteria.update({f"CLICK {e['index']}": {
            "click_the_link": e["text"], "kind": e["class"],
            **({"in": e["context"]} if e.get("context") else {})} for e in elements})
    criteria.update({op: MOVES.get(op, text) for op, text in live.items() if op != "CLICK"})
    instructions = {"goal": goal, "current_step": as_target(subgoal)}
    stop = {"stop": {"type": "choice", "criteria": OPERATIONS, "instructions": {
        **instructions, "rules": ["Choose the one operation that advances the current step.",
                                  "Only answer DONE when the goal is visible right now."]}}}
    return {"model": model, "state": state, "questions": {"next": {
        "type": "choice", "criteria": criteria, "instructions": {
            **instructions, "rules": ["Choose the one action that advances the current "
                                      "step."]}}, **(stop if acted else {})}}


def decide_together(verdict_url, model, state, goal, subgoal, elements, keep,
                    min_confidence, dead, acted, at_start=False, done_confidence=None):
    """one call: links and moves compete in a single question, and whether to
    stop is asked beside it. see mimic_agent.decide_together for why."""
    if keep:
        kept, _ = shortlist({e["index"]: {"link": e["text"]} for e in elements}, goal, keep)
        elements = [e for e in elements if e["index"] in kept]
    answers = ask(verdict_url, together_request(
        model, state, goal, subgoal, elements, dead, acted, at_start), 0)
    probabilities = dict(answers["next"].get("probabilities") or {})
    floor = min_confidence if done_confidence is None else done_confidence
    choice, confidence, gated = gate_stopping(answers["next"]["choice"], probabilities, floor)
    stop = (answers.get("stop") or {}) if acted else {}
    if stop:
        probabilities["DONE"] = (stop.get("probabilities") or {}).get("DONE", 0.0)
        if stop.get("choice") == "DONE" and probabilities["DONE"] >= floor:
            choice, confidence = "DONE", probabilities["DONE"]
    operation, _, target = choice.partition(" ")
    if operation in NEEDS_TARGET and confidence < min_confidence:
        operation, gated = "BLOCKED", True
    return {"operation": operation, "target": target or None,
            "match": next((e for e in elements if e["index"] == target), None),
            "confidence": confidence, "target_confidence": confidence, "gated": gated,
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "step_done": False}


def untried(elements, page_url, visited):
    """drop targets whose page has already been read from.

    the front page offers a story's comments as "229 comments", and after
    reading it and coming back it offers exactly the same link again. "which
    story have I already opened" is a question the model answers badly from
    history, and it does not have to answer it at all: the harness knows which
    urls it collected from. this is the lever that works -- it cannot choose
    what it is not offered -- and it is the same one as retiring an operation
    that changed nothing.
    """
    kept = []
    here = (page_url or "").split("#")[0]
    for element in elements:
        href = element.get("href") or ""
        if not href:
            kept.append(dict(element, index=str(len(kept) + 1)))
            continue
        target = urllib.parse.urljoin(page_url, href).split("#")[0]
        # a link to the page we are already on is a no-op. measured:
        # granite-4.2-3b clicked the nav link 'show' eleven times while
        # already on /show, which DeadEnds missed and the ledger could not
        # see, since it is neither collected-from nor came-back-from.
        if target == here or target in visited:
            continue
        kept.append(dict(element, index=str(len(kept) + 1)))
    return kept


def may_stop(acted, harvested, pages):
    """whether DONE is an answer the agent may give yet: something has been
    done with effect, and the task has the pages it said it needs. a claim the
    harness can already see is false is not offered, as on the first step."""
    return acted and sum(1 for items in harvested.values() if items) >= pages


def collects_here(url, pattern):
    """whether this page is one the task wants harvested.

    without this the score counts comment-shaped TEXT rather than the pages
    the goal names: qwen3.5-2b took the site-wide `comments` nav link, landed
    on /newcomments, and harvested five bylines that satisfied the pattern.
    """
    return not pattern or re.search(pattern, url or "") is not None


def collect(text, pattern, count):
    """the first `count` matches of `pattern` in the page text.

    this is the task's actual output and it needs no model. "read the first
    five comments" is a navigation goal plus a read: the agent decides which
    page to open, and the comments are then plain text sitting in that page.
    a model asked to repeat them back would be slower, dearer and able to get
    them wrong, which none of reading them is.
    """
    if not pattern:
        return []
    # MULTILINE, because a pattern anchored with ^ is the natural way to name
    # a line in page text and without it ^ only matches position zero
    rx = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
    return [WHITESPACE.sub(" ", m.group(0)).strip()
            for m in rx.finditer(text or "")][:count]


def run(args, chrome):
    goal = subgoal = args.goal
    history, progress = [], []
    dead, acted = DeadEnds(), False
    witness = Witnessed(args.require or [])
    # keyed by url: "five comments from each of three stories" is three pages
    # with five matches each, and flattening them loses which story they came
    # from -- which is most of the answer
    harvested = {}
    # urls we navigated away from with BACK. going somewhere and coming
    # back is finishing with it, whether or not it yielded anything.
    left = set()
    outcome = {"goal": goal, "status": "running", "steps": 0,
               "decision_ms": [], "gated": 0, "probabilities": []}

    chrome.navigate(args.url)
    for step in range(1, args.max_steps + 1):
        elements, page = chrome.screen()
        if not elements:
            outcome["status"] = "no-elements"
            break
        # the page text is where an answer lives; the links are where a
        # decision lives. witness reads the first, the model decides the second.
        witness.observe([{"text": line, "id": ""}
                         for line in (page.get("text") or "").splitlines()], step)
        url = page.get("url", "")
        found = (collect(page.get("text"), args.collect, args.collect_count)
                 if collects_here(url, args.collect_url) else [])
        if len(found) > len(harvested.get(url, [])):
            # progress the model is told about is what was actually read off a
            # page, never what a step claimed. a goal covering several pages
            # needs it: without a true count of what is already gathered, DONE
            # looks correct on the first one.
            if url not in harvested:
                progress.append(f"read {len(found)} from "
                                f"{page.get('title') or url!r}")
            harvested[url] = found

        if args.retire_read:
            # finished with means collected from OR came back from. keyed on
            # collected alone, a page that yields nothing is never retired:
            # measured, CLICK 'RohanAdwankar' / BACK seven times in one run,
            # because a user page has no comments to collect and every action
            # really did change the page so DeadEnds saw nothing either.
            offered = untried(elements, url, set(harvested) | left)
            if offered:
                elements = offered

        now = signature(elements)
        state = build_state(goal, subgoal, page, elements, history, progress)
        started = time.monotonic()
        # BACK is only useful when there is somewhere useful behind us
        at_start = url.rstrip('/') == args.url.rstrip('/')
        stoppable = may_stop(acted, harvested, args.collect_pages)
        if args.together:
            d = decide_together(args.verdict, args.model, state, goal, subgoal, elements,
                                args.shortlist, args.min_confidence, dead.on(now), stoppable,
                                at_start=at_start, done_confidence=args.done_confidence)
        else:
            d = decide(args.verdict, args.model, state, goal, subgoal, elements,
                       args.shortlist, args.min_confidence, dead.on(now), stoppable,
                       args.trim_state, at_start=at_start,
                       done_confidence=args.done_confidence)
        elapsed = (time.monotonic() - started) * 1000
        outcome["decision_ms"].append(round(elapsed))
        outcome["gated"] += int(d["gated"])
        # every decision's whole distribution, for the reader and not the model
        outcome["probabilities"].append(d["probabilities"])

        label = (d["match"] or {}).get("text", "")
        print(f"  {step:>3} {elapsed:>6.0f}ms  {d['operation']:<12} "
              f"conf={d['confidence']:.2f} target={label[:34]!r} "
              f"tconf={d['target_confidence']:.2f}", flush=True)

        if d["operation"] in ("DONE", "BLOCKED"):
            outcome["status"] = d["operation"].lower()
            outcome["final_confidence"] = round(d["confidence"], 4)
            break
        try:
            chrome.act(d["operation"], d["match"])
        except Exception as e:
            # keep WHY. 17 of 33 runs in a model sweep ended here and the
            # summary recorded only that they had, which made a harness fault
            # indistinguishable from a model that could not decide.
            print(f"      action failed: {type(e).__name__}: {e}", flush=True)
            outcome["status"] = "action-failed"
            outcome["error"] = f"{type(e).__name__}: {e}"
            outcome["failed_operation"] = d["operation"]
            break
        if d["operation"] == "BACK":
            left.add(url)
        history.append({"operation": d["operation"], "target": label,
                        "from": url})
        outcome["steps"] = step

        _, page_after, changed = chrome.screen_after(now)
        # acted must mean acted WITH EFFECT, or a no-op unlocks DONE
        acted = acted or changed
        # where the step LANDED, not just what it chose. without it a loop
        # cannot be diagnosed from the summary afterwards.
        history[-1]["to"] = page_after.get("url", "")
        history[-1]["changed"] = changed
        if not changed:
            retired = dead.add(now, d["operation"])
            print(f"      page unchanged; {d['operation']} retired here "
                  f"(dead: {sorted(retired)})", flush=True)
    else:
        outcome["status"] = "out-of-steps"

    outcome["history"] = history
    outcome["witness"] = witness.report()
    outcome["collected"] = harvested
    return outcome


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cdp", default=os.environ.get("BU_CDP_URL", "http://127.0.0.1:9222"))
    ap.add_argument("--verdict", default="http://127.0.0.1:8477")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--url", default="https://news.ycombinator.com/")
    ap.add_argument("--goal", required=True)
    ap.add_argument("--require", action="append", default=None, metavar="REGEX",
                    help="must actually appear in the page text. this is the "
                         "score; the agent's own DONE is only its opinion")
    ap.add_argument("--collect", default=None, metavar="REGEX",
                    help="the task output: every match of REGEX in the page "
                         "text. read, never generated")
    ap.add_argument("--collect-count", type=int, default=5)
    ap.add_argument("--collect-pages", type=int, default=0, metavar="N",
                    help="how many pages the task needs collected from. DONE is "
                         "not offered before the ledger holds that many")
    ap.add_argument("--collect-url", default=None, metavar="REGEX",
                    help="only collect on pages whose url matches. without "
                         "it any page carrying the right-shaped text scores")
    ap.add_argument("--trim-state", action="store_true",
                    help="with --no-together: "
                         "drop rows from the state that no question can choose. "
                         "cuts the prompt 69%% and the call 68%%, and removes "
                         "page context the model was reading")
    ap.add_argument("--retire-read", action=argparse.BooleanOptionalAction, default=True,
                    help="stop offering links to pages already collected from or "
                         "come back from. changes the answer set: a removed link "
                         "cannot be chosen, which is the point. --no-retire-read "
                         "offers every link every time")
    ap.add_argument("--together", action=argparse.BooleanOptionalAction, default=True,
                    help="ask one question whose options are every offered link "
                         "and every move, with whether to stop asked beside it. "
                         "half the calls. --no-together asks which link and then "
                         "which operation, the form every table before "
                         "docs/EVALS.md 5c was taken under")
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
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--summary", default=None)
    return ap.parse_args(argv)


def main():
    args = parse_args()
    print(f"goal   {args.goal}", file=sys.stderr)
    print(f"start  {args.url}", file=sys.stderr)
    chrome = Chrome(args.cdp, args.url)
    try:
        outcome = run(args, chrome)
    finally:
        chrome.close()

    w = outcome["witness"]
    print(f"\nstatus {outcome['status']} after {outcome['steps']} steps",
          file=sys.stderr)
    if w["required"]:
        print(f"witnessed {w['witnessed']}/{w['required']}"
              + (f", missing {w['missing']}" if w["missing"] else ""), file=sys.stderr)
        for pattern, value in w["answers"].items():
            print(f"  answer {pattern!r}: {value[:100]!r} (step {w['at_step'][pattern]})",
                  file=sys.stderr)
    for url, items in outcome.get("collected", {}).items():
        print(f"collected {len(items)} from {url}", file=sys.stderr)
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item[:96]}", file=sys.stderr)
    if args.summary:
        with open(args.summary, "w") as f:
            json.dump(outcome, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
