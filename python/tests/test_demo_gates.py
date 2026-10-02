"""the gates both demo agents put in front of an answer that ends a run.

these are pure decision logic over a canned scorer, so they run without a
model. the behaviours here were each paid for by a wrong run, and both agents
have to keep them -- a fix applied to one of them and not the other is how the
browser agent stopped two stories into a three-story goal on a DONE it held at
0.32 while the android agent already refused exactly that.
"""

import contextlib
import io
import os
import sys

import pytest

DEMOS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "demos")
sys.path.insert(0, DEMOS)

import browser_agent  # noqa: E402
import mimic_agent  # noqa: E402
import run_hn_demo  # noqa: E402
import shortlist  # noqa: E402

from llama_verdict import spec  # noqa: E402

ELEMENTS = [{"index": "1", "text": "229 comments", "class": "a", "id": "",
             "x": 10, "y": 20, "onscreen": True}]


def scorer(target_probs, operation_probs, step_done=0.0):
    """stand in for the verdict server: same shape, no weights."""
    def ask(url, request, retries):
        answers = {}
        for name in request["questions"]:
            if name == "click_target":
                answers[name] = {"choice": max(target_probs, key=target_probs.get),
                                 "probabilities": target_probs}
            elif name == "step_done":
                answers[name] = {"noul": step_done}
            else:
                answers[name] = {
                    "choice": max(operation_probs, key=operation_probs.get),
                    "probabilities": operation_probs}
        return answers
    return ask


def browser_decide(monkeypatch, operation_probs, min_confidence, acted=True,
                   done_confidence=None, target_probs=None):
    monkeypatch.setattr(browser_agent, "ask",
                        scorer(target_probs or {"1": 0.99}, operation_probs))
    return browser_agent.decide("http://x", "m", {}, "goal", "goal", ELEMENTS,
                                0, min_confidence, frozenset(), acted,
                                done_confidence=done_confidence)


def test_browser_done_below_floor_is_not_accepted(monkeypatch):
    """a DONE ends the run and claims the goal is met; nothing downstream
    checks it. observed live at 0.32, three stories asked for and two read."""
    d = browser_decide(monkeypatch, {"DONE": 0.32, "BACK": 0.30, "CLICK": 0.20},
                       min_confidence=0.5)
    assert d["operation"] != "DONE"
    assert d["operation"] == "BACK"


def test_browser_done_above_floor_is_accepted(monkeypatch):
    """the gate has to let a confident DONE through, or no run ever ends."""
    d = browser_decide(monkeypatch, {"DONE": 0.88, "BACK": 0.10},
                       min_confidence=0.5)
    assert d["operation"] == "DONE"


def test_the_browser_holds_giving_up_to_the_floor_too(monkeypatch):
    d = browser_decide(monkeypatch, {"BLOCKED": 0.45, "SCROLL_DOWN": 0.3, "BACK": 0.25},
                       min_confidence=0.0, done_confidence=0.81)
    assert (d["operation"], d["gated"]) == ("SCROLL_DOWN", True)
    assert d["probabilities"]["BLOCKED"] == 0.45


def test_the_browser_confirms_done_against_every_operation(monkeypatch):
    """the android finding, on the agent that shares its floor: DONE's share
    inflates when operations are withheld, so one reached that way is asked
    again with nothing withheld."""
    seen = []

    def ask(url, request, keep):
        if "operation" not in request["questions"]:
            return {"click_target": {"choice": "1", "probabilities": {"1": 0.9}},
                    "step_done": {"noul": 0.0}}
        offered = set(request["questions"]["operation"]["criteria"])
        seen.append(offered)
        probs = ({"DONE": 0.5, "CLICK": 0.45, "BACK": 0.05} if "CLICK" in offered
                 else {"DONE": 0.9, "BACK": 0.1})
        return {"operation": {"choice": "DONE", "probabilities": probs}}

    monkeypatch.setattr(browser_agent, "ask", ask)
    d = browser_agent.decide("http://x", "m", {}, "goal", "goal", ELEMENTS, 0, 0.0,
                             frozenset({"CLICK"}), True, done_confidence=0.81)
    assert len(seen) == 2 and "CLICK" in seen[1]
    assert (d["operation"], d["gated"]) == ("BACK", True)


def browser_together(monkeypatch, probabilities, done=0.0, **more):
    asked = []

    def ask(url, request, keep):
        asked.append(request["questions"])
        stop = {"DONE": done, "CLICK": 1.0 - done}
        return {"next": {"choice": max(probabilities, key=probabilities.get),
                         "probabilities": probabilities},
                "stop": {"choice": max(stop, key=stop.get), "probabilities": stop}}

    monkeypatch.setattr(browser_agent, "ask", ask)
    links = [dict(ELEMENTS[0], index=str(i), text=f"{i} comments") for i in range(1, 41)]
    more = {"dead": frozenset(), "acted": True, **more}
    d = browser_agent.decide_together("http://x", "m", {}, "goal comments", "goal", links,
                                      26, 0.0, **more)
    return d, asked


def test_the_browser_can_ask_links_and_moves_as_rivals_in_one_call(monkeypatch):
    """two calls a step, each re-reading a whole page, is 40 to 76 s a decision
    on a model whose requests reuse no prefix. one call halves it, and lets
    "nothing here, go back" compete with the links."""
    d, asked = browser_together(monkeypatch, {"CLICK 3": 0.7, "BACK": 0.3},
                                dead=frozenset({"SCROLL_UP"}), at_start=True)
    assert len(asked) == 1
    offered = set(asked[0]["next"]["criteria"])
    assert {"CLICK 3", "SCROLL_DOWN", "HOME"} <= offered
    assert not offered & {"SCROLL_UP", "BACK", "DONE", "CLICK"}
    # the shortlist keeps 26 links and never drops a move
    assert sum(k.startswith("CLICK ") for k in offered) == 26
    assert (d["operation"], d["target"], d["match"]["text"]) == ("CLICK", "3", "3 comments")


def test_the_browser_stops_on_the_stop_question_and_not_on_a_thinned_done(monkeypatch):
    d, asked = browser_together(monkeypatch, {"CLICK 3": 0.7, "BACK": 0.3}, done=0.9,
                                done_confidence=0.81)
    assert set(asked[0]) == {"next", "stop"}
    assert (d["operation"], d["confidence"]) == ("DONE", 0.9)
    d, asked = browser_together(monkeypatch, {"BACK": 1.0}, done=0.99, acted=False)
    assert set(asked[0]) == {"next"} and d["operation"] == "BACK"


def test_the_browser_cannot_stop_before_the_task_has_its_pages():
    """measured: qwen3.5-9b claimed DONE after two stories of three, six runs
    of six in both question forms, at 0.41 to 0.44, while clef-flash stopped
    correctly after three at 0.39 to 0.41. no floor divides those. the harness
    already counts the pages it collected from, so a task that names how many
    it needs does not offer DONE before it has them."""
    two = {"item?id=1": ["a"], "item?id=2": ["b"]}
    three = {**two, "item?id=3": ["c"]}
    assert not browser_agent.may_stop(True, two, 3)
    assert browser_agent.may_stop(True, three, 3)
    assert not browser_agent.may_stop(False, three, 3)
    # a task that names no count stops as it did before
    assert browser_agent.may_stop(True, {}, 0)
    # and a page that yielded nothing is not one the task has
    assert not browser_agent.may_stop(True, {**two, "user?id=x": []}, 3)


def test_browser_done_withheld_until_something_was_done(monkeypatch):
    """a run that starts on the answer screen is not a run that succeeded."""
    monkeypatch.setattr(browser_agent, "ask", scorer({"1": 0.99}, {"BACK": 1.0}))
    request = browser_agent.operation_request(
        "m", {}, "goal", "goal", None, 0.5, frozenset(), acted=False)
    assert "DONE" not in request["questions"]["operation"]["criteria"]


def test_android_done_below_floor_is_not_accepted(monkeypatch):
    """the same gate on the agent it was first written for."""
    monkeypatch.setattr(mimic_agent, "ask",
                        scorer({"1": 0.99}, {"DONE": 0.44, "BACK": 0.40}))
    d = mimic_agent.decide("http://x", "m", {}, "goal", "goal", ELEMENTS, [],
                           0, 0.5)
    assert d["operation"] != "DONE"


# DONE and "touch this element" are different bets and the evidence says so.
# measured across three models and two tasks: five TRUE completions held at
# 0.899 to 0.973 and five FALSE ones at 0.522 to 0.757. a single threshold that
# admits the true ones at 0.9 would block most ordinary taps, and one loose
# enough to act on taps admits every false DONE.
#
# the obvious objection was that this measured model size rather than truth --
# the first true DONEs all came from qwen3.5-9b and every false one from a
# smaller model. a single-screen goal broke that: minicpm5-2b at 2.5b completes
# it 3/3 and holds DONE at 0.8993, 0.9387 and 0.9543.
#
# still a knob defaulting to off. ten observations name a threshold; they do
# not move a default that every other measurement was taken under.

def test_done_takes_its_own_threshold_when_given(monkeypatch):
    d = browser_decide(monkeypatch, {"DONE": 0.80, "BACK": 0.15},
                       min_confidence=0.5, done_confidence=0.9)
    assert d["operation"] == "BACK", "a 0.80 DONE cleared a 0.9 done threshold"


def test_the_done_threshold_leaves_the_target_gate_alone(monkeypatch):
    """raising the bar for stopping must not raise it for acting, or the agent
    is merely slower to do everything rather than harder to stop."""
    d = browser_decide(monkeypatch, {"CLICK": 0.70, "DONE": 0.20},
                       min_confidence=0.5, done_confidence=0.95,
                       target_probs={"1": 0.60})
    assert d["operation"] == "CLICK"
    assert not d["gated"]


def test_without_a_done_threshold_the_gate_is_unchanged(monkeypatch):
    """the default has to be the behaviour that was measured, since every
    number in docs/EVALS.md was produced without this flag."""
    d = browser_decide(monkeypatch, {"DONE": 0.88, "BACK": 0.10},
                       min_confidence=0.5, done_confidence=None)
    assert d["operation"] == "DONE"


# every DONE confidence measured so far, with whether the goal had actually
# been reached. the threshold below is derived from these two sets and the
# documented drift, so a later observation landing in the gap fails this test
# rather than quietly outliving the claim in docs/EVALS.md section 5a.
TRUE_DONES = (0.8615, 0.8993, 0.934, 0.9387, 0.9543, 0.973, 0.9836)
FALSE_DONES = (0.522, 0.575, 0.642, 0.716, 0.757)

# 0.7568 is the highest false and 0.8615 the lowest true, a gap of 0.105.
# probabilities are reproducible only to about 0.032 under load
# (spec/constants.json observed_max_drift), so a usable threshold has to clear
# BOTH sides by that much: 0.789 to 0.8295. 0.81 sits in the middle of it.
DONE_THRESHOLD = 0.81


def test_the_measured_threshold_separates_every_observation(monkeypatch):
    for p in TRUE_DONES:
        d = browser_decide(monkeypatch, {"DONE": p, "BACK": round(1 - p, 4)},
                           min_confidence=0.5, done_confidence=DONE_THRESHOLD)
        assert d["operation"] == "DONE", f"a true completion at {p} was blocked"
    for p in FALSE_DONES:
        d = browser_decide(monkeypatch, {"DONE": p, "BACK": round(1 - p, 4)},
                           min_confidence=0.5, done_confidence=DONE_THRESHOLD)
        assert d["operation"] != "DONE", f"a false completion at {p} got through"


def test_the_threshold_clears_both_sides_by_more_than_the_drift():
    """separating the observations is not enough when the observations move.

    0.85 was the first choice and separates every case, but it clears the
    lowest true DONE by 0.0115 against a drift of 0.032 -- so the same run on a
    loaded server could have blocked a real completion. that is the failure
    this gate exists to avoid, inverted.
    """
    drift = spec.constants()["numeric_tolerance"]["observed_max_drift"]
    assert DONE_THRESHOLD - max(FALSE_DONES) > drift, "a false DONE is within drift"
    assert min(TRUE_DONES) - DONE_THRESHOLD > drift, "a true DONE is within drift"


def test_android_done_takes_its_own_threshold(monkeypatch):
    monkeypatch.setattr(mimic_agent, "ask",
                        scorer({"1": 0.99}, {"DONE": 0.80, "BACK": 0.15}))
    d = mimic_agent.decide("http://x", "m", {}, "goal", "goal", ELEMENTS, [],
                           0, 0.5, done_confidence=0.9)
    assert d["operation"] == "BACK"


@pytest.mark.parametrize("pattern,expected", [
    (r"^alice [0-9]+ hours ago", ["alice 2 hours ago"]),
    (r"^ALICE [0-9]+ HOURS AGO", ["alice 2 hours ago"]),
])
def test_collect_anchors_per_line(pattern, expected):
    """^ in a page-text pattern means the start of a line. without MULTILINE it
    means position zero, so a pattern written the obvious way silently returns
    nothing -- which reads as "the page had no comments"."""
    text = "Hacker News\nalice 2 hours ago\nsomething they said\n"
    assert browser_agent.collect(text, pattern, 5) == expected


def test_collect_squeezes_a_multiline_match():
    """a byline and its comment are two lines in page text, and the match that
    spans them is one item."""
    text = "bob 1 hour ago | next\n\n\nit depends on the workload\n"
    got = browser_agent.collect(text, r"^bob [^\n]*\n+[^\n]+", 5)
    assert got == ["bob 1 hour ago | next it depends on the workload"]


def test_collect_without_a_pattern_returns_nothing():
    assert browser_agent.collect("anything", None, 5) == []


def linked(index, href):
    return {"index": index, "text": f"link {index}", "class": "a", "id": "",
            "x": 0, "y": 0, "onscreen": True, "href": href}


def test_a_page_already_read_is_not_offered_again():
    """"which story have I already opened" is a question the model answers
    badly from history, and it does not have to: the harness knows. same lever
    as retiring an operation that changed nothing -- it cannot choose what it
    is not offered."""
    elements = [linked("1", "item?id=1"), linked("2", "item?id=2")]
    kept = browser_agent.untried(elements, "https://news.ycombinator.com/",
                                 {"https://news.ycombinator.com/item?id=1"})
    assert [e["href"] for e in kept] == ["item?id=2"]


def test_untried_renumbers_so_labels_stay_contiguous():
    """labels are assigned by position, so a gap in the indices would point a
    chosen label at the wrong element."""
    elements = [linked(str(i), f"item?id={i}") for i in range(1, 5)]
    kept = browser_agent.untried(elements, "https://news.ycombinator.com/",
                                 {"https://news.ycombinator.com/item?id=2"})
    assert [e["index"] for e in kept] == ["1", "2", "3"]
    assert [e["href"] for e in kept] == ["item?id=1", "item?id=3", "item?id=4"]


def test_untried_keeps_elements_with_no_href():
    """buttons and form controls lead nowhere in particular and are never the
    thing being retired."""
    elements = [linked("1", "item?id=1"), dict(linked("2", ""), **{"class": "button"})]
    kept = browser_agent.untried(elements, "https://news.ycombinator.com/",
                                 {"https://news.ycombinator.com/item?id=1"})
    assert [e["class"] for e in kept] == ["button"]


def test_shortlist_state_trims_a_links_key():
    """the two demos name the row list differently -- `elements` on the android
    side, `links` in the browser agent -- and a trimmer that knows only one name
    returns the body untouched and reports success. measured: 22,030 character
    prompts, 190 rows in the state where 26 were choosable."""
    body = {"state": {"links": [{"index": str(i), "text": f"row {i}"}
                                for i in range(1, 51)]},
            "questions": {"click_target": {"criteria": {"1": {}, "2": {}, "3": {}}}}}
    out, report = shortlist.shortlist_state(body)
    assert report == {"from": 50, "to": 3}
    assert [e["index"] for e in out["state"]["links"]] == ["1", "2", "3"]


def test_the_state_carries_only_choosable_rows(monkeypatch):
    """the state is the cached prefix and by far the largest thing sent. a row
    that no question can choose is prefill nobody can act on, and it is paid on
    every one of the three scoring passes in a step."""
    sent = []

    def capture(url, request, retries):
        sent.append(request)
        return scorer({"1": 0.99}, {"CLICK": 0.9, "BACK": 0.1})(url, request, retries)

    monkeypatch.setattr(browser_agent, "ask", capture)
    elements = [dict(linked(str(i), f"item?id={i}"), text=f"{i} comments")
                for i in range(1, 121)]
    state = browser_agent.build_state("goal", "goal", {"url": "u", "title": "t"},
                                      elements, [], [])
    assert len(state["links"]) == 120, "precondition: the state starts full"

    browser_agent.decide("http://x", "m", state, "goal", "goal", elements,
                         26, 0.5, frozenset(), True, trim_state=True)

    for request in sent:
        rows = request["state"]["links"]
        assert len(rows) <= 26, (
            f"{len(rows)} rows in a state where at most 26 are choosable")


def test_a_failed_reset_aborts_the_run():
    """`mimic_reset.sh` exits 1 and warns when it leaves the device in a bad
    state, and the harness ran it with check=False and capture_output=True --
    so the check built to catch bad start states fed a caller that did not
    look. a run would proceed from a dialler popup and be scored as a model
    result."""
    import reliability

    out = reliability.one_run(sys.executable, "/nonexistent/demo.py", [],
                              dict(os.environ), before_each="exit 1")
    assert out["status"] == "reset-failed"
    assert out["stories_visited"] == 0


def test_a_successful_reset_lets_the_run_proceed():
    import reliability

    out = reliability.one_run(sys.executable, "/nonexistent/demo.py", [],
                              dict(os.environ), before_each="true")
    assert out["status"] != "reset-failed"


def test_a_run_can_be_watched_while_it_runs_and_still_reports_why_it_crashed(tmp_path):
    """every step a demo printed was captured and dropped, so a sweep said
    nothing until a run ended and a fault could only be diagnosed afterwards.
    with a log the demo writes straight to it, and a crash is still named."""
    import reliability

    demo = tmp_path / "demo.py"
    demo.write_text("import sys\nprint('  1 TAP', flush=True)\nsys.exit('went wrong')\n")
    log = tmp_path / "steps.log"
    for _ in range(2):
        out = reliability.one_run(sys.executable, str(demo), [], dict(os.environ),
                                  log=str(log))
    assert out["status"] == "crashed" and out["error"] == ["went wrong"]
    assert log.read_text().count("  1 TAP") == 2


def test_a_demo_that_collects_nothing_scores_zero(monkeypatch):
    """the witness fallback exists for the android demo, which has no ledger.
    a browser run that collected NOTHING was scoring 1 because the witness had
    seen its pattern on the front page -- three dead runs reported as 1/3."""
    import reliability

    row = reliability.normalise({"status": "no-elements", "steps": 1,
                                 "collected": {},
                                 "witness": {"witnessed": 1, "required": 1}})
    assert row["stories_visited"] == 0


def test_a_witnessed_run_is_complete_at_what_its_own_witness_required():
    """the android runs were counted against the browser task's three stories,
    so a sweep that witnessed its one fact three runs of three printed
    `task complete: 0/3` and exported `complete: 0`."""
    import reliability

    runs = [reliability.normalise({"status": "done", "steps": 4, "decision_ms": [1],
                                   "witness": {"witnessed": 1, "required": 1}}),
            reliability.normalise({"status": "blocked", "steps": 2, "decision_ms": [1],
                                   "witness": {"witnessed": 0, "required": 1}})]
    for r in runs:
        r["wall_s"] = 1.0
    with contextlib.redirect_stdout(io.StringIO()):
        summary = reliability.report(runs, 3)
    assert (summary["complete"], summary["reached_one"]) == (1, 1)


def test_a_demo_with_no_ledger_still_scores_on_the_witness(monkeypatch):
    import reliability

    row = reliability.normalise({"status": "done", "steps": 4,
                                 "witness": {"witnessed": 1, "required": 1}})
    assert row["stories_visited"] == 1


def test_the_state_is_untrimmed_by_default(monkeypatch):
    """trimming costs page context, so it is opted into rather than assumed."""
    sent = []

    def capture(url, request, retries):
        sent.append(request)
        return scorer({"1": 0.99}, {"CLICK": 0.9, "BACK": 0.1})(url, request, retries)

    monkeypatch.setattr(browser_agent, "ask", capture)
    elements = [dict(linked(str(i), f"item?id={i}"), text=f"{i} comments")
                for i in range(1, 121)]
    state = browser_agent.build_state("goal", "goal", {"url": "u", "title": "t"},
                                      elements, [], [])
    browser_agent.decide("http://x", "m", state, "goal", "goal", elements,
                         26, 0.5, frozenset(), True)
    assert len(sent[0]["state"]["links"]) == 120


def scripted_chrome(frames):
    """a Chrome whose page state follows a script, with no browser attached."""
    chrome = object.__new__(browser_agent.Chrome)
    chrome.asked = []

    def evaluate(expression):
        chrome.asked.append(expression)
        return list(frames[min(len(chrome.asked) - 1, len(frames) - 1)])

    chrome.evaluate = evaluate
    return chrome


def test_settle_waits_for_the_document_it_navigated_to():
    """readyState belongs to whatever is loaded RIGHT NOW. a fresh tab holds
    about:blank, which is already 'complete', so a settle that trusts it
    returns before the target page exists and the agent decides against an
    empty page. it does not crash -- there are a few elements, the confidence
    is low, and the run blocks on step one looking like a model problem."""
    chrome = scripted_chrome([
        ("complete", "about:blank"),
        ("loading", "about:blank"),
        ("complete", "https://news.ycombinator.com/"),
    ])
    chrome.settle(pause=0, expect="https://news.ycombinator.com/")
    assert len(chrome.asked) == 3, (
        f"settled after {len(chrome.asked)} checks; about:blank was still loaded")


def test_settle_tolerates_a_target_that_is_mid_navigation():
    """`history.back()` destroys the execution context, so the settle that
    follows it asks a target that is navigating and CDP answers
    'Inspected target navigated or closed'. settle was raising on exactly the
    condition it exists to wait for: 17 of 33 runs in a model sweep died here,
    recorded as action-failed and indistinguishable from a model that could
    not decide."""
    chrome = object.__new__(browser_agent.Chrome)
    calls = []

    def evaluate(expression):
        calls.append(expression)
        if len(calls) < 3:
            raise RuntimeError("Runtime.evaluate: {'code': -32000, 'message': "
                               "'Inspected target navigated or closed'}")
        return ["complete", "https://news.ycombinator.com/"]

    chrome.evaluate = evaluate
    chrome.settle(pause=0)
    assert len(calls) == 3, "settle gave up while the target was navigating"


def test_back_treats_a_navigating_target_as_success():
    """`history.back()` races the navigation it starts, so the evaluate that
    triggers it can itself come back 'Inspected target navigated or closed'.
    that error means the navigation HAPPENED. it cannot simply be retried --
    calling history.back() twice goes back two pages -- so the operation
    swallows it and settles instead. all 12 remaining action-failed runs in a
    model sweep were this, every one of them on BACK."""
    chrome = object.__new__(browser_agent.Chrome)
    calls = []

    def evaluate(expression):
        calls.append(expression)
        if expression == "history.back()":
            raise RuntimeError("Runtime.evaluate: {'code': -32000, 'message': "
                               "'Inspected target navigated or closed'}")
        return ["complete", "https://news.ycombinator.com/"]

    chrome.evaluate = evaluate
    chrome.act("BACK", None)
    assert calls.count("history.back()") == 1, "went back more than once"
    assert len(calls) > 1, "did not settle after navigating"


def test_settle_still_raises_an_unrelated_cdp_error():
    """tolerating everything would hide a real fault."""
    chrome = object.__new__(browser_agent.Chrome)

    def evaluate(expression):
        raise RuntimeError("Runtime.evaluate: {'code': -32601, "
                           "'message': 'method not found'}")

    chrome.evaluate = evaluate
    with pytest.raises(RuntimeError, match="method not found"):
        chrome.settle(pause=0)


def test_settle_without_a_target_accepts_the_current_document():
    """after a click there is no new url to wait for, only readiness."""
    chrome = scripted_chrome([("complete", "https://news.ycombinator.com/")])
    chrome.settle(pause=0)
    assert len(chrome.asked) == 1


def test_the_tab_is_closed_even_when_the_websocket_close_fails(monkeypatch):
    """the two cleanup steps used to share one try block, so a websocket that
    refused to close skipped the tab close and the exception was swallowed. a
    33-run sweep leaked 13 about:blank tabs that way, and the growing tab count
    is the most likely reason half its runs died in the harness."""
    closed = []

    class RefusingSocket:
        def close(self):
            raise OSError("already closed")

    def fake_urlopen(url, timeout=None):
        closed.append(url)
        return contextlib.nullcontext(io.BytesIO(b""))

    chrome = object.__new__(browser_agent.Chrome)
    chrome.ws = RefusingSocket()
    chrome.cdp_url = "http://127.0.0.1:9222"
    chrome.target_id = "ABC123"
    monkeypatch.setattr(browser_agent.urllib.request, "urlopen", fake_urlopen)

    chrome.close()
    assert closed == ["http://127.0.0.1:9222/json/close/ABC123"], (
        "the tab close was skipped because the websocket close raised first")


def test_back_is_not_offered_on_the_start_page():
    """the tab is created at about:blank and then navigated, so the start page
    has a history entry BEHIND it and BACK genuinely works -- it lands on a
    blank page with no elements and the run is over.

    gating on "has acted yet" is not enough: measured, CLICK then BACK then
    BACK walked front page -> item -> front page -> about:blank. the rule is
    positional, not temporal. on the start page there is nowhere useful behind
    us, whatever we did to get here."""
    request = browser_agent.operation_request(
        "m", {}, "goal", "goal", None, 0.9, frozenset(), acted=True,
        at_start=True)
    assert "BACK" not in request["questions"]["operation"]["criteria"]


def test_back_is_offered_once_we_have_navigated_away():
    request = browser_agent.operation_request(
        "m", {}, "goal", "goal", None, 0.9, frozenset(), acted=True,
        at_start=False)
    assert "BACK" in request["questions"]["operation"]["criteria"]


def test_a_candidate_carries_the_row_it_sits_in():
    """"238 comments" is not a choice a model can make: a front page offers
    twenty of them differing only in a number, the mass spreads, and the gate
    correctly refuses a coin flip. measured: 2 of 3 runs blocked on step one
    with the page fully loaded and the witness satisfied."""
    elements = [dict(linked("1", "item?id=1"), text="238 comments",
                     context="Some story title (example.com) | 238 comments")]
    request = browser_agent.target_request("m", {}, "goal", "goal", elements)
    criterion = request["questions"]["click_target"]["criteria"]["1"]
    assert criterion["link"] == "238 comments"
    assert "Some story title" in criterion["in"]


def test_an_element_off_the_screen_is_not_offered():
    """measured: granite-4.2-3b picked an element with negative bounds and the
    run died on its first action, `mimic TAP: Path bounds must not be
    negative`. android's gesture dispatcher refuses the coordinates, so the
    element was never tappable and should never have been a candidate."""
    rows = [{"index": "1", "text": "offscreen", "class": "TextView", "id": "",
             "x": -12, "y": 400},
            {"index": "2", "text": "real", "class": "TextView", "id": "",
             "x": 540, "y": 400}]
    assert [e["text"] for e in mimic_agent.tappable_now(rows)] == ["real"]


def test_an_element_with_no_coordinates_is_still_offered():
    """a missing coordinate is not a negative one, and LAUNCH targets carry
    none at all."""
    rows = [{"index": "1", "text": "no coords", "class": "TextView", "id": ""}]
    assert len(mimic_agent.tappable_now(rows)) == 1


def tappable(*indices):
    return [{"index": i, "text": f"row {i}", "class": "TextView", "id": ""}
            for i in indices]


def test_a_link_to_the_page_we_are_already_on_is_not_offered():
    """measured: granite-4.2-3b clicked the nav link 'show' eleven times in a
    row while already on /show. a self-link is a no-op that DeadEnds can miss
    (something made it think the page changed) and that the retire ledger
    cannot see, since it is neither collected-from nor came-back-from."""
    here = "https://news.ycombinator.com/show"
    elements = [linked("1", "show"), linked("2", "item?id=9")]
    kept = browser_agent.untried(elements, here, set())
    assert [e["href"] for e in kept] == ["item?id=9"]


def test_collection_is_restricted_to_pages_the_task_wants():
    """qwen3.5-2b took the site-wide `comments` nav link, landed on
    /newcomments, and harvested five bylines that satisfied the pattern. the
    score counted comment-shaped TEXT rather than story pages."""
    assert browser_agent.collects_here(
        "https://news.ycombinator.com/item?id=49771110", r"/item\?id=")
    assert not browser_agent.collects_here(
        "https://news.ycombinator.com/newcomments", r"/item\?id=")


def test_without_a_pattern_every_page_collects():
    assert browser_agent.collects_here("https://example.com/anything", None)


def test_a_page_returned_from_is_not_offered_again():
    """measured on the browser: CLICK 'RohanAdwankar' / BACK, seven times in
    one run. the user page yields no comments, so `--retire-read` never
    retires it -- its ledger is what was COLLECTED -- and DeadEnds cannot see
    it because every action really does change the page.

    so "finished with" is collected-from OR came-back-from, and the android
    agent's Exhausted already said so. this is the same rule."""
    elements = [linked("1", "user?id=x"), linked("2", "item?id=9")]
    kept = browser_agent.untried(
        elements, "https://news.ycombinator.com/",
        {"https://news.ycombinator.com/user?id=x"})
    assert [e["href"] for e in kept] == ["item?id=9"]


def test_a_destination_already_left_is_not_offered_again():
    """the android oscillation DeadEnds cannot see. measured: TAP 'Model' /
    BACK four times until the step budget ran out. every one of those actions
    genuinely changed the screen, so nothing was ever a no-op -- the pair of
    screens simply repeated. this is the android form of `--retire-read`."""
    exhausted = mimic_agent.Exhausted()
    exhausted.record("about-screen", "3", "model-dialog")
    exhausted.leaving("model-dialog")
    offered = exhausted.offer("about-screen", tappable("1", "3", "7"))
    assert [e["index"] for e in offered] == ["1", "7"]


def test_a_destination_visited_but_not_left_is_still_offered():
    """going somewhere is not finishing with it. only a screen the agent came
    back from is exhausted."""
    exhausted = mimic_agent.Exhausted()
    exhausted.record("about-screen", "3", "model-dialog")
    offered = exhausted.offer("about-screen", tappable("1", "3"))
    assert [e["index"] for e in offered] == ["1", "3"]


def test_retiring_every_target_falls_back_to_offering_them_all():
    """an empty candidate list is worse than a stale one: the agent cannot
    answer at all, and BLOCKED on no options says nothing about the screen."""
    exhausted = mimic_agent.Exhausted()
    for index in ("1", "2"):
        exhausted.record("screen", index, f"dest-{index}")
        exhausted.leaving(f"dest-{index}")
    offered = exhausted.offer("screen", tappable("1", "2"))
    assert [e["index"] for e in offered] == ["1", "2"]


def test_back_is_given_a_second_try_before_it_is_retired():
    """measured on the settings search screen: the first BACK closes the
    keyboard and leaves the rows as they were, the second leaves the screen.
    retiring BACK on the first no-op left HOME as the only way out, and jev-omni
    spent twenty steps going HOME and launching back into the same screen."""
    dead = mimic_agent.DeadEnds()
    assert "BACK" not in dead.add("search", "BACK")
    assert "BACK" in dead.add("search", "BACK")
    assert "TAP" in dead.add("search", "TAP")


def android_row(index, text):
    return {"index": index, "x": 10, "y": 10 * int(index), "class": "Row", "text": text,
            "id": "", "borrowed": False}


def test_the_app_already_in_front_is_not_offered_for_launch(monkeypatch):
    """measured: every jev-omni run spent its second step launching Settings
    from inside Settings, and again after each scroll, because a no-op is
    remembered per screen and a scroll makes a new one."""
    asked = []

    def ask(url, request, keep):
        asked.append(request["questions"])
        return {"tap_target": {"choice": "1", "probabilities": {"1": 0.9}},
                "step_done": {"noul": 0.0},
                "launch_target": {"choice": "com.android.settings",
                                  "probabilities": {"com.android.settings": 0.99}},
                "operation": {"choice": "TAP", "probabilities": {"TAP": 0.9}}}

    monkeypatch.setattr(mimic_agent, "ask", ask)
    apps = [{"package": "com.android.settings", "label": "Settings"}]
    for foreground, offered in (("com.android.settings", False), (None, True)):
        mimic_agent.decide("http://x", "m", {}, "goal", "goal", [android_row("1", "About")],
                           apps, 0, 0.0, foreground=foreground)
        assert ("LAUNCH" in asked[-1]["operation"]["criteria"]) is offered


class ScriptedDevice:
    """a device of three screens, one of which cannot be read."""

    SCREENS = {"home": [android_row("1", "Settings")],
               "settings": [android_row("1", "Security"), android_row("2", "About")],
               "security": []}

    def __init__(self):
        self.at, self.acts = "home", []

    def settled_screen(self):
        return self.SCREENS[self.at]

    def screen_after(self, previous):
        screen = self.settled_screen()
        return screen, mimic_agent.signature(screen) != previous

    def packages(self):
        return []

    def act(self, operation, element, package=None):
        self.acts.append(operation)
        tapped = (element or {}).get("text")
        self.at = {("home", "Settings"): "settings", ("settings", "Security"): "security",
                   ("security", None): "settings"}.get((self.at, tapped), self.at)


def test_a_screen_that_cannot_be_read_is_backed_out_of_and_not_offered_again(monkeypatch):
    """measured twice: tapping 'Security & privacy' opens a window mimic reports
    as `no active window` for as long as it is up, and the run ended there as
    `no-elements`. BACK returns to the list, so the agent backs out and the row
    that led there stops being offered."""
    offered = []

    def decide(verdict, model, state, goal, subgoal, tappable, *rest, **more):
        offered.append([e["text"] for e in tappable])
        operation = "TAP" if len(offered) < 3 else "DONE"
        return {"operation": operation, "target": "1", "package": None, "app_label": None,
                "confidence": 0.9, "target_confidence": 0.9, "gated": False,
                "probabilities": {}, "step_done": False}

    monkeypatch.setattr(mimic_agent, "decide", decide)
    device = ScriptedDevice()
    args = mimic_agent.parse_args(["--goal", "find the version", "--no-together"])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        outcome = mimic_agent.run(args, device)
    assert outcome["status"] == "done"
    assert device.acts == ["TAP", "TAP", "BACK"]
    assert offered == [["Settings"], ["Security", "About"], ["About"]]
    assert outcome["history"][-1]["operation"] == "BACK"


def test_a_field_the_agent_cannot_type_into_is_not_a_tap_target():
    """the agent has no operation that enters text, so tapping a text field
    only focuses it. measured: jev-omni tapped the settings search field in
    every run, the tap changed nothing, and the run ended there."""
    nodes = [{"class": "android.widget.ImageButton", "label": "Navigate up",
              "center": [94, 212], "actions": ["click", "focus"]},
             {"class": "android.widget.AutoCompleteTextView", "text": "Search",
              "center": [561, 211], "actions": ["click", "long", "edit", "focus"]}]
    assert [e["text"] for e in mimic_agent.parse_screen(nodes)] == ["Navigate up"]


def test_giving_up_is_held_to_the_same_floor_as_claiming_success(monkeypatch):
    """BLOCKED ends a run exactly as DONE does, and nothing checks it either.
    measured: jev-omni ended three runs of three on BLOCKED at 0.41 to 0.49,
    with SCROLL_DOWN still available and the goal two scrolls away."""
    probs = {"BLOCKED": 0.41, "SCROLL_UP": 0.26, "SCROLL_DOWN": 0.25, "DONE": 0.04}
    monkeypatch.setattr(mimic_agent, "ask", scorer({"1": 0.99}, probs))
    d = mimic_agent.decide("http://x", "m", {}, "goal", "goal", ELEMENTS, [], 0, 0.0,
                           done_confidence=0.81)
    assert (d["operation"], d["gated"]) == ("SCROLL_UP", True)
    monkeypatch.setattr(mimic_agent, "ask", scorer({"1": 0.99}, {"BLOCKED": 0.9, "BACK": 0.1}))
    d = mimic_agent.decide("http://x", "m", {}, "goal", "goal", ELEMENTS, [], 0, 0.0,
                           done_confidence=0.81)
    assert (d["operation"], d["gated"]) == ("BLOCKED", False)


def test_a_tap_already_taken_twice_from_a_screen_is_not_offered_a_third_time():
    """an oscillation whose every action changes the screen. measured:
    clef-flash tapped the same row from the same screen six times in one run,
    and jev-omni went into the search screen and back until it gave up. twice,
    because the second arrival can differ: a screen re-entered showed the row
    the first visit had below the fold."""
    repeated = mimic_agent.Repeated()
    rows = tappable("1", "3")
    repeated.record(ABOUT, "row 3")
    assert [e["index"] for e in repeated.offer(ABOUT, rows)] == ["1", "3"]
    repeated.record(ABOUT, "row 3")
    assert [e["index"] for e in repeated.offer(ABOUT, rows)] == ["1"]
    assert [e["index"] for e in repeated.offer(("some", "other", "screen"), rows)] == ["1", "3"]


ABOUT = ("row 1", "row 3", "Device name", "Model", "Android version", "Build number")


def test_a_screen_is_still_itself_when_one_row_scrolls_into_view():
    """measured: the settings list came back from a sub-screen with one more
    row in view, its signature no longer matched, and a row already tapped
    twice was offered and taken twice more. a tap is remembered by the row's
    name on a screen that is mostly the same rows, not by position on an
    identical one."""
    repeated = mimic_agent.Repeated()
    for _ in range(2):
        repeated.record(ABOUT, "row 3")
    shifted = (*ABOUT, "Uptime")
    # the row's index moved with the list; its name did not
    rows = [{"index": "2", "text": "row 1", "id": ""}, {"index": "4", "text": "row 3", "id": ""}]
    assert [e["text"] for e in repeated.offer(shifted, rows)] == ["row 1"]


def test_when_every_row_has_been_tapped_out_tapping_stops_being_offered():
    """offering the whole list again let jev-omni tap its way round one screen
    for forty steps, its SCROLL_DOWN never above 0.17. with nothing left to
    tap here, the question that remains is which way to move."""
    repeated = mimic_agent.Repeated()
    rows = tappable("1", "3")
    for label in ("row 1", "row 1", "row 3", "row 3"):
        repeated.record(ABOUT, label)
    offered, dead = mimic_agent.narrowed(repeated, frozenset({"SCROLL_UP"}), ABOUT, rows)
    assert offered == rows and dead == {"SCROLL_UP", "TAP"}
    offered, dead = mimic_agent.narrowed(repeated, frozenset(), ("another", "screen"), rows)
    assert offered == rows and dead == frozenset()


def one_question(monkeypatch, probabilities, done=0.0, **more):
    """decide with rows, apps and moves as rivals, against a scripted scorer."""
    asked = []

    def ask(url, request, keep):
        asked.append(request["questions"])
        stop = {"DONE": done, "TAP": 1.0 - done}
        return {"next": {"choice": max(probabilities, key=probabilities.get),
                         "probabilities": probabilities},
                "stop": {"choice": max(stop, key=stop.get), "probabilities": stop}}

    monkeypatch.setattr(mimic_agent, "ask", ask)
    rows = [android_row("1", "Search settings"), android_row("2", "Apps")]
    apps = [{"package": "com.android.settings", "label": "Settings"},
            {"package": "com.android.dialer", "label": "Phone"}]
    d = mimic_agent.decide_together("http://x", "m", {}, "goal", "goal", rows, apps, **more)
    return d, asked


def test_rows_apps_and_moves_are_rivals_in_one_question(monkeypatch):
    """asked which row to tap and THEN which operation, jev-omni named the
    search row at 0.97 and then tapped it at 0.85, forty steps a run, with
    SCROLL_DOWN never above 0.17. asked once, with scrolling an option beside
    the rows, it put 0.988 on scrolling, on the same screen."""
    d, asked = one_question(monkeypatch, {"SCROLL_DOWN": 0.9, "TAP 1": 0.1},
                            dead=frozenset({"SCROLL_UP"}),
                            foreground="com.android.settings")
    assert len(asked) == 1
    offered = set(asked[0]["next"]["criteria"])
    assert {"TAP 1", "TAP 2", "LAUNCH com.android.dialer", "SCROLL_DOWN", "BACK"} <= offered
    assert not offered & {"SCROLL_UP", "LAUNCH com.android.settings", "TAP", "LAUNCH"}
    assert (d["operation"], d["confidence"]) == ("SCROLL_DOWN", 0.9)


def test_a_chosen_row_or_app_comes_back_as_its_operation_and_target(monkeypatch):
    d, _ = one_question(monkeypatch, {"TAP 2": 0.7, "BACK": 0.3})
    assert (d["operation"], d["target"], d["target_confidence"]) == ("TAP", "2", 0.7)
    d, _ = one_question(monkeypatch, {"LAUNCH com.android.dialer": 0.8, "BACK": 0.2})
    assert (d["operation"], d["package"], d["app_label"]) == (
        "LAUNCH", "com.android.dialer", "Phone")


def test_whether_to_stop_is_asked_beside_the_moves_as_done_against_the_operations(
        monkeypatch):
    """measured, twice. as one option among twenty DONE thinned out: true ones
    came in at 0.57 to 0.70 and never cleared the floor, and clef-flash stopped
    in no run having stopped correctly in all three. asked as a bare yes or no
    it went the other way: 0.90 on the settings list, where a row merely
    mentions the answer, three runs of three. DONE against the operations is
    the form whose false ones sit below its true ones, so that is what is
    asked, in the same request."""
    d, asked = one_question(monkeypatch, {"TAP 2": 0.6, "BACK": 0.4}, done=0.9,
                            done_confidence=0.81)
    assert set(asked[0]) == {"next", "stop"}
    assert "DONE" not in asked[0]["next"]["criteria"]
    assert set(asked[0]["stop"]["criteria"]) == set(mimic_agent.OPERATIONS)
    assert (d["operation"], d["confidence"], d["probabilities"]["DONE"]) == ("DONE", 0.9, 0.9)
    d, _ = one_question(monkeypatch, {"TAP 2": 0.6, "BACK": 0.4}, done=0.7,
                        done_confidence=0.81)
    assert (d["operation"], d["target"]) == ("TAP", "2")


def test_stopping_is_not_asked_about_before_anything_has_been_done(monkeypatch):
    d, asked = one_question(monkeypatch, {"BACK": 1.0}, done=0.99, acted=False)
    assert set(asked[0]) == {"next"} and d["operation"] == "BACK"


def test_one_question_holds_giving_up_to_the_floor(monkeypatch):
    d, _ = one_question(monkeypatch, {"BLOCKED": 0.5, "TAP 2": 0.3}, done_confidence=0.81)
    assert (d["operation"], d["target"], d["gated"]) == ("TAP", "2", True)


def test_a_move_already_made_twice_from_a_screen_is_not_offered_a_third_time():
    """measured, both invisible to a limit on taps alone: clef-flash scrolled
    down and up between the same two positions for thirty steps, and jev-omni
    launched one app and then another ten times over."""
    repeated = mimic_agent.Repeated()
    for move in ("SCROLL_DOWN", "SCROLL_DOWN", "LAUNCH", "SCROLL_UP", "BACK", "BACK"):
        repeated.record(ABOUT, move)
    _, dead = mimic_agent.narrowed(repeated, frozenset(), ABOUT, tappable("1"))
    # BACK is the way out of anywhere and stays
    assert dead == {"SCROLL_DOWN"}
    repeated.record(ABOUT, "LAUNCH")
    assert mimic_agent.narrowed(repeated, frozenset(), ABOUT, tappable("1"))[1] == {
        "SCROLL_DOWN", "LAUNCH"}


def test_done_is_confirmed_against_every_operation_when_some_were_withheld(monkeypatch):
    """measured: granite-4.2-3b claimed DONE at 0.90 to 0.97 three runs of
    three with the answer never on screen, each time on a screen where TAP and
    LAUNCH had been withheld. a probability over five options is not the one
    the floor was measured on, so a DONE reached that way is asked again with
    nothing withheld."""
    seen = []

    def ask(url, request, keep):
        questions = request["questions"]
        if "operation" not in questions:
            return {"tap_target": {"choice": "1", "probabilities": {"1": 0.9}},
                    "step_done": {"noul": 0.0}}
        offered = set(questions["operation"]["criteria"])
        seen.append(offered)
        probs = ({"DONE": 0.55, "TAP": 0.4, "BACK": 0.05} if "TAP" in offered
                 else {"DONE": 0.93, "BACK": 0.07})
        return {"operation": {"choice": "DONE", "probabilities": probs}}

    monkeypatch.setattr(mimic_agent, "ask", ask)
    d = mimic_agent.decide("http://x", "m", {}, "goal", "goal", ELEMENTS, [], 0, 0.0,
                           dead=frozenset({"TAP", "LAUNCH"}), done_confidence=0.81)
    assert len(seen) == 2 and "TAP" in seen[1]
    assert (d["operation"], d["gated"], d["probabilities"]["DONE"]) == ("BACK", True, 0.55)
    seen.clear()
    mimic_agent.decide("http://x", "m", {}, "goal", "goal", ELEMENTS, [], 0, 0.0,
                       done_confidence=0.81)
    assert len(seen) == 1


def test_the_state_says_whether_the_screen_continues_below():
    """the one failure every model shared: all of them reach the About screen,
    and when the version row is below the fold none scrolls down. qwen3.5-9b
    tapped 'Device name' or claimed DONE at 0.83; clef-flash went back out.
    the state listed the rows in view and said nothing about there being more,
    so "it is not here" and "it is further down" read the same."""
    nodes = [{"class": "android.widget.ScrollView", "actions": ["scroll"]},
             {"class": "android.widget.TextView", "text": "Device name",
              "center": [1, 2], "actions": ["click"]}]
    assert mimic_agent.scrolls(nodes) and not mimic_agent.scrolls(nodes[1:])
    more = mimic_agent.what_is_below(True, frozenset())
    assert "below" in more and "SCROLL_DOWN" in more
    assert "end" in mimic_agent.what_is_below(True, frozenset({"SCROLL_DOWN"}))
    assert mimic_agent.what_is_below(False, frozenset()) is None
    rows = [android_row("1", "Device name")]
    assert mimic_agent.build_state("g", "g", rows, [], [], more)["below_this_screen"] == more
    assert "below_this_screen" not in mimic_agent.build_state("g", "g", rows, [], [])


def test_an_unreadable_screen_is_left_however_the_agent_got_there():
    """measured: jev-omni launched an app whose window mimic could not read,
    and the run ended `no-elements` because only a TAP was being backed out
    of. BACK first, HOME if that did not help, and only then give up."""
    device, history, repeated = ScriptedDevice(), [], mimic_agent.Repeated()
    assert mimic_agent.back_out(device, history, None, repeated)
    assert mimic_agent.back_out(device, history, None, repeated)
    assert not mimic_agent.back_out(device, history, None, repeated)
    assert device.acts == ["BACK", "HOME"]


def test_both_agents_ask_one_question_unless_told_to_ask_two():
    """measured before it became the default. android: ahead on all three
    models it was run on, 6 runs of 9 against 2, at half the latency. browser:
    half the decision time on the two head models and nothing lost. the
    two-question form stays reachable, since every earlier table was taken
    under it."""
    for agent in (mimic_agent, browser_agent):
        assert agent.parse_args(["--goal", "g"]).together is True
        assert agent.parse_args(["--goal", "g", "--together"]).together is True
        assert agent.parse_args(["--goal", "g", "--no-together"]).together is False


def test_the_browser_stops_offering_pages_it_has_finished_with_by_default():
    """the largest single intervention measured on either demo, 0/3 to 3/3,
    and the flag every sweep since has passed. a sweep that forgot it had
    clef-flash cycling between one story and the front page until the steps
    ran out, and was discarded."""
    assert browser_agent.parse_args(["--goal", "g"]).retire_read is True
    assert browser_agent.parse_args(["--goal", "g", "--retire-read"]).retire_read is True
    assert browser_agent.parse_args(["--goal", "g", "--no-retire-read"]).retire_read is False


def test_the_default_android_loop_runs_on_one_question(monkeypatch):
    asked = []

    def decide_together(verdict, model, state, goal, subgoal, tappable, apps, *rest, **more):
        asked.append([e["text"] for e in tappable])
        return {"operation": "TAP" if len(asked) < 2 else "DONE", "target": "1",
                "package": None, "app_label": None, "confidence": 0.9,
                "target_confidence": 0.9, "gated": False, "probabilities": {},
                "step_done": False}

    monkeypatch.setattr(mimic_agent, "decide_together", decide_together)
    monkeypatch.setattr(mimic_agent, "decide", None)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        outcome = mimic_agent.run(mimic_agent.parse_args(["--goal", "g"]), ScriptedDevice())
    assert outcome["status"] == "done" and asked == [["Settings"], ["Security", "About"]]


def test_stopping_is_held_to_the_measured_threshold_unless_told_otherwise():
    """every false DONE recorded on the device sits at 0.757 or below and every
    true one at 0.850 or above, across six models: clef-flash navigated to the
    right screen three runs of three and claimed DONE at 0.55 and 0.59 with the
    answer below the fold. so the gate is the default and 0 turns it off."""
    assert mimic_agent.parse_args(["--goal", "g"]).done_confidence == 0.81
    assert mimic_agent.parse_args(["--goal", "g", "--done-confidence", "0"]
                                  ).done_confidence == 0


def test_target_first_flag_reaches_the_proxy(monkeypatch):
    """a flag that is parsed, documented and then dropped at the call site is
    worse than a missing one: the run looks like evidence that the technique
    does not help, and target-first is the single largest measured win there
    is. `--target-first` was exactly that."""
    class Stop(Exception):
        pass

    seen = {}

    def recorder(*a, **kw):
        seen["positional"], seen["keyword"] = a, kw
        raise Stop()

    monkeypatch.setattr(run_hn_demo, "redirect_to_verdict", recorder)
    monkeypatch.setattr(sys, "argv", ["run_hn_demo.py", "--target-first"])
    with pytest.raises(Stop):
        run_hn_demo.main()
    assert seen["keyword"].get("target_first") is True
