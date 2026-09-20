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


def test_a_demo_that_collects_nothing_scores_zero(monkeypatch):
    """the witness fallback exists for the android demo, which has no ledger.
    a browser run that collected NOTHING was scoring 1 because the witness had
    seen its pattern on the front page -- three dead runs reported as 1/3."""
    import reliability

    row = reliability.normalise({"status": "no-elements", "steps": 1,
                                 "collected": {},
                                 "witness": {"witnessed": 1, "required": 1}})
    assert row["stories_visited"] == 0


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
