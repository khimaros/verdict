"""cut an oversized candidate list down to one that fits a single pass.

this is an application concern and it deliberately does not live in the
library. shortlisting changes the answer set: an option that is removed cannot
be chosen, so a bad shortlist produces a confident wrong answer with a healthy
option mass and nothing in the result to reveal it. the caller has to own that
trade, which means the caller has to see it.

jev needs none of this below 255 options. see docs/JEV_API.md.
"""

import json
import re

# words that appear in every goal and so distinguish nothing
STOPWORDS = frozenset("""
a an and are as at be by for from has have in into is it its of on onto or
that the their then there these this to was were what when where which who
will with then each its it's your you top first next back open read go
""".split())

WORD = re.compile(r"[a-z0-9]+")

# the two demos name their row list differently: the android state calls them
# elements, the browser state calls them links. a trimmer that knows only one
# name silently returns the body whole and reports that it did its job.
STATE_ROW_KEYS = ("elements", "links")


def keywords(text):
    """content words from a goal, lowercased."""
    return {w for w in WORD.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS}


def describe(value):
    """a candidate's description as searchable text."""
    if isinstance(value, str):
        return value.lower()
    return json.dumps(value, sort_keys=True).lower()


def rank(criteria, goal_words):
    """(-score, position) per candidate, so ties keep the page's own order.

    page order is meaningful on a list-shaped site: the top stories are the
    first elements. preserving it on ties is the difference between keeping the
    top 5 stories and keeping 5 arbitrary ones.
    """
    ranked = []
    for position, (option_id, value) in enumerate(criteria.items()):
        text = describe(value)
        hits = sum(1 for w in goal_words if w in text)
        ranked.append((-hits, position, option_id))
    ranked.sort()
    return ranked


def shortlist(criteria, goal, keep):
    """the top `keep` candidates by overlap with the goal.

    returns (kept criteria, report). the report is the point: it names how many
    candidates were removed and how many of the kept ones actually matched the
    goal at all. a shortlist where nothing matched is page order in disguise.
    """
    if len(criteria) <= keep:
        return criteria, None

    goal_words = keywords(goal if isinstance(goal, str) else json.dumps(goal))
    ranked = rank(criteria, goal_words)
    chosen = ranked[:keep]
    matched = sum(1 for score, _, _ in chosen if score < 0)

    kept = {option_id: criteria[option_id] for _, _, option_id in chosen}
    # restore declaration order among the survivors, so the prompt still reads
    # in page order rather than in relevance order
    kept = {k: criteria[k] for k in criteria if k in kept}

    return kept, {"from": len(criteria), "to": len(kept),
                  "matched_goal": matched, "keywords": sorted(goal_words)}


def shortlist_request(body, keep):
    """apply the shortlist to every oversized question in a jev request.

    also returns the ids that were removed, per question, because they have to
    be put back before the client sees the answer. see restore_answers.
    """
    goal_text = ""
    for question in body.get("questions", {}).values():
        instructions = question.get("instructions")
        if isinstance(instructions, dict) and "goal" in instructions:
            goal_text = instructions["goal"]
            break

    reports, dropped = {}, {}
    for name, question in body.get("questions", {}).items():
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            kept, report = shortlist(criteria, goal_text, keep)
            if report:
                dropped[name] = [k for k in criteria if k not in kept]
                question["criteria"] = kept
                reports[name] = report
    return body, reports, dropped


def shortlist_state(body):
    """drop elements from the state that no question can choose any more.

    the state is the cached prefix and by far the largest thing sent. a hacker
    news front page carries 161 elements, about 10.6k tokens, and after
    shortlisting only 26 of them can be chosen. the rest are prefill nobody can
    act on.

    this is a real trade, not free: the discarded rows are still page context,
    and a model that cannot see a row cannot reason about what surrounds the
    row it picks.
    """
    state = body.get("state", {})
    for key in STATE_ROW_KEYS:
        elements = state.get(key)
        if isinstance(elements, list):
            break
    else:
        return body, None

    choosable = set()
    for question in body.get("questions", {}).values():
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            choosable |= set(criteria)

    kept = [e for e in elements if str(e.get("index")) in choosable]
    if len(kept) == len(elements):
        return body, None
    state[key] = kept
    return body, {"from": len(elements), "to": len(kept)}


def restore_answers(result, dropped):
    """put the removed candidates back into the response at probability zero.

    a jev client validates that the returned distribution covers exactly the
    option set it declared: browser-use checks `set(probabilities) == set(ids)`
    and rejects the whole response otherwise. so shortlisting cannot be done
    transparently behind the api, and a shortlisting proxy has to restore the
    shape it changed.

    zero here means "not considered", which is not the same as "considered and
    found unlikely". the two are indistinguishable in the response, which is
    the strongest argument for shortlisting being the caller's decision rather
    than something a server does quietly.
    """
    for name, ids in dropped.items():
        answer = result.get("answers", {}).get(name)
        if not answer or "probabilities" not in answer:
            continue
        for option_id in ids:
            answer["probabilities"].setdefault(option_id, 0.0)
    return result
