#!/usr/bin/env python3
"""generate the conformance fixtures from the reference implementation.

fixtures render against `spec/formatters/reference.json`, a synthetic affix
table, so that they are model-independent and readable. a real model's
formatter is covered separately by the pinned artifacts in spec/formatters/.

these are golden files: generated once, reviewed by a human, then frozen. they
cannot catch a bug that was present at generation time, only drift away from
the reviewed output, which is what a second implementation needs.

usage:
  ./scripts/build_fixtures.py            # write spec/fixtures/
  ./scripts/build_fixtures.py --check    # fail if regenerating would differ
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from llama_verdict import prompt, spec, types

SPEC = os.path.join(HERE, "..", "spec")
OUT = os.path.join(SPEC, "fixtures")

TICKET = "My invoice charged me twice for the same month and I want a refund."

ROUTING = {"access": "Account access: logins, passwords, permissions.",
           "billing": "Billing: invoices, payments, refunds.",
           "technical": "Technical: bugs, outages, integrations."}

URGENCY = ["Not urgent.", "Should be handled soon.", "Blocking, handle now."]


def wide(n, prefix="opt"):
    """n options, enough to push past each alphabet boundary."""
    return {f"{prefix}{i:02d}": f"Option number {i} of {n}." for i in range(n)}


def long_state(target_words=3000):
    """a state large enough that prefix reuse is the whole point."""
    para = ("The customer contacted support about a recurring charge on their "
            "account and the agent escalated the matter for further review. ")
    return (para * (target_words // len(para.split()) + 1)).strip()


CASES = [
    {"id": "choice-two", "description": "the smallest choice question",
     "state": "Customer cannot access an account after a password reset.",
     "questions": {"queue": {"type": "choice", "instructions": "Which queue should handle this?",
                             "criteria": {"access": "Account access support.",
                                          "billing": "Billing support."}}}},
    {"id": "choice-three", "description": "three options, string state",
     "state": TICKET,
     "questions": {"queue": {"type": "choice", "instructions": "Which queue should handle this?",
                             "criteria": ROUTING}}},
    {"id": "choice-null-description",
     "description": "a null description falls back to the option id",
     "state": TICKET,
     "questions": {"queue": {"type": "choice", "instructions": "Which queue should handle this?",
                             "criteria": {"billing": None, "technical": "Bugs and outages."}}}},
    {"id": "choice-empty-description",
     "description": "an empty description behaves as null",
     "state": TICKET,
     "questions": {"queue": {"type": "choice", "instructions": "Which queue should handle this?",
                             "criteria": {"billing": "", "technical": "Bugs and outages."}}}},
    {"id": "boolean-default-criteria",
     "description": "boolean with no criteria uses the spec defaults",
     "state": "If this is not fixed today I am cancelling our subscription.",
     "questions": {"churn": {"type": "boolean",
                             "instructions": "Does the sender threaten to cancel?"}}},
    {"id": "boolean-custom-criteria", "description": "boolean with caller supplied labels",
     "state": "If this is not fixed today I am cancelling our subscription.",
     "questions": {"churn": {"type": "boolean",
                             "instructions": "Does the sender threaten to cancel?",
                             "criteria": {"true": "Yes, they threaten to leave.",
                                          "false": "No, they do not."}}}},
    {"id": "boolean-wire-alias-noul",
     "description": "the jev wire name for boolean is accepted and normalised",
     "state": "If this is not fixed today I am cancelling our subscription.",
     "questions": {"churn": {"type": "noul",
                             "instructions": "Does the sender threaten to cancel?"}}},
    {"id": "score-three-levels", "description": "an ordered rubric",
     "state": "This is the third outage this week and we are losing sales.",
     "questions": {"urgency": {"type": "score", "instructions": "How urgent is this?",
                               "criteria": URGENCY}}},
    {"id": "score-four-levels", "description": "a longer rubric",
     "state": "The export button throws a 500 on every browser we tried.",
     "questions": {"severity": {"type": "score", "instructions": "How severe is this?",
                                "criteria": ["Cosmetic or informational.",
                                             "Degraded, but a workaround exists.",
                                             "Blocking with no workaround.",
                                             "Blocking and causing data loss."]}}},
    {"id": "multi-question", "description": "several questions over one shared state",
     "state": TICKET,
     "questions": {"queue": {"type": "choice", "instructions": "Which queue should handle this?",
                             "criteria": ROUTING},
                   "churn": {"type": "boolean", "instructions": "Do they threaten to cancel?"},
                   "urgency": {"type": "score", "instructions": "How urgent is this?",
                               "criteria": URGENCY}}},
    {"id": "multi-label-booleans",
     "description": "multi-label is several booleans, not a type of its own",
     "state": TICKET,
     "questions": {"refund": {"type": "boolean", "instructions": "Do they want money back?"},
                   "angry": {"type": "boolean", "instructions": "Are they angry?"},
                   "repeat": {"type": "boolean", "instructions": "Is this a repeat contact?"}}},
    {"id": "state-object", "description": "an object state, serialised deterministically",
     "state": {"subject": "Stripe sync broken", "plan": "pro", "previousTickets": 2,
               "message": "My Stripe connection has failed for three days."},
     "questions": {"queue": {"type": "choice", "instructions": "Which queue should handle this?",
                             "criteria": ROUTING}}},
    {"id": "state-object-key-order",
     "description": "the same object declared in a different key order must serialise identically",
     "state": {"previousTickets": 2, "message": "My Stripe connection has failed for three days.",
               "plan": "pro", "subject": "Stripe sync broken"},
     "questions": {"queue": {"type": "choice", "instructions": "Which queue should handle this?",
                             "criteria": ROUTING}}},
    {"id": "state-nested-object", "description": "nested structures sort at every level",
     "state": {"ticket": {"id": 41, "body": "Charged twice.", "author": {"tier": "pro", "id": 7}},
               "account": {"region": "eu", "created": "2024-01-02"}},
     "questions": {"queue": {"type": "choice", "instructions": "Which queue should handle this?",
                             "criteria": ROUTING}}},
    {"id": "state-array", "description": "an array state",
     "state": [{"role": "user", "text": "Where is my refund?"},
               {"role": "agent", "text": "Let me check."},
               {"role": "user", "text": "It has been two weeks."}],
     "questions": {"resolved": {"type": "boolean", "instructions": "Was the issue resolved?"}}},
    {"id": "state-non-ascii",
     "description": "non-ascii in an object state is escaped, so the prompt stays byte-stable",
     "state": {"name": "José", "note": "café charge disputed"},
     "questions": {"queue": {"type": "choice", "instructions": "Which queue should handle this?",
                             "criteria": ROUTING}}},
    {"id": "state-trailing-newline",
     "description": "a state ending in a newline still meets the delimiter cleanly",
     "state": "The report is wrong but we can work around it.\n",
     "questions": {"urgency": {"type": "score", "instructions": "How urgent is this?",
                               "criteria": URGENCY}}},
    {"id": "state-empty-string", "description": "an empty state is legal",
     "state": "",
     "questions": {"queue": {"type": "choice", "instructions": "Which queue should handle this?",
                             "criteria": ROUTING}}},
    {"id": "boundary-question-leading-digit",
     "description": "a suffix opening on a digit, where naive joint tokenisation can merge",
     "state": "Ticket ends with a digit 42",
     "questions": {"q": {"type": "choice",
                         "instructions": "42 is the answer, but what is the question?",
                         "criteria": {"yes": "It is.", "no": "It is not."}}}},
    {"id": "boundary-state-ends-punctuation",
     "description": "a state ending in punctuation against the delimiter",
     "state": "Ticket ends with punctuation!",
     "questions": {"q": {"type": "boolean", "instructions": "Is this urgent?"}}},
    {"id": "options-eight", "description": "eight options, the old label ceiling",
     "state": TICKET, "questions": {"pick": {"type": "choice",
                                             "instructions": "Which applies?",
                                             "criteria": wide(8)}}},
    {"id": "options-twentysix",
     "description": "26 options, the last count using uppercase labels only",
     "state": TICKET, "questions": {"pick": {"type": "choice",
                                             "instructions": "Which applies?",
                                             "criteria": wide(26)}}},
    {"id": "options-twentyseven",
     "description": "27 options, the first count to extend into lowercase labels",
     "state": TICKET, "questions": {"pick": {"type": "choice",
                                             "instructions": "Which applies?",
                                             "criteria": wide(27)}}},
    {"id": "options-fiftytwo", "description": "52 options, the single-pass maximum",
     "state": TICKET, "questions": {"pick": {"type": "choice",
                                             "instructions": "Which applies?",
                                             "criteria": wide(52)}}},
    {"id": "single-option",
     "description": "one option is legal; only option mass carries information",
     "state": "Page: flight search.",
     "questions": {"type_text_target": {
         "type": "choice", "instructions": "Which field should be filled?",
         "criteria": {"1": "[1] Departure date"}}}},
    {"id": "instructions-object",
     "description": "instructions as an object, the shape browser-use's jev agent sends",
     "state": "Page: flight search.",
     "questions": {"operation": {
         "type": "choice",
         "instructions": {"goal": "Find a flight to Reykjavik",
                          "rules": ["Fill required fields before submitting.",
                                    "Prefer visible controls over waiting."]},
         "criteria": {"CLICK": "Click an element.", "DONE": "Everything is satisfied."}}}},
    {"id": "description-object",
     "description": "option descriptions as objects, one line each so the label column survives",
     "state": "Page: flight search.",
     "questions": {"click_target": {
         "type": "choice",
         "instructions": "Which element should be clicked?",
         "criteria": {
             "1": {"element": "[1] Departure date", "current_value": "12 Mar",
                   "role": "textbox"},
             "2": {"element": "[2] Search", "current_value": "", "role": "button"}}}}},
    {"id": "description-array",
     "description": "an array description also renders on one line",
     "state": "Page: flight search.",
     "questions": {"pick": {"type": "choice", "instructions": "Which applies?",
                            "criteria": {"a": ["first", "second"], "b": "plain"}}}},
    {"id": "state-long",
     "description": "a state of several thousand tokens, where prefix reuse is the whole point",
     "state": long_state(),
     "questions": {"queue": {"type": "choice", "instructions": "Which queue should handle this?",
                             "criteria": ROUTING},
                   "urgency": {"type": "score", "instructions": "How urgent is this?",
                               "criteria": URGENCY}}},
]


def build(case, formatter):
    questions = types.parse_questions(case["questions"])
    prefix = prompt.render_prefix(formatter, case["state"])
    expected = {"prefix": prefix, "questions": {}}
    for q in questions:
        expected["questions"][q.name] = {
            "kind": q.kind,
            "suffix": prompt.render_suffix(formatter, q),
            "labels": prompt.label_map(q),
            "flags": prompt.alphabet_flags(q),
        }
        if q.legend:
            expected["questions"][q.name]["legend"] = q.legend
    return {"id": case["id"], "description": case["description"],
            "spec_version": spec.constants()["spec_version"], "formatter": "reference",
            "state": case["state"], "questions": case["questions"], "expected": expected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if regenerating would change a fixture")
    args = ap.parse_args()

    formatter = types.Formatter.load(os.path.join(SPEC, "formatters", "reference.json"))
    os.makedirs(OUT, exist_ok=True)

    changed = []
    for case in CASES:
        body = json.dumps(build(case, formatter), indent=2) + "\n"
        path = os.path.join(OUT, case["id"] + ".json")
        if args.check:
            existing = None
            if os.path.exists(path):
                with open(path) as f:
                    existing = f.read()
            if existing != body:
                changed.append(case["id"])
            continue
        with open(path, "w") as f:
            f.write(body)

    if args.check:
        if changed:
            sys.exit(f"fixtures would change: {', '.join(changed)}")
        print(f"{len(CASES)} fixtures match")
    else:
        print(f"wrote {len(CASES)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
