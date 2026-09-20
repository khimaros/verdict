"""ask which target first, then ask which operation knowing what each would hit.

the android agent found this and it applies unchanged to the browser: a jev
client's speculative fan-out asks `operation` and `<operation>_target`
independently against one shared state, so the operation head cannot see that
a target head is certain.

measured on android, same screen and same moment:

    tap_target   'Folder: Settings'   0.9991
    operation    TAP                  0.1323   -> chose SCROLL_DOWN, wrong

naming the winning target in the operation question moved TAP to 0.9955. the
browser logs carry the same signature, operation confidence sitting below
target confidence on almost every step of the hacker news run:

    operation CLICK 0.646 / click_target 0.787
    operation CLICK 0.505 / click_target 0.873
    operation SCROLL_DOWN 0.726 / click_target 0.919

this splits one jev request into two and merges the answers back, so the
client sends and receives exactly what it did before.
"""

import sys

# the operation is decided last, once every target is known
OPERATION = "operation"
TARGET_SUFFIX = "_target"

# an operation that claims the task is finished, before anything has been done
TERMINAL = ("DONE",)


def split(body):
    """(targets request, the operation question) from one jev request."""
    questions = body.get("questions", {})
    targets = {name: q for name, q in questions.items() if name != OPERATION}
    operation = questions.get(OPERATION)
    if not targets or not operation:
        return None, None
    return dict(body, questions=targets), operation


def best_of(answer):
    """the chosen option and its probability, from a jev choice answer."""
    if not answer or "probabilities" not in answer:
        return None, 0.0
    choice = answer.get("choice")
    return choice, (answer["probabilities"] or {}).get(choice, 0.0)


def describe_targets(target_answers, criteria_by_question):
    """what each operation would actually do, named for the operation head."""
    described = {}
    for name, answer in target_answers.items():
        if not name.endswith(TARGET_SUFFIX):
            continue
        operation = name[: -len(TARGET_SUFFIX)].upper()
        choice, confidence = best_of(answer)
        if choice is None:
            continue
        criteria = criteria_by_question.get(name) or {}
        label = criteria.get(choice)
        if isinstance(label, dict):
            label = label.get("element") or label.get("label") or choice
        described[operation] = {"would_hit": str(label)[:120],
                                "confidence": round(confidence, 3)}
    return described


def operation_request(body, operation_question, described, withhold):
    """the operation question, told what each choice would do.

    `withhold` drops options the agent must not be offered: DONE before any
    action has been taken, and any operation already shown to change nothing
    on this page. an option that cannot honestly be chosen is better removed
    than argued against -- a threshold cannot catch a model that is confidently
    wrong, and this one claimed DONE at 0.95 having done nothing at all.
    """
    criteria = {k: v for k, v in operation_question["criteria"].items()
                if k not in withhold}
    if not criteria:
        criteria = dict(operation_question["criteria"])

    instructions = operation_question.get("instructions")
    instructions = dict(instructions) if isinstance(instructions, dict) else {
        "rules": instructions}
    instructions["what_each_operation_would_do"] = described
    if withhold:
        instructions["not_available_here"] = sorted(withhold)

    return dict(body, questions={OPERATION: dict(
        operation_question, criteria=criteria, instructions=instructions)})


def restore_operations(answer, criteria):
    """put withheld operations back at zero.

    the client validates that the returned distribution covers exactly the set
    it declared and rejects the whole response otherwise, so a proxy that
    removes an option has to restore the shape it changed.
    """
    probabilities = answer.get("probabilities")
    if probabilities is None:
        return answer
    for option in criteria:
        probabilities.setdefault(option, 0.0)
    return answer


def decide(body, ask, withhold=frozenset(), verbose=True):
    """one decision, target first. `ask` posts a jev request and returns answers."""
    targets_body, operation_question = split(body)
    if not targets_body:
        return ask(body)

    target_answers = ask(targets_body)
    criteria_by_question = {name: q.get("criteria") or {}
                            for name, q in body["questions"].items()}
    described = describe_targets(target_answers, criteria_by_question)

    op_body = operation_request(body, operation_question, described, withhold)
    operation_answers = ask(op_body)
    restore_operations(operation_answers.get(OPERATION, {}),
                       operation_question["criteria"])

    if verbose and described:
        summary = ", ".join(f"{op}->{d['would_hit'][:28]!r}@{d['confidence']}"
                            for op, d in described.items())
        print(f"  targets: {summary}", file=sys.stderr)

    return {**target_answers, **operation_answers}
