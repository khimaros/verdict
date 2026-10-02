"""the prompt of a joint layout: every question in one prompt, read by span.

spec/SPEC.md 5.5. a single layout asks one question per prompt and reads one
position after it. a joint layout lists every question of the request as a
field of a schema and is read by a head that scores each option from the token
span it occupies, so the builder returns where each question and option sits
as well as the tokens.

each piece is tokenized on its own and the ids joined, as the author of the
one joint layout does: a span is then exact by construction, where splitting a
jointly tokenized prompt would have to guess at merges across a boundary.
"""

import typing

from . import spec, types


class Spans(typing.NamedTuple):
    """where one question sits in the prompt, in token positions: its type as
    the head numbers it, its instruction, and each of its options."""
    kind: int
    question: tuple
    options: tuple


class Field(typing.NamedTuple):
    name: str
    kind: str
    wire_type: str
    spans: Spans
    option_ids: tuple


def options_of(c, kind, body):
    """a question's options as (id, description), in the order they are listed."""
    criteria = body.get("criteria")
    if kind == types.BOOLEAN:
        described = {**c["boolean_default_criteria"], **(criteria or {})}
        return [(oid, described[oid]) for oid in c["boolean_order"]]
    if not criteria:
        raise ValueError(f"{kind} question declares no criteria")
    if kind == types.CHOICE:
        return sorted((str(oid), description) for oid, description in criteria.items())
    return [(str(level), description) for level, description in enumerate(criteria)]


def encode(tokenize, layout, state, questions, max_length=None):
    """the prompt as token ids, and a Field per question saying where it sits.

    `tokenize` turns text into token ids with special tokens parsed and none
    added. a state too long for the window is cut at its end, as the author
    cuts it, so that the schema and the answer opening always survive.
    """
    c = spec.layout(layout)
    limit = max_length or c["max_length"]
    pieces = tokenize

    def tokenize(text):
        return list(pieces(text))

    def render(value):
        return spec.dump(value, c["value_json"])

    schema, fields = tokenize(c["schema_open"]), []
    for number, (name, body) in enumerate(questions.items(), 1):
        kind = types.WIRE_ALIASES.get(body["type"], body["type"])
        if kind not in c["type_ids"]:
            raise ValueError(f"question {name!r} has unknown type {body['type']!r}")
        schema += tokenize(c["field_open"].format(number=number, id=name,
                                                  type=c["type_names"][kind]))
        asked = len(schema)
        # a question with no instructions is asked by its name
        instructions = body.get("instructions")
        schema += tokenize(render(str(name) if instructions in (None, "") else instructions))
        question, spans, ids = (asked, len(schema)), [], []
        schema += tokenize(c["options_open"])
        for position, (oid, description) in enumerate(options_of(c, kind, body), 1):
            schema += tokenize(c["option_open"].format(number=position))
            start = len(schema)
            schema += tokenize(render({"option_id": oid} if description is None else
                                      {"option_id": oid, "description": description}))
            spans.append((start, len(schema)))
            ids.append(oid)
            schema += tokenize(c["option_close"])
        schema += tokenize(c["field_close"])
        fields.append((name, kind, question, spans, ids))

    opening, closing = tokenize(c["prompt_open"]), tokenize(c["prompt_close"])
    room = limit - len(opening) - len(schema) - len(closing)
    if room < 0:
        raise ValueError(f"the schema takes {limit - room} tokens before any state and the "
                         f"window is {limit}. ask fewer questions at once.")
    body = tokenize(render(state))[:room]
    shift = len(opening) + len(body)

    def moved(span):
        return (span[0] + shift, span[1] + shift)

    return opening + body + schema + closing, [
        Field(name, kind, c["type_names"][kind],
              Spans(c["type_ids"][kind], moved(question), tuple(moved(s) for s in spans)),
              tuple(ids))
        for name, kind, question, spans, ids in fields]
