"""the question and formatter types, and parsing of the wire shape.

the wire shape is a named map of typed questions, which is what the jev
protocol uses. adopting it keeps the jev adapter a rename rather than a
translation. see docs/JEV_API.md.
"""

import dataclasses
import json

from . import spec

CHOICE, BOOLEAN, SCORE = "choice", "boolean", "score"

# jev calls the boolean type `noul`. it is translated at the wire edge and the
# name is not used internally.
WIRE_ALIASES = {"noul": BOOLEAN, "bool": BOOLEAN}


@dataclasses.dataclass(frozen=True)
class Option:
    id: str
    description: str


@dataclasses.dataclass(frozen=True)
class Question:
    name: str
    kind: str
    instructions: str
    options: tuple

    @property
    def legend(self):
        """score questions echo their rubric so a caller can render an answer
        without holding on to the request."""
        if self.kind != SCORE:
            return None
        return {o.id: o.description for o in self.options}


@dataclasses.dataclass(frozen=True)
class Formatter:
    """a derived affix table. see spec/SPEC.md section 4."""

    model: str
    system_open: str
    system_close: str
    user_open: str
    user_close: str
    assistant_open: str
    template_sha256: str = ""
    synthetic: bool = False
    # label token ids are a per-model constant. pinning them here removes one
    # http round trip per label from every cold decision.
    label_ids: dict = dataclasses.field(default_factory=dict)
    # the alphabet past the pinned 52, read out of the model's own vocabulary.
    # empty until a question needs it, because resolving it costs a
    # whole-vocabulary completion. insertion order IS the alphabet order.
    wide_label_ids: dict = dataclasses.field(default_factory=dict)
    # what scoring a list labelled entirely from that alphabet measured. a
    # model can be perfectly steerable on A-Za-z and put its mass somewhere
    # else entirely when handed characters it does not use.
    wide_verification: dict = dataclasses.field(default_factory=dict)
    verification: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, d):
        a = d["affixes"]
        return cls(model=d.get("model", ""), template_sha256=d.get("template_sha256", ""),
                   synthetic=d.get("synthetic", False),
                   label_ids=d.get("label_ids", {}),
                   wide_label_ids=d.get("wide_label_ids", {}),
                   wide_verification=d.get("wide_verification") or {},
                   verification=d.get("verification", {}), **a)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def check_usable(self):
        """a formatter that cannot steer the model is refused, not warned about.

        a wrong assistant opening still yields plausible looking answers: one
        model scored 10/10 on a smoke set at an option mass of 1.7e-08. the
        floor is set high because healthy and broken sit six orders of
        magnitude apart.
        """
        if self.synthetic:
            raise ValueError(
                f"formatter {self.model!r} is synthetic. it exists to make "
                f"conformance fixtures model-independent and is not a real chat "
                f"format; scoring a model with it is meaningless.")
        floor = spec.constants()["option_mass_floor_formatter"]
        mass = self.verification.get("mean_option_mass")
        if mass is None:
            raise ValueError(f"formatter for {self.model!r} carries no verification")
        if mass < floor:
            raise ValueError(
                f"formatter for {self.model!r} has mean option mass {mass:.4g}, "
                f"below the floor of {floor}. its assistant opening does not "
                f"steer this model to the labels.")


def parse_question(name, body):
    """one entry of the questions map into a Question."""
    kind = WIRE_ALIASES.get(body["type"], body["type"])
    # instructions and descriptions arrive as objects from real clients, not
    # just strings: browser-use's jev agent sends {"goal": ..., "rules": [...]}
    instructions = spec.serialise(body["instructions"])
    criteria = body.get("criteria")

    def describe(value, fallback):
        """a null or empty description means the option id speaks for itself."""
        return spec.serialise_inline(value) if value else fallback

    if kind == BOOLEAN:
        criteria = criteria or spec.constants()["boolean_default_criteria"]
        options = (Option("true", describe(criteria["true"], "true")),
                   Option("false", describe(criteria["false"], "false")))
    elif kind == CHOICE:
        if not criteria:
            raise ValueError(f"choice question {name!r} declares no criteria")
        options = tuple(Option(k, describe(v, k)) for k, v in criteria.items())
    elif kind == SCORE:
        if not criteria:
            raise ValueError(f"score question {name!r} declares no levels")
        options = tuple(Option(str(i), describe(d, str(i)))
                        for i, d in enumerate(criteria))
    else:
        raise ValueError(f"question {name!r} has unknown type {body['type']!r}")

    # a single option is legal and real clients send it: a target question for
    # an operation that only one element on the page supports. option mass
    # still says whether the model was steered, which is the useful part.
    if not options:
        raise ValueError(f"question {name!r} declares no options")
    return Question(name=name, kind=kind, instructions=instructions, options=options)


def parse_questions(questions):
    """option order is declaration order, and it reaches the prompt.

    python preserves insertion order for dicts and json.load preserves the
    order of object keys, so the caller's order survives. that order biases the
    answer; see spec section 9.
    """
    return [parse_question(name, body) for name, body in questions.items()]
