"""the prompt builder. spec/SPEC.md section 5 is normative; this implements it.

the split into prefix and suffix is the whole performance story: the prefix is
everything shared across questions and is prefilled once, each question is a
short suffix appended to it.
"""

import hashlib

from . import spec


def render_prefix(formatter, state):
    """system instructions plus the state. identical for every question."""
    c = spec.constants()
    return (formatter.system_open + c["system_instructions"] + formatter.system_close
            + formatter.user_open + spec.serialise_state(state) + c["state_delimiter"])


def render_suffix(formatter, question, alphabet=None):
    """the question, its labelled options, and the assistant opening.

    the assistant opening ends the suffix so that the very next token position
    is the label, which is the position being scored.
    """
    c = spec.constants()
    labels = spec.labels(len(question.options), alphabet)
    lines = "\n".join(
        c["option_line"].format(label=label, description=option.description)
        for label, option in zip(labels, question.options, strict=True))
    return (question.instructions + "\n" + lines + c["answer_instruction"]
            + formatter.user_close + formatter.assistant_open)


def label_map(question, alphabet=None):
    """option id to label, in declaration order."""
    labels = spec.labels(len(question.options), alphabet)
    return {option.id: label
            for label, option in zip(labels, question.options, strict=True)}


def prompt_sha256(prefix, suffix):
    """provenance, so a result can be traced to the exact bytes that produced it."""
    return hashlib.sha256((prefix + suffix).encode()).hexdigest()


def alphabet_flags(question):
    """what the caller should know about how this question had to be encoded."""
    c = spec.constants()
    flags = []
    if len(question.options) > c["max_options_plain_alphabet"]:
        # the lowercase labels are verified single-token but not quality tested
        flags.append("wide_alphabet")
    if len(question.options) > len(c["labels"]):
        # past A-Za-z the labels come from the model's own vocabulary. verified
        # single tokens at the scored position, and nothing beyond that: whether
        # a model reaches for an unfamiliar label as readily as `A` is measured
        # in docs/EVALS.md, not assumed here.
        flags.append("extended_alphabet")
    if len(question.options) == 1:
        # nothing was chosen between; only option mass carries information
        flags.append("single_option")
    return flags
