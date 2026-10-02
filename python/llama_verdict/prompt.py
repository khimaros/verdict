"""the prompt builder. spec/SPEC.md section 5 is normative; this implements it.

the split into prefix and suffix is the whole performance story: the prefix is
everything shared across questions and is prefilled once, each question is a
short suffix appended to it.
"""

import hashlib

from . import spec

# stands for the number of options, in a layout's `answer_instruction`
COUNT = "{count}"

def render_prefix(formatter, state):
    """system instructions plus the state. identical for every question."""
    c = spec.layout(formatter.layout)
    opening = (formatter.system_open + c["system_instructions"] + formatter.system_close
               + formatter.user_open) if c["system_turn"] else formatter.bare_user_open
    return (opening + c["state_open"] + spec.render(state, "state", formatter.layout)
            + c["state_delimiter"])


def render_suffix(formatter, question, alphabet=None):
    """the question, its labelled options, and the assistant opening.

    the assistant opening, and any text the layout's model expects after it,
    ends the suffix so that the very next token position is the label, which
    is the position being scored.
    """
    c = spec.layout(formatter.layout)
    labels = spec.labels(len(question.options), alphabet, formatter.layout)
    lines = c["option_separator"].join(
        c["option_line"].format(label=label, description=option.description)
        for label, option in zip(labels, question.options, strict=True))
    # replaced rather than formatted: a layout's closing text may hold braces
    closing = c["answer_instruction"].replace(COUNT, str(len(labels)))
    return (c["question_open"] + question.instructions + c["options_open"] + lines
            + closing + formatter.user_close + formatter.assistant_open
            + c["assistant_prefill"])


def label_map(question, alphabet=None, layout=spec.CHAT):
    """option id to label, in declaration order."""
    labels = spec.labels(len(question.options), alphabet, layout)
    return {option.id: label
            for label, option in zip(labels, question.options, strict=True)}


def prompt_sha256(prefix, suffix):
    """provenance, so a result can be traced to the exact bytes that produced it."""
    return hashlib.sha256((prefix + suffix).encode()).hexdigest()


def alphabet_flags(question, layout=spec.CHAT):
    """what the caller should know about how this question had to be encoded."""
    c = spec.layout(layout)
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
