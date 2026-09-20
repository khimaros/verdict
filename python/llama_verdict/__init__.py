"""verdict: typed answers read out of a model, with no text generated.

the shared specification in spec/SPEC.md is normative. this package is the
reference implementation and the evaluation harness; the definitive
implementation is the rust core. see DESIGN.md.
"""

from .prompt import label_map, prompt_sha256, render_prefix, render_suffix
from .spec import constants, labels, serialise_state
from .types import (
    BOOLEAN,
    CHOICE,
    SCORE,
    Formatter,
    Option,
    Question,
    parse_question,
    parse_questions,
)

__all__ = [
    "BOOLEAN", "CHOICE", "SCORE",
    "Formatter", "Option", "Question",
    "constants", "labels", "serialise_state",
    "label_map", "parse_question", "parse_questions",
    "prompt_sha256", "render_prefix", "render_suffix",
]
