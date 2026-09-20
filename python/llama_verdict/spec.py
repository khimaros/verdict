"""access to the shared specification.

the constants live in spec/constants.json rather than in this file so that
every implementation reads the same bytes. nothing here may hardcode a value
that the spec also defines.
"""

import functools
import json
import os

SPEC_DIR = os.environ.get(
    "LLAMA_VERDICT_SPEC",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "spec"),
)


@functools.lru_cache(maxsize=1)
def constants():
    with open(os.path.join(SPEC_DIR, "constants.json")) as f:
        return json.load(f)


def labels(n, alphabet=None):
    """the first n option labels, in order.

    the alphabet is letters only. digits tokenise singly but add just ten and
    are easily confused with numbers appearing in option text.

    `alphabet` extends past the pinned 52 with single-character tokens read out
    of the model's own vocabulary. it must open with the pinned letters, so a
    list short enough for `A`-`Za`-`z` is labelled identically either way.
    """
    c = constants()
    alphabet = c["labels"] if alphabet is None else alphabet
    if n > len(alphabet):
        raise ValueError(
            f"{n} options exceeds the {len(alphabet)} single-token labels available. "
            f"shortlist the options, enable the wide alphabet so the model's own "
            f"vocabulary supplies more, or call tournament() to opt in to grouped "
            f"scoring with approximate probabilities.")
    return list(alphabet[:n])


def _dump(value, profile):
    if isinstance(value, str):
        return value
    opts = constants()[profile]
    return json.dumps(
        value, sort_keys=opts["sort_keys"], indent=opts["indent"],
        ensure_ascii=opts["ensure_ascii"],
        separators=(opts["item_separator"], opts["key_separator"]))


def serialise(value):
    """the block form, for a state or a question's instructions.

    sorted keys are not cosmetic: the state is the cached prefix, and an
    unstable serialisation destroys prefix reuse while still returning correct
    answers, which is the hardest kind of performance bug to notice.
    """
    return _dump(value, "block_json")


def serialise_inline(value):
    """the one-line form, for an option description.

    real clients send descriptions as objects: browser-use's jev agent
    describes each candidate element as label, current value and aria role.
    indenting those would spread one option over several lines and destroy the
    one-line-per-option layout that makes the label column legible.
    """
    return _dump(value, "inline_json")


# the state is the most visible caller, and the name reads better at the call
# site in prompt.render_prefix
serialise_state = serialise
