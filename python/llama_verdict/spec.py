"""access to the shared specification.

the constants live in spec/constants.json rather than in this file so that
every implementation reads the same bytes. nothing here may hardcode a value
that the spec also defines.
"""

import functools
import json
import os

from . import config

# an empty value counts as unset, whether it comes from the environment or
# from a .env: a tool that exports its own unset variables would otherwise
# silently point the spec at ""
SPEC_DIR = config.get("LLAMA_VERDICT_SPEC") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "spec")


CHAT = "chat"


@functools.lru_cache(maxsize=1)
def constants():
    with open(os.path.join(SPEC_DIR, "constants.json")) as f:
        return json.load(f)


@functools.cache
def load_layout(name):
    """a named layout from spec/layouts, for a model trained on a text layout
    of its own rather than on the chat template its gguf carries."""
    path = os.path.join(SPEC_DIR, "layouts", f"{name}.json")
    if not os.path.exists(path):
        known = sorted(p.removesuffix(".json")
                       for p in os.listdir(os.path.join(SPEC_DIR, "layouts")))
        raise ValueError(f"unknown layout {name!r}; spec/layouts has {known}")
    with open(path) as f:
        return json.load(f)


@functools.cache
def layout(name=CHAT):
    """the constants a prompt is rendered with. the chat layout is the top
    level of constants.json; a named layout overrides some of them."""
    if name == CHAT:
        return constants()
    return {**constants(), **load_layout(name)["constants"]}


def labels(n, alphabet=None, layout_name=CHAT):
    """the first n option labels, in order.

    the alphabet is letters only. digits tokenise singly but add just ten and
    are easily confused with numbers appearing in option text. a named layout
    may use fewer, the letters its model was trained on.

    `alphabet` extends past the pinned 52 with single-character tokens read out
    of the model's own vocabulary. it must open with the pinned letters, so a
    list short enough for `A`-`Za`-`z` is labelled identically either way.
    """
    alphabet = layout(layout_name)["labels"] if alphabet is None else alphabet
    if n > len(alphabet):
        raise ValueError(
            f"{n} options exceeds the {len(alphabet)} single-token labels available. "
            f"shortlist the options, enable the wide alphabet so the model's own "
            f"vocabulary supplies more, or call tournament() to opt in to grouped "
            f"scoring with approximate probabilities.")
    return list(alphabet[:n])


def dump(value, profile):
    """a value as prompt text, under a serialisation profile.

    a string is used verbatim unless the profile quotes strings, which a
    layout whose model reads every value as a json literal needs. `escape_lt`
    is winnow's guard against a value opening a chat marker.
    """
    if isinstance(value, str) and not profile.get("quote_strings"):
        return value
    text = json.dumps(
        value, sort_keys=profile["sort_keys"], indent=profile["indent"],
        ensure_ascii=profile["ensure_ascii"],
        separators=(profile["item_separator"], profile["key_separator"]))
    return text.replace("<", "\\u003c") if profile.get("escape_lt") else text


def render(value, field, layout_name=CHAT):
    """a value as the layout renders `field`: state, instructions, description
    or option. a field with no profile is used as it is."""
    c = layout(layout_name)
    name = c[f"{field}_json"]
    return value if name is None else dump(value, c[name])


def serialise(value):
    """the block form, for a state or a question's instructions.

    sorted keys are not cosmetic: the state is the cached prefix, and an
    unstable serialisation destroys prefix reuse while still returning correct
    answers, which is the hardest kind of performance bug to notice.
    """
    return dump(value, constants()["block_json"])


def serialise_inline(value):
    """the one-line form, for an option description.

    real clients send descriptions as objects: browser-use's jev agent
    describes each candidate element as label, current value and aria role.
    indenting those would spread one option over several lines and destroy the
    one-line-per-option layout that makes the label column legible.
    """
    return dump(value, constants()["inline_json"])


# the chat layout's state form, kept under the name the public api exports
serialise_state = serialise
