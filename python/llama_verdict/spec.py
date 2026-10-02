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
# what a layout that declares its affixes has to declare
AFFIXES = ("system_open", "system_close", "user_open", "user_close", "assistant_open")
# the values a layout renders through a serialisation profile
SERIALISED = ("state", "instructions", "description", "option")
# a layout's kind, and the list in constants.json naming the knobs it may set.
# `single` asks one question per prompt and is read at one position; `joint`
# asks every question in one prompt and is read by token span (SPEC 5.5)
SINGLE, JOINT = "single", "joint"
KNOBS = {SINGLE: "layout_knobs", JOINT: "joint_knobs"}


@functools.lru_cache(maxsize=1)
def constants():
    with open(os.path.join(SPEC_DIR, "constants.json")) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def models():
    """the models this copy of the registry recognises, and how each is read.
    refreshed with the layouts by scripts/import_layouts.py."""
    with open(os.path.join(SPEC_DIR, "models.json")) as f:
        return json.load(f)["models"]


def stem(model_id):
    """a served id as the registry's short name spells it: `jevk5 4b` and
    `jevk5-4b:Q8_0` are one model."""
    return model_id.split(":")[0].strip().lower().replace(" ", "-")


def known_readout(repo=None, model_id=None, key=None):
    """the readout of a model this copy recognises, or None.

    by the registry's own key where the server names one, which is the model's
    identity. else by the hub repository its weights were loaded from, because
    the weights are the model and a served name is whatever a config chose to
    call it; by name only for weights loaded from a plain file.
    """
    if key in models():
        return models()[key]["readout"]
    table = list(models().values())
    for m in table:
        if repo and repo in m["repos"]:
            return m["readout"]
    for m in table:
        if model_id and m.get("short") and stem(m["short"]) == stem(model_id):
            return m["readout"]
    return None


def assistant_opening(model_id):
    """the assistant opening of a model whose template stops before content
    begins, so that deriving it reads nothing. None for every other model."""
    return constants()["assistant_openings"].get(stem(model_id))


def check_layout(block):
    """refuse a layout block verdict cannot render exactly.

    the knob vocabulary is closed (`layout_knobs` in constants.json). the
    blocks are imported from the model registry, and a knob from a registry
    newer than this verdict means bytes it does not know how to write:
    ignoring it would send a prompt the model was not trained on while every
    answer still looked fine.
    """
    name, knobs = block.get("layout"), block.get("constants")
    kind = block.get("kind", SINGLE)
    if kind not in KNOBS:
        raise ValueError(f"layout {name!r} is of kind {kind!r}; this verdict renders "
                         f"{sorted(KNOBS)}")
    if not isinstance(knobs, dict):
        raise ValueError(f"layout {name!r} declares no constants")
    unknown = sorted(set(knobs) - set(constants()[KNOBS[kind]]))
    if unknown:
        raise ValueError(f"layout {name!r} sets {unknown}, which this verdict does not "
                         f"know how to render. the model registry is newer than verdict.")
    missing = [a for a in AFFIXES if a not in block["affixes"]] if block.get("affixes") else []
    if missing:
        raise ValueError(f"layout {name!r} declares affixes without {missing}")
    if kind == JOINT:
        return block
    merged = {**constants(), **knobs}
    for field in SERIALISED:
        profile = merged[f"{field}_json"]
        if profile is not None and profile not in merged["profiles"] and profile not in merged:
            raise ValueError(f"layout {name!r} renders its {field} with the profile "
                             f"{profile!r}, which it does not define")
    return block


@functools.cache
def load_layout(name):
    """a named layout from spec/layouts, for a model trained on a text layout
    of its own rather than on the chat template its gguf carries. the files
    are verdict's copy of the model registry's blocks, refreshed by
    scripts/import_layouts.py."""
    path = os.path.join(SPEC_DIR, "layouts", f"{name}.json")
    if not os.path.exists(path):
        known = sorted(p.removesuffix(".json")
                       for p in os.listdir(os.path.join(SPEC_DIR, "layouts")))
        raise ValueError(f"unknown layout {name!r}; spec/layouts has {known}. if the "
                         f"model registry carries it, run make layouts.")
    with open(path) as f:
        return check_layout(json.load(f))


@functools.cache
def layout(name=CHAT):
    """the constants a prompt is rendered with. the chat layout is the top
    level of constants.json; a named layout overrides some of them."""
    if name == CHAT:
        return constants()
    c = {**constants(), **load_layout(name)["constants"]}
    if c["numbered_labels"]:
        # options numbered from 1, for a model read through a head: nothing
        # scores the label, so it need not be a single token
        c["labels"] = [str(i + 1) for i in range(c["numbered_labels"])]
    return c


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
    # a layout's own profiles first, then the two the chat layout defines
    return value if name is None else dump(value, c["profiles"].get(name) or c[name])


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
