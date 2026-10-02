"""which readout a recognised model needs.

the model registry is canonical for how a model is read. spec/models.json is
verdict's committed copy of its answer for the models it knows, and the only
place verdict looks: no server has to advertise anything, so a bare
llama-server, an unlisted id and a llama-swap all read a model the same way.
a model the copy does not know falls back to its own chat template.
"""

import pytest

from llama_verdict import derive, spec

HUB = "/srv/hub/models--{}/snapshots/0123abc/model.gguf"


def props(repo=None):
    return {"model_path": HUB.format(repo.replace("/", "--"))} if repo else {}


def test_every_recognised_model_names_a_layout_that_reads_it_that_way():
    for key, model in spec.models().items():
        assert model["repos"], key
        kind, _, name = model["readout"].partition(":")
        if (kind, name) == ("head", ""):
            # a head nobody can apply yet: recognised so that it is refused
            continue
        declares_head = bool(spec.load_layout(name).get("head"))
        assert (kind, declares_head) in (("layout", False), ("head", True)), key


def test_a_model_is_recognised_by_the_weights_the_server_loaded():
    """the served name is whatever a config calls it; the weights are the model."""
    assert spec.known_readout("ngquocvinh/Jev-Omni-GGUF", "anything:Q4_K_M") == "head:jev-omni"
    assert spec.known_readout("Mapika/decider-4b-GGUF", None) == "layout:decider-plain"


def test_a_model_loaded_from_a_plain_file_is_recognised_by_its_served_name():
    assert spec.known_readout(None, "jevk5-4b:Q8_0") == "layout:semif"
    assert spec.known_readout(None, "clef-flash:Q8_0") == "head:clef-flash"


def test_the_weights_outrank_the_name():
    assert spec.known_readout("EldanRing/Winnow-12B", "decider-4b:Q8_0") == "layout:winnow"


def test_an_unrecognised_model_has_no_entry():
    assert spec.known_readout("someone/New-Model-GGUF", "new-model:Q8_0") is None


@pytest.mark.parametrize("repo,model,want", [
    ("Mapika/decider-4b-GGUF", "served-as-something:Q8_0", "layout:decider-plain"),
    (None, "standardone-8b:Q8_0", "layout:standardone-native"),
    ("bartowski/Cloudflare_clef-flash-GGUF", "clef-flash:Q8_0", "head:clef-flash"),
    # a model nobody recognises is read by its own chat template
    ("someone/New-Model-GGUF", "new-model:Q8_0", None),
], ids=["by-weights", "by-name", "a-head", "unrecognised"])
def test_the_readout_is_the_copys_word_then_the_models_own_template(repo, model, want):
    assert derive.readout_for(model, props(repo)) == want
    if want is None:
        assert derive.layout_for(want) == spec.CHAT


@pytest.mark.parametrize("key,repo,model,want", [
    # the registry's own key for the model, where a server names it, is the
    # model's identity: it outranks the weights and the served name
    ("alibiserikbay/JevK5", "Mapika/decider-4b-GGUF", "winnow-12b:Q8_0", "layout:semif"),
    ("Cloudflare/clef", None, "anything:Q8_0", "head:clef"),
    # a key the copy does not carry says nothing, and the weights still do
    ("someone/New-Model", "Mapika/decider-4b-GGUF", "new:Q8_0", "layout:decider-plain"),
    ("someone/New-Model", None, "new:Q8_0", None),
], ids=["key-wins", "key-alone", "unknown-key-falls-to-weights", "unknown-everything"])
def test_a_model_the_server_names_by_registry_key_is_recognised_by_it(key, repo, model, want):
    assert derive.readout_for(model, props(repo), key) == want


def test_a_recognised_head_nobody_can_apply_is_refused_and_not_read_as_chat(monkeypatch):
    """the registry can name a model that answers through a head verdict has
    no way to apply yet. reading one by its chat template would measure its
    backbone and pass every health check doing it."""
    headed = {"someone/Headed": {"readout": "head", "repos": ["someone/Headed-GGUF"],
                                 "short": "headed"}}
    monkeypatch.setattr(spec, "models", lambda: headed)
    readout = derive.readout_for("headed:Q8_0", props("someone/Headed-GGUF"))
    assert readout == "head"
    with pytest.raises(ValueError, match="head"):
        derive.layout_for(readout)


def test_a_model_whose_format_stops_before_content_has_its_opening_on_file():
    assert spec.assistant_opening("gpt-oss-20b:Q8_0") == (
        "<|start|>assistant<|channel|>final<|message|>")
    assert spec.assistant_opening("qwen3.5-4b:Q8_0") is None
