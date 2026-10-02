"""spec/layouts is verdict's copy of the model registry's layout table.

the registry is canonical for how a model is read, bytes included. verdict
keeps the blocks in its own form, committed, so that rendering never depends
on a server being up or on what it publishes, and refreshes them with
scripts/import_layouts.py. what makes a copy safe is a closed vocabulary: a
knob verdict does not know is bytes it cannot render, so an import carrying
one is refused whole rather than half applied.
"""

import copy
import glob
import json
import os
import sys

import pytest

from llama_verdict import derive, spec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))

import import_layouts  # noqa: E402

LAYOUTS = os.path.join(spec.SPEC_DIR, "layouts")
FILES = sorted(glob.glob(os.path.join(LAYOUTS, "*.json")))
NEWCOMER = {
    "source": "someone/newcomer @ abc123: prompt.py build()",
    "affixes": {"system_open": "", "system_close": "", "user_open": "[in]\n",
                "user_close": "\n", "assistant_open": "[out] ", "bare_user_open": "[in]\n"},
    "constants": {"system_turn": False, "question_open": "Ask: "},
    "head": {"kind": "linear", "repo": "someone/newcomer-gguf", "revision": "a" * 40,
             "file": "head.npz", "sha256": "b" * 64},
}


def table(**layouts):
    return {"version": 1, "layouts": layouts}


def read(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("path", FILES, ids=[os.path.basename(p) for p in FILES])
def test_every_layout_file_uses_only_knobs_the_spec_declares(path):
    spec.check_layout(read(path))


def test_a_knob_verdict_does_not_know_is_refused_by_name():
    with pytest.raises(ValueError, match="option_suffix"):
        spec.check_layout({"layout": "newcomer", "constants": {"option_suffix": "!"}})


def test_declared_affixes_have_to_be_complete():
    with pytest.raises(ValueError, match="user_close"):
        spec.check_layout({"layout": "newcomer", "constants": {},
                           "affixes": {"system_open": ""}})


def test_the_committed_layouts_survive_a_round_trip_through_the_registrys_shape(tmp_path):
    """what verdict hands the registry and what it reads back are the same
    blocks, so nothing is lost in either direction."""
    exported = import_layouts.table_from(LAYOUTS)
    report = import_layouts.import_layouts(exported, str(tmp_path))
    assert sorted(report["added"]) == sorted(exported["layouts"])
    for path in FILES:
        assert read(tmp_path / os.path.basename(path)) == read(path)
    again = import_layouts.import_layouts(exported, LAYOUTS, write=False)
    assert not again["added"] and not again["changed"]


def test_a_layout_new_to_verdict_is_written_with_its_head(tmp_path):
    report = import_layouts.import_layouts(table(newcomer=NEWCOMER), str(tmp_path))
    written = read(tmp_path / "newcomer.json")
    assert report["added"] == ["newcomer"]
    assert written["layout"] == "newcomer" and written["head"] == NEWCOMER["head"]
    assert written["spec_version"] == spec.constants()["spec_version"]
    spec.check_layout(written)


def test_an_unknown_knob_refuses_the_whole_import_and_writes_nothing(tmp_path):
    bad = copy.deepcopy(NEWCOMER)
    bad["constants"]["option_suffix"] = "!"
    with pytest.raises(ValueError, match="option_suffix"):
        import_layouts.import_layouts(table(fine=NEWCOMER, broken=bad), str(tmp_path))
    assert not os.listdir(tmp_path)


def test_a_check_reports_a_changed_block_and_leaves_the_file_alone(tmp_path):
    import_layouts.import_layouts(table(newcomer=NEWCOMER), str(tmp_path))
    moved = copy.deepcopy(NEWCOMER)
    moved["constants"]["question_open"] = "Query: "
    report = import_layouts.import_layouts(table(newcomer=moved), str(tmp_path), write=False)
    assert report["changed"] == ["newcomer"]
    assert read(tmp_path / "newcomer.json")["constants"]["question_open"] == "Ask: "


def test_importing_the_same_table_twice_changes_nothing(tmp_path):
    import_layouts.import_layouts(table(newcomer=NEWCOMER), str(tmp_path))
    report = import_layouts.import_layouts(table(newcomer=NEWCOMER), str(tmp_path))
    assert report == {"added": [], "changed": [], "unchanged": ["newcomer"], "local_only": []}


def test_a_layout_the_registry_dropped_is_reported_and_not_deleted(tmp_path):
    import_layouts.import_layouts(table(newcomer=NEWCOMER, other=NEWCOMER), str(tmp_path))
    report = import_layouts.import_layouts(table(newcomer=NEWCOMER), str(tmp_path))
    assert report["local_only"] == ["other"]
    assert os.path.exists(tmp_path / "other.json")


MODELS = {"someone/Newcomer": {"short": "newcomer", "repos": ["someone/Newcomer-GGUF"],
                               "readout": "layout:newcomer"}}


def test_the_recognised_models_are_imported_with_the_layouts(tmp_path):
    path = tmp_path / "models.json"
    assert import_layouts.import_models(dict(table(), models=MODELS), str(path)) == "added"
    assert read(path) == {"version": 1, "models": MODELS}
    assert import_layouts.import_models(dict(table(), models=MODELS), str(path)) == "unchanged"
    moved = copy.deepcopy(MODELS)
    moved["someone/Newcomer"]["readout"] = "head:newcomer"
    assert import_layouts.import_models(dict(table(), models=moved), str(path),
                                        write=False) == "changed"
    assert read(path)["models"] == MODELS


def test_a_table_without_models_leaves_the_copy_alone(tmp_path):
    path = tmp_path / "models.json"
    assert import_layouts.import_models(table(), str(path)) == "absent"
    assert not path.exists()


def test_a_model_entry_that_names_no_readout_is_refused(tmp_path):
    broken = {"someone/Newcomer": {"short": "newcomer", "repos": [], "readout": "newcomer"}}
    with pytest.raises(ValueError, match="someone/Newcomer"):
        import_layouts.import_models(dict(table(), models=broken), str(tmp_path / "models.json"))


def test_the_committed_models_survive_a_round_trip():
    exported = import_layouts.table_from(LAYOUTS)
    assert exported["models"] == spec.models()


def test_a_table_version_verdict_does_not_know_is_refused(tmp_path):
    with pytest.raises(ValueError, match="version"):
        import_layouts.import_layouts({"version": 2, "layouts": {}}, str(tmp_path))


def test_a_formatter_cached_for_one_set_of_bytes_is_not_served_for_another(
        tmp_path, monkeypatch):
    """the cache is keyed on what the prompt is built from, and an import can
    change a layout's bytes under a name that stays the same."""
    # read from the real spec before the directory is swapped, and then cached
    spec.constants()
    monkeypatch.setattr(spec, "SPEC_DIR", str(tmp_path))
    spec.load_layout.cache_clear()
    try:
        import_layouts.import_layouts(table(newcomer=NEWCOMER), str(tmp_path / "layouts"))
        first = derive.cache_key("template", None, "newcomer", None)
        moved = copy.deepcopy(NEWCOMER)
        moved["constants"]["question_open"] = "Query: "
        import_layouts.import_layouts(table(newcomer=moved), str(tmp_path / "layouts"))
        spec.load_layout.cache_clear()
        assert derive.cache_key("template", None, "newcomer", None) != first
    finally:
        spec.load_layout.cache_clear()
