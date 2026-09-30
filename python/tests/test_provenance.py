"""which weights a result was measured on, read off the server that loaded them.

the served model name is not a stable key: decider-4b:Q8_0 was one file in the
morning and another after a config reload. the gguf path llama-server reports
is, because the huggingface cache names the repo and revision in it.
"""

from llama_verdict.backend import weights_from_props

CACHED = ("/var/cache/huggingface/hub/models--Mapika--decider-4b-GGUF/"
          "snapshots/b79f09d9ba7837f1b744295ea267b55d08e958ec/decider-4b-v2.1-Q8_0.gguf")


def test_a_cached_hub_file_names_its_repo_revision_and_file():
    assert weights_from_props({"model_path": CACHED}) == {
        "repo": "Mapika/decider-4b-GGUF",
        "revision": "b79f09d9ba7837f1b744295ea267b55d08e958ec",
        "file": "decider-4b-v2.1-Q8_0.gguf"}


def test_a_file_in_a_subdirectory_keeps_its_path_within_the_repo():
    path = CACHED.replace("decider-4b-v2.1-Q8_0.gguf", "gguf/Winnow-12B-Q8_0.gguf")
    assert weights_from_props({"model_path": path})["file"] == "gguf/Winnow-12B-Q8_0.gguf"


def test_a_local_file_is_reported_as_a_path_and_nothing_is_guessed():
    assert weights_from_props({"model_path": "/models/x.gguf"}) == {
        "repo": None, "revision": None, "file": "/models/x.gguf"}


def test_a_server_that_does_not_say_reports_nothing():
    assert weights_from_props({}) is None
