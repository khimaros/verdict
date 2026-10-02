"""a text prompt reaches the model with one bos, not two.

gemma's chat template renders `<bos>` itself, so the derived system opening
starts with it, and llama-server adds the model's bos again when it tokenizes a
text prompt: its log says "the final prompt starts with 2 BOS tokens". the
affixes stay exactly what the template renders; sending is what must not
repeat it. whether the server adds one is settled at derivation, which already
talks to the tokenizer, so serving makes no extra round trip.
"""

from llama_verdict import derive, types
from llama_verdict.decide import Decider

TEMPLATE = ("{{ bos_token }}{% for m in messages %}<|turn>{{ m['role'] }}\n"
            "{{ m['content'] }}<turn|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|turn>model\n{% endif %}")


class Server:
    """llama-server's shape: `/tokenize` with special tokens on adds a bos
    when the model's vocab says so. every label is one token and all the mass
    lands on the first."""

    def __init__(self, adds_bos):
        self.adds_bos = adds_bos
        self.prompts = []
        self.model = "fake"

    def props(self):
        return {"chat_template": TEMPLATE, "bos_token": "<bos>", "eos_token": ""}

    def registry_key(self):
        return None

    def tokenize(self, text, add_special=False):
        return ([2] if add_special and self.adds_bos else []) + [ord(c) for c in text]

    def score(self, prompt, label_ids, delimiters=None):
        self.prompts.append(prompt)
        return {"raw": {i: (1.0 if n == 0 else 0.0) for n, i in enumerate(label_ids)},
                "truncated": False, "n_probs_used": 64, "retries": 0, "score_ms": 1.0,
                "prompt_n": 1, "prompt_total": 1, "tokens_cached": 0,
                "top_unconstrained": "A"}


QUESTION = {"q": {"type": "choice", "instructions": "which?",
                  "criteria": {"a": "first", "b": "second"}}}


def decide_with(server, tmp_path):
    formatter, _ = derive.build(server, "fake", cache_dir=str(tmp_path))
    server.prompts.clear()
    Decider(server, formatter).decide("state", QUESTION)
    return formatter, server.prompts


def test_derivation_records_the_bos_a_server_adds(tmp_path):
    formatter, _ = decide_with(Server(adds_bos=True), tmp_path)
    assert formatter.server_bos == "<bos>"
    assert formatter.system_open.startswith("<bos>")


def test_a_server_that_adds_bos_gets_the_prompt_without_its_own(tmp_path):
    _, prompts = decide_with(Server(adds_bos=True), tmp_path)
    assert prompts and not prompts[0].startswith("<bos>")
    assert prompts[0].startswith("<|turn>system")


def test_a_server_that_adds_none_gets_the_prompt_as_rendered(tmp_path):
    formatter, prompts = decide_with(Server(adds_bos=False), tmp_path)
    assert formatter.server_bos == ""
    assert prompts[0].startswith("<bos><|turn>system")


def test_a_formatter_from_before_this_is_sent_unchanged():
    old = types.Formatter.from_dict({"affixes": {
        "system_open": "<bos>", "system_close": "", "user_open": "",
        "user_close": "", "assistant_open": "A"}})
    assert old.server_bos == ""
