"""affix recovery from a chat template, offline.

spec/SPEC.md 4.3: the formatter is derived when a model is first used. these
cases run against templates written here rather than a live model, so the
diffing logic is covered without weights. the byte-for-byte agreement with
`spec/formatters/` is asserted in tests_e2e, since it needs the real template.
"""

import pytest

from llama_verdict import derive

QWEN = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# a different marker set entirely, and a system turn that opens as a user turn
# the way gemma's does -- the affixes are whatever the template emits, never a
# chatml assumption
GEMMA_ISH = (
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}<|turn>user\n{{ m['content'] }}\n"
    "{% elif m['role'] == 'user' %}<|turn>user\n{{ m['content'] }}<|end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n{% endif %}"
)


def test_recovers_a_chatml_template():
    affixes = derive.affixes_from_template(QWEN)
    assert affixes["system_open"] == "<|im_start|>system\n"
    assert affixes["system_close"] == "<|im_end|>\n"
    assert affixes["user_open"] == "<|im_start|>user\n"
    assert affixes["user_close"] == "<|im_end|>\n"
    assert affixes["assistant_open"] == "<|im_start|>assistant\n"


def test_recovers_a_template_that_is_not_chatml():
    """the affixes are whatever the template emits. measured on the real
    models: qwen opens the assistant turn with a thinking block and gemma with
    `<|turn>model\\n`, and a formatter fitted to one steers the other to about
    1e-7 of the label mass."""
    affixes = derive.affixes_from_template(GEMMA_ISH)
    assert affixes["assistant_open"] == "<|turn>model\n"
    assert affixes["system_open"] == "<|turn>user\n"
    assert affixes["user_open"] == "<|turn>user\n"


def test_the_bos_token_lands_in_the_system_opening():
    """a template that emits bos puts it before everything, and it has to end
    up in the prefix rather than being silently dropped."""
    withbos = "{{ bos_token }}" + QWEN
    affixes = derive.affixes_from_template(withbos, bos="<s>")
    assert affixes["system_open"] == "<s><|im_start|>system\n"


class FakeBackend:
    """enough of a backend to exercise build() without weights.

    every label is one token and the scored mass all lands on the first label,
    so the only thing under test is which opening reaches the prompt.
    """

    def __init__(self, template):
        self.template = template
        self.prompts = []

    def props(self):
        return {"chat_template": self.template, "bos_token": "", "eos_token": "",
                "build_info": "fake", "model_alias": "fake"}

    def readout(self):
        return None

    def tokenize(self, text, add_special=False):
        # one token per character, so that tokenize(a + b) really does equal
        # tokenize(a) + tokenize(b) -- the merge invariant under test
        return [ord(c) for c in text]

    def score(self, prompt, label_ids):
        self.prompts.append(prompt)
        return {"raw": {label_ids[0]: 1.0, **{i: 0.0 for i in label_ids[1:]}},
                "truncated": False, "n_probs_used": 64, "retries": 0,
                "score_ms": 1.0, "prompt_n": 1, "prompt_total": 1,
                "tokens_cached": 0, "top_unconstrained": "A"}


def test_an_explicit_assistant_opening_overrides_the_derived_one(tmp_path):
    """harmony's generation prompt ends one control token before content: at
    `<|start|>assistant` gpt-oss puts p=1.0000 on `<|channel|>` and the label
    never gets a turn, so option mass reads 0.0000. spelling the channel out
    takes the same model to 0.9909. the template is right and the derivation
    is right, so the escape hatch is an override rather than a fix."""
    backend = FakeBackend(QWEN)
    harmony = "<|start|>assistant<|channel|>final<|message|>"
    formatter, _ = derive.build(backend, "fake:Q8_0", cache_dir=str(tmp_path),
                                assistant_open=harmony)
    assert formatter.assistant_open == harmony
    assert backend.prompts and backend.prompts[0].endswith(harmony)


def test_without_an_override_the_derived_opening_is_used(tmp_path):
    backend = FakeBackend(QWEN)
    formatter, _ = derive.build(backend, "fake:Q8_0", cache_dir=str(tmp_path))
    assert formatter.assistant_open == "<|im_start|>assistant\n"


def test_an_unclosed_block_in_the_opening_is_closed():
    """lfm2.5-2.6b hardcodes `<|im_start|>assistant\\n<think>` as its generation
    prompt with no enable_thinking branch, so the label position lands inside
    an open reasoning block and the model starts reasoning: 'The' at p=0.9956,
    option mass 5.7e-07. closing the block takes the same model to 0.9582.
    REMOVING it gives 0.0000, so the repair is to close, never to strip."""
    assert derive.close_open_blocks("<|im_start|>assistant\n<think>") == \
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_a_closed_block_is_left_alone():
    opening = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert derive.close_open_blocks(opening) == opening


def test_special_tokens_are_not_mistaken_for_blocks():
    """`<|start|>assistant` is not an unclosed tag, and closing it would be
    nonsense. gpt-oss needs a real override, not this repair."""
    for opening in ("<|start|>assistant", "<|turn>model\n",
                    "<|im_start|>assistant\n"):
        assert derive.close_open_blocks(opening) == opening


def test_a_template_that_drops_the_content_is_refused():
    """silence here would produce affixes that look plausible and steer
    nothing, which is the failure option mass exists to catch."""
    with pytest.raises(ValueError, match="dropped a sentinel"):
        derive.affixes_from_template("no messages here at all")


def test_a_template_with_no_generation_prompt_is_refused():
    """without an assistant opening there is no position to score at."""
    ungenerated = (
        "{% for m in messages %}"
        "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
        "{% endfor %}"
    )
    with pytest.raises(ValueError, match="empty assistant opening"):
        derive.affixes_from_template(ungenerated)
