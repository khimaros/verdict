"""the state ends at a checkpoint the server can resume from.

a sliding-window or recurrent model cannot roll its cache back to an arbitrary
position, so llama-server keeps checkpoints and resumes from one that sits
before the point where two prompts diverge. it places one at the start of a
user span it finds by matching `message_delimiters` against the prompt's
tokens. marking the end of the shared state that way let a second question
over a ~4.5k token state reuse it: 4.4 s to 1.05 s on qwen3.5-4b, 5.3 s to
0.9 s on winnow-12b. the match is by tokens, so an anchor is kept only where
its tokens really do end the rendered state, which differs by tokenizer.
"""

from llama_verdict import derive

TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")


class Server:
    """a tokenizer that folds whitespace before a newline into it, as qwen's
    does: ` \\n` is one token, so a delimiter opening on `\\n` will not match
    after a state that ends on a space."""

    model = "fake"

    def props(self):
        return {"chat_template": TEMPLATE, "bos_token": "", "eos_token": ""}

    def readout(self):
        return None

    def tokenize(self, text, add_special=False):
        tokens, i = [], 0
        while i < len(text):
            if text[i] == " " and text[i + 1:i + 2] == "\n":
                tokens.append(1000)
                i += 2
            else:
                tokens.append(ord(text[i]))
                i += 1
        return tokens

    def score(self, prompt, label_ids, delimiters=None):
        return {"raw": {i: (1.0 if n == 0 else 0.0) for n, i in enumerate(label_ids)},
                "truncated": False, "n_probs_used": 64, "retries": 0, "score_ms": 1.0,
                "prompt_n": 1, "prompt_total": 1, "tokens_cached": 0,
                "top_unconstrained": "A"}


def test_the_anchor_is_one_the_tokenizer_keeps_at_the_end_of_the_state(tmp_path):
    formatter, _ = derive.build(Server(), "fake", cache_dir=str(tmp_path))
    # the chat layout's state delimiter is "\n---\n\n"; its leading newline
    # merges with a trailing space, so the anchor drops it
    assert formatter.checkpoint_anchor == "---\n\n"


def test_a_layout_whose_state_end_merges_falls_back_to_its_question_opening(tmp_path):
    # decider's state ends in a bare blank line, which qwen's tokenizer folds
    # into a trailing space; the suffix opens on "Question:" for every
    # question, and a checkpoint at its start is still before they diverge
    formatter, _ = derive.build(Server(), "fake", cache_dir=str(tmp_path),
                                layout="decider-plain")
    assert formatter.checkpoint_anchor == "Question:"


def test_a_tokenizer_that_keeps_newlines_apart_gets_the_whole_delimiter(tmp_path):
    class Plain(Server):
        def tokenize(self, text, add_special=False):
            return [ord(c) for c in text]

    formatter, _ = derive.build(Plain(), "fake", cache_dir=str(tmp_path))
    assert formatter.checkpoint_anchor == "\n---\n\n"
