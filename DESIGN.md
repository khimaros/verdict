# design

## the shape of the thing

verdict reads typed answers out of a model instead of generating them. one
forward pass, read the probability on each declared option label, renormalise.
everything else in this document exists to make that one operation identical
everywhere it runs.

## the contract is `/spec`, not any implementation

`/spec` holds the prompt layout, the question types, the result object and the
fitted formatter artifacts, plus golden fixtures. every implementation is
tested against the same fixtures. an implementation that disagrees with the
fixtures is wrong, including the one that came first.

this matters more than usual here because the definitive implementation is
going to be rewritten in a different language from the one it starts in. the
spec is the part that survives that.

## implementations and what each is for

| layer | language | role |
|---|---|---|
| core | rust | **definitive.** prompt building, formatter loading, scoring, calibration. owns all llama.cpp contact |
| reference client | python | evaluation harness, calibration fitting, spec conformance oracle. http only |
| bindings | dart | flutter library, on device and http backends |
| jev server | python now, rust later | drop-in jev wire protocol. deliberately a thin adapter |

the rust core is the long term target and the flutter library binds to it.
that also settles the backend language question the jev server raised: the
python server is a stopgap chosen because it is a thin adapter over code that
already exists, and it is expected to be replaced rather than grown.

the consequence for work happening now: **keep the python side thin.** it earns
its place as the eval harness and as a second implementation that proves the
spec is unambiguous. it is not the place to accumulate logic that will have to
be ported.

## layering

```
              llama.cpp (c++, pinned submodule)
                         |
                    ffi to llama.h
                         |
              verdict core (rust)  ..... definitive
                         |
                  extern "C", lv_ prefix
                         |
        +----------------+----------------+
        |                |                |
   dart ffi         python ctypes     td-cli / jev server
        |                |
  flutter plugin    eval harness
```

the narrow `lv_` c abi from the plan stays exactly as planned. the only change
is that rust provides it rather than hand-written c. that keeps the dart side
insulated from llama.cpp churn, which was the point of the shim, and it means
python can later drop its http dependency by loading the same library.

binding approach for flutter: plain `extern "C"` plus `dart:ffi`, not a
codegen bridge. it preserves the narrow surface, adds no build-time dependency
and matches what the plan already specified.

## the prefix and the suffix

the prompt is built in two pieces because the expensive half is reusable:

```
prefix   system instructions + state + delimiter     prefilled once per state
suffix   question + labelled options + assistant opening   one per question
```

both halves are tokenised separately and their token ids concatenated. the
boundary is chosen so that separate tokenisation equals joint tokenisation,
which is asserted per model rather than assumed. see `docs/DECISIONS.md`.

## formatters are derived, not written

a formatter is a small declarative table of affixes: how a system turn opens
and closes, how a user turn opens and closes, and what text opens the assistant
turn so the very next token is the label.

all of it is derived mechanically by rendering the model's own chat template
against sentinel messages and diffing the results. the assistant opening is
that template's generation prompt with `enable_thinking=false`. the template is
read from the gguf header, which is what llama.cpp actually loads, rather than
from a base repo that may not match.

formatters are per model, not per family: two models in one family routinely
ship different templates, and the difference is often exactly the generation
prompt. family is not a unit of anything here.

**derivation runs at model load, not ahead of time.** it is mechanical and
needs only the template the model already carries, so requiring someone to
have pinned a model in advance would buy nothing and would fail the case that
matters most: a user pointing the library at an arbitrary gguf on their own
device. the result is cached by template sha256, so it happens once per model.

exactly one jinja engine does this, inside the rust core. dart and python
reach it through the c abi rather than each shipping an engine, which is what
makes byte-identical output across languages achievable rather than
aspirational; two engines would mean two sets of whitespace and filter
semantics and therefore two prompts. the engine should be the one llama.cpp
already carries, since the core links llama.cpp anyway and that engine is what
the serving path applies.

`spec/formatters/<model>.json` are **golden regression fixtures**, not the
runtime source. they record what derivation produced for a known model on a
known build, so a change to the engine, the derivation or the template shows up
as a diff instead of as a silent change in behaviour.

deriving is cheap and deterministic; the measurement below is what confirms it
landed.

**the template describes how a model was packaged, which is not always how it
was trained.** a base model fine-tuned on a plain decision layout keeps its
base's chat template in the gguf, and derivation then builds a prompt the model
never saw. those models are read with a named layout, `spec/layouts/<name>.json`,
which declares its affixes and overrides the prompt constants as data, so the
chat layout stays the top level of `spec/constants.json` byte for byte.

which model needs which layout is not verdict's to record. it is a fact about
the model, and the model registry (aimbot) is canonical for model facts: it
reads the author's own config, such as decider's `decider_config.json`, and the
llama-swap config generated from it advertises `meta.llamaswap.readout` on
`/v1/models`. verdict reads that at derivation, so a new model in the registry
needs nothing in verdict. the split is deliberate: the registry says WHICH
readout, the spec says WHAT BYTES.

## option mass is load-bearing

option mass is the raw probability on the label tokens before renormalising. it
is not a diagnostic; it is the correctness check for the whole mechanism.

a model can score perfectly on an option mass of 1.7e-08, because renormalising
a rounding error still ranks it. accuracy cannot see that. option mass can. a
formatter that cannot clear the mass floor is refused at load time, and a
single result below the floor is flagged on the result object.

**it is also the only thing that says whether a model can do this work at
all.** measured across the local roster, with quality and evidence from
aimbot's aggregated boards:

| model | quality | evidence | option mass | usable |
|---|---|---|---|---|
| functiongemma-270m-it | 48 | **1/7** | **0.6936** | **no, refused at fit time** |
| gemma-4-e2b-it | 30 | 3/7 | 1.0000 | yes |
| qwen3.5-9b | 39 | 3/7 | 0.9999 | yes |
| qwen3.5-0.8b | 27 | 3/7 | 0.9979 | yes, and weak on the task |

this table first went in here reading "functiongemma scores higher than
qwen3.5-9b and cannot be used", which overstated it. **the composite weights
unmeasured factors as the median, not as zero**, so a model measured on one
factor of seven sits near 50 by construction. at 1/7 the 48 is mostly
imputation, and the right reading is that a thinly measured model behaved
unpredictably -- which is what `evidence` was telling us. filter on evidence
before sorting on quality.

what survives the correction is the part that matters: **a function-calling
tune is the closest thing on paper to typed decisions, and 31% of its mass
still went somewhere other than the labels.** no public leaderboard carries
the property, so the fit-time check is the answer rather than a formality.
whether a WELL measured model ever fails it is open and being measured.

## the label alphabet is asked for, not written down

the same argument as formatters, one layer along. `A`-`Za`-`z` gives 52
options, and past that the spec fell back to a tournament whose probabilities
are a hierarchical approximation sensitive to how the list happened to be
partitioned. a model's vocabulary carries thousands of single-character
tokens, and one high-`n_probs` call returns the whole of it with ids and text,
so the model can enumerate its own alphabet instead of being matched against a
table someone thought to write.

three constraints shape which characters qualify:

- **letters only.** a digit reads as part of the option text it labels and `)`
  is the option line delimiter. running out of ascii does not make either one
  a good label.
- **the pinned 52 first**, so a short list is labelled byte-identically either
  way and every fixture in `spec/fixtures/` stays valid.
- **one token after the assistant opening**, which is the position being read.
  being one token in isolation is a different claim and not the one relied on.
- **then by ascending token id**, because it is deterministic and a property of
  the tokenizer rather than of the process reading it. probabilities would be
  the more meaningful sort key and cannot be used as one: they are not
  bit-reproducible under load, so two derivations could order the alphabet
  differently and build different prompts for the same question.

two plausible refinements to that ordering have been measured and **both
rejected**. filtering characters that look like ascii letters takes qwen3.5-9b
from 16/16 to 11/16 and option mass from 0.7780 to 0.3146. filtering on
emission base rate is refused because base rate does not separate the group
that failed from the one that did not -- the failing group is emitted more
readily. base rate does separate ascii from everything else by about 300x,
which is already expressed by putting the pinned 52 first.

**so the alphabet is verified rather than predicted**, which is the same move
formatters made. an unambiguous question is labelled entirely from the wide
alphabet at three counts and both its option mass and its answer are read; the
gate needs both, because the model with the worst wide-alphabet accuracy of
four holds the highest wide option mass. a model that resolves an alphabet it
will not use is refused, or falls back to the tournament where one is enabled.
cheap properties of characters have now failed three times at predicting this
and a scoring pass answers it directly. see `docs/EVALS.md` 2a.

**the encoding limit and the quality limit are different limits, and removing
one does not touch the other.** measured with list length held fixed and every
label unfamiliar -- the worst case, not the shipped one -- qwen3.5-9b scores
16/16 under both alphabets and pays only in option mass, 0.9996 down to
0.7780. minicpm5-2b scores 16/16 on ascii and **9/16** on vocabulary labels:
same questions, same distractors, same positions. an unfamiliar label is cheap
for a 9b model and expensive for everything smaller.

**putting the pinned letters first is what keeps that worst case off the
shipped path**, and it was chosen for a different reason -- byte-identical
prompts below 52. a 104 option list is half ascii, and option mass measures
0.92 to 0.99 on all four models rather than the 0.36 to 0.78 of the stress
arm.

that is why it is opt-in and flagged `extended_alphabet` rather than a silent
fallback. it was also only visible because the question was asked with length
held fixed -- at 104 options the same drop would have been charged to the list
being long, which spec 10.3 already warns about.

## verdict decides; it never produces the answer text

this falls out of the mechanism and it shapes anything built on top. a decider
that emits typed choices and generates no tokens can navigate to the screen
holding `Android version / 15` and cannot say `15`. so an agent's actual output
has to be READ rather than generated:

| | works when | cost |
|---|---|---|
| read it from the structured data | the value is on the screen or in the page | a substring or regex match |
| a typed `choice` over candidates | the candidates are known in advance | one scoring pass |
| a generative extraction call | neither of the above | a generation, and the thing verdict declines |

the first is almost always available, because whatever was navigated to is
already text. both demos are built that way -- `--require` for scoring and
`--collect` for the task's output -- and neither puts a model on a value that
is sitting there in plain text. jev-ultrafast draws the same line: its
TEXT_MODEL exists only to invent a string to type into a field, never to decide
anything and never to report anything.

the corollary for evaluation is that a run is scored on what reached the
screen, not on the agent's own claim of completion. the agent's `DONE` is an
opinion, and one measured at 0.32 while a third of the goal was outstanding.

## what is provisional

- the python jev server, per above
- `scripts/probe_*.py`, which are phase 0 instruments and not product code
- everything about the native path until the rust crate exists; the plan's
  phase 4 acceptance criteria still apply, against rust rather than c
