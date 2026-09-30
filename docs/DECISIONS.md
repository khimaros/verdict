# decisions

running log of what was verified and what was chosen. every entry states the
evidence, not just the conclusion. supersede entries in place, do not delete.

## the model ladder, measured on element selection

`scripts/smoke_models.py`, 16 cases per model: 4 option counts x 4 positions
for the correct answer, fixed distractors, no browser and no device. about a
minute a model.

| model | gib | correct | min option mass | median ms | position |
|---|---|---|---|---|---|
| qwen3.5-9b | 9.1 | **16/16** | 0.9996 | 1033 | even |
| **qwen3.5-4b** | **4.3** | **16/16** | 0.9981 | 959 | even |
| gemma-4-e4b-it | 7.7 | **16/16** | 0.9998 | 1175 | even |
| **minicpm5-2b** | **2.5** | **16/16** | 0.9995 | **374** | even |
| gpt-oss-20b | 11.3 | **16/16** | 0.9814 | 1025 | even |
| lfm2.5-2.6b | 2.9 | 16/16 | **0.1535** | 1511 | even |
| granite-4.2-3b | 3.6 | 14/16 | 0.9793 | 941 | even |
| gemma-4-e2b-it | 4.8 | 10/16 | 1.0000 | 459 | weak in the middle |
| g9v3-3b | 3.0 | 10/16 | 0.9390 | 390 | **never correct first** |
| qwen3.5-0.8b | 0.8 | 7/16 | 0.9932 | 535 | **never correct middle** |
| minicpm5-1b | 1.1 | 6/16 | 0.5200 | 327 | never correct first |

**qwen3.5-0.8b scores 7/16, reproducing the figure from the phase 0b run
exactly.** the benchmark is stable across months and code changes, which is
what makes the rest of the table worth reading.

**minicpm5-2b is the result that matters for phase 8**: 16/16 at 2.5 gib and
374 ms, the fastest model that is also perfect. qwen3.5-4b matches the 9b at
half the size. the phone-class band is no longer a compromise.

### lfm2.5 scores 16/16 on an option mass of 0.1535

the case "option mass is load-bearing" describes, now measured on a real model
rather than argued: **perfect accuracy on a readout that cannot be trusted.**
renormalising over a sixth of the distribution still ranks the options, and
accuracy cannot see it.

it also refines the phase 0b claim that "option mass does not degrade with
option count", which was measured on two models. lfm2.5 clears the formatter
check at 0.9569 on a two-option question and collapses to 0.15 by 52 options.
**so the fit-time verification is not sufficient on its own** -- it should
sample realistic option counts, not one small case.

### order averaging moves the errors, it does not remove them

supersedes the earlier one-line note ("no change at ceiling, +2/16 on a weak
model") with a mechanism and a boundary. scoring each option list under N
orders and averaging: N=2 is forward and reversed, N=4 adds seeded shuffles.

| model | baseline | N=2 | N=4 | the blind position |
|---|---|---|---|---|
| g9v3-3b | 10/16 | **14/16** | - | first 0 -> **3** |
| gemma-4-e2b-it | 10/16 | **14/16** | - | quarter 1 -> 3 |
| minicpm5-1b | 6/16 | 8/16 | 8/16 | first 0 -> 2 |
| qwen3.5-0.8b | 7/16 | 9/16 | 9/16 | middle 0 -> 0 -> **2** |
| granite-4.2-3b | 14/16 | 13/16 | - | - |
| qwen3.5-9b | 16/16 | 16/16 | - | - |
| minicpm5-2b | 16/16 | 16/16 | - | - |

the prediction was written down before the run and held, including its
negative half: **reversal cancels a DIRECTIONAL bias and cannot touch a
middle one**, because the middle of a reversed list is still the middle. both
first-blind models recovered at N=2; `qwen3.5-0.8b` stayed at 0 in the middle
and only moved once N=4 introduced shuffles that actually relocate it.

**and N=4 bought no accuracy at all** -- 9/16 at both N=2 and N=4, with first
4->3, last 4->3 and middle 0->2. the errors moved rather than went away. so:

- order averaging removes POSITION as a systematic confound
- it does not make a weak model stronger; the ceiling is the model
- the two ceiling models did not move, which is the control working

on n=16 a swing of +-2 is weak evidence. the clean signal is a position column
going 0 -> 3, not the total.

**so it is worth paying for trust, not for accuracy.** a probability that is
not a function of where the option happened to sit is worth more than one that
is, and that matters for phase 7: calibrating a position-contaminated
distribution fits the contamination. it is not worth paying to lift a weak
model, because it costs Nx latency and `minicpm5-2b` is already 16/16 at 2.5
gib and 374 ms with no correction.

### position bias is where the small models actually fail

three models are not uniformly weak, they are weak at a POSITION: `g9v3-3b`
and `minicpm5-1b` at FIRST, `qwen3.5-0.8b` in the MIDDLE. a benchmark that
always put the answer in one place would rank these models completely
differently, which is why the four positions are the point rather than a
detail.

**this said "never" and that was overstated.** each column is 4 cases and the
benchmark is not bit-reproducible: g9v3-3b read first 0/4 once and 1/4 on
re-run, and the case that flipped was a near-tie, correct at 0.487 against the
best distractor at 0.432. the DIRECTION holds -- first is its weakest column
by a wide margin -- and the absolute "never" does not. see `docs/EVALS.md` for
how much slop every number here carries.

## a generation prompt does not always end where content begins

surveyed option mass across thirteen served models, deriving each formatter at
runtime. eleven cleared the floor on the first try. the two that did not were
both TEMPLATE properties rather than model deficiencies, and both are usable
once the opening reaches the position where an answer would actually start:

| model | derived opening | mass | with the opening fixed |
|---|---|---|---|
| gpt-oss-20b | `<\|start\|>assistant` | 0.0000 | **0.9909** |
| lfm2.5-2.6b | `<\|im_start\|>assistant\n<think>` | 5.7e-07 | **0.9582** |

**gpt-oss** is harmony: the template's generation prompt is authoritative and
ends there, because the model is expected to pick a channel next. at that
position `<|channel|>` holds p=1.0000. three independent sources agree on the
opening -- the template's own generation-prompt block, our sentinel
derivation, and llama-server's `/apply-template` -- so nothing is
misconfigured. it needs `<|channel|>final<|message|>` spelled out, which is
what the `assistant_open` override is for.

**lfm2.5** hardcodes `"<|im_start|>assistant\n<think>"` with no
`enable_thinking` branch at all, so the label lands inside an open reasoning
block and the model reasons: `'The'` at p=0.9956. closing the block fixes it.
REMOVING the block gives 0.0000, so the repair is to close, never to strip.

closing an unclosed tag is a property of the string rather than knowledge
about a model, so `close_open_blocks()` does it automatically and there is no
per-model table. it is deliberately narrow: it fixes an unclosed block and
nothing else, and harmony still needs the explicit override.

**both failures were invisible to accuracy.** each model answered the smoke
question CORRECTLY while holding a millionth of the label mass, which is the
case in section "option mass is load-bearing" and the reason the floor is a
refusal rather than a warning.

## sentinel derivation stops where the template stops, not where content begins

surveyed option mass across seven models on the eval server, all at evidence
3/7 or better on aimbot's boards. six cleared the floor at 0.9963 to 1.0000.
`gpt-oss-20b:Q8_0` read **0.0000**, and that is not the model:

| assistant opening | option mass | correct |
|---|---|---|
| `<\|start\|>assistant` (derived) | 0.0000 | 0/1 |
| `<\|start\|>assistant<\|channel\|>final<\|message\|>` | **0.9909** | 1/1 |

harmony expects the model itself to emit the channel marker before any
content, so the label position never arrived. **a derived formatter's option
mass is therefore a property of the model AND of our derivation, and a low
reading does not distinguish them.** the refusal was still right -- that
formatter must not be used -- but the message now names the opening it
derived, because seeing `'<|start|>assistant'` is what makes the cause obvious.

gpt-oss also fails the 52-label alphabet: `'a'` is not a single token after its
opening while `A-Z` is fine. that caps it at 26 options per pass rather than
ruling it out, and it is the first model measured here where the label ceiling
and the mass floor disagree.

open: whether the opening can be completed without per-format special-casing
(the model's own greedy continuation takes the `analysis` channel, not `final`,
so it is not simply a generation), and whether option mass is stable across
quants, which needs a second quant on the server.

## phase 0: probe and decide

date: 2026-09-19
probe script: `scripts/probe_llama_server.py` (stdlib only, re-runnable anywhere)
raw report: `/tmp/lv_probe.json` (not committed; regenerate with the script)

### environment under test

| item | value |
|---|---|
| endpoint | the eval server's llama-swap, on the local network |
| fronted by | llama-swap (OpenAI surface at `/v1`, native endpoints at `/upstream/<model>/`) |
| llama.cpp build | `b11028-972d2313b` |
| model | `unsloth/Qwen3.5-0.8B-MTP-GGUF:Q8_0` (`qwen3.5-0.8b:Q8_0`) |
| n_ctx | 262144 |
| total_slots | 1 |

this is not a bare llama-server. `GET /props`, `POST /tokenize`,
`POST /apply-template` and `POST /completion` all return 404 at the root and
200 under `/upstream/<model>/`. the client must therefore take a base url plus
an optional upstream model segment, and support both deployments.

`total_slots = 1`. there is no server side parallelism on this box, so the
"concurrent" timing mode from the plan cannot beat the serial one here. the
client still implements bounded concurrency; the bound is just 1 by default
until `/props` reports more slots.

### probability readout

`POST /completion` with `n_predict: 1, n_probs: N` returns:

```
completion_probabilities[0].top_logprobs[] = {id, token, bytes, logprob}
```

`id` is the token id, which is what we match on. matching on `token` text is
wrong: the same letter appears as `"A"` and `" A"` with different ids.

**probabilities are a plain softmax of the logits and ignore sampler settings.**
verified: `temperature` of -1.0, 0.0, 0.5 and 1.0 (with `top_k: 0, top_p: 1.0`)
all returned identical probabilities to 6 decimal places. the server readme
frames this as a `temperature < 0` special case; on this build the readout is
pre-sampling regardless. we still send `temperature: -1.0` so the behaviour is
explicit and stable if that changes.

`post_sampling_probs: true` returns post-sampler probabilities instead: the
probe got a single entry at 1.0. **do not use it.** we rely on the default
`false`, sent explicitly.

**a grammar does not change the reported probabilities.** with
`grammar: root ::= "A" | "B"` the label probabilities were byte-identical to
the unconstrained call, and the reported top token was still the unconstrained
one (`"\n\n"`). grammars constrain sampled output, which we discard. no grammar.

**`n_probs` is not capped below the vocabulary on this build.** the sweep
returned exactly 1, 10, 20, 100, 1000 and 100000 entries for those requests.
full vocabulary readout is available over http, so plan risk 5 (top-n
truncation) does not bite here. we still implement the retry-then-flag path
from the plan, because other builds do cap it and the flag is cheap.

we do not need full vocabulary anyway: option mass is the sum of the raw
probabilities on the label tokens, and each label is read directly. a modest
`n_probs` with a retry when a label is missing is the right default.

### token id prompts and prefix reuse

`prompt` accepts an array of token ids. sending `tokenize(text)` produced the
same top token and the same probability (0.12823143008390858) as sending the
text. section 4.2 of the spec is therefore implementable as written.

prefix reuse verified with a 4219 token prefix:

| run | `cache_prompt` sent | `prompt_n` | `prompt_ms` |
|---|---|---|---|
| cold | true | 4219 | 702.1 |
| same suffix again | true | 4 | 104.8 |
| different suffix | true | 8 | 209.2 |
| omitted | (not sent) | 4 | 104.0 |

`prompt_n` counts tokens actually evaluated, so the server reuses the longest
common token prefix. `timings.cache_n` and the top level `tokens_cached` both
report the reuse and are worth surfacing as provenance.

**`cache_prompt` defaults to true on this build** (last row). the plan warns
its default has moved between versions, so we always send it explicitly.

cold and warm probabilities were bit-identical for the same prompt in this
single-slot configuration. that is weaker evidence than it looks: llama.cpp
does not guarantee it across batch shapes, and with one slot there is little
batch variation to expose. keep the argmax-exact / probability-tolerant
comparison discipline the plan calls for.

### prompt layout

qwen3.5 is a reasoning model and its template opens the assistant turn with
`<think>`. the relevant branch is:

```jinja
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is true %}
    {{- '<think>\n' }}
{%- else %}
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}
```

so the non-thinking assistant opening is:

```
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

`/apply-template` returned the **thinking** form (`...assistant\n<think>\n`),
which would put our label after an open think block. this is the concrete
reason the spec hand-writes formatters rather than delegating to the server:
we need the non-thinking form, and we need it byte-stable across two
implementations. `/apply-template` stays useful as a conformance oracle for the
system and user turns only.

decision: **labels are `A`..`H`, matched by token id, read at the first
position of the assistant turn.** verified on this tokenizer:

- `A`..`H` tokenize to single tokens, ids 32..39, with `add_special: false`
- `tokenize(assistant_open + "A") == tokenize(assistant_open) + [32]`, so the
  label does not merge with the header and there is no leading-space variant

### prefix/suffix boundary

the spec requires tokenising prefix and suffix separately and concatenating
ids. that only matches joint tokenisation if the boundary does not sit inside
a merge. it frequently does:

| boundary | result over 6 states x 3 questions |
|---|---|
| prefix ends `\n---\n`, suffix starts `\n` | 18/18 mismatches (one extra token) |
| prefix ends `\n---\n\n`, suffix starts with the question | clean |
| prefix ends `\n---\n`, suffix starts with the question | clean |
| prefix ends `<\|im_end\|>\n`, suffix starts with the question | clean |

decision: **the prefix ends with `\n---\n\n` and the suffix begins directly
with the question text.** chosen over the one-token-cheaper variant because the
blank line makes the rendered prompt easier to read, which the spec's
readability acceptance criterion asks for.

this is tokenizer dependent, so it is not a one-time result. the conformance
suite asserts `tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)`
per model and fails loudly otherwise.

### end to end sanity

the full layout, scored on the 0.8b model:

| state | question | result | option mass |
|---|---|---|---|
| locked out after password reset | which queue | access 0.89 / billing 0.11 | 0.9967 |
| charged twice, wants refund | which queue | billing 0.90 / access 0.10 | 0.9972 |
| third outage, threatens to cancel | how urgent | high 0.79 / med 0.15 / low 0.06 | 0.9940 |

3/3 correct argmax with option mass above 0.99 on a 0.8b model. option mass is
as good a health check as the plan claims; a drop below ~0.5 would mean the
prompt stopped steering the model to the labels.

this is three hand-picked cases, not an evaluation. it establishes that the
mechanism works, nothing about accuracy. that is phase 3.

### naming

repo is `verdict`. per the product owner: python package `llama_verdict`, dart
packages `llama_verdict` and `llama_verdict_llama`, c shim prefix `lv_`.

## phase 0b: cross-model layout, and where the formatter comes from

date: 2026-09-19
probe script: `scripts/probe_prompt_layout.py`

run against four models on the same server. gemma-4 is served by build
`b11056-e613ef2c8`, not the `b11028` that serves qwen3.5: llama-swap runs a
different llama-server binary per model, so "the build" is per model and the
provenance field must record it per request.

### the assistant opening is the model's own no-think template render

the plan assumes one hand-written formatter per model *family*. that is wrong,
but not for the reason this section first recorded. the correction is from the
aimbot session and is confirmed below.

mean option mass over the ten smoke cases, by model and candidate opening:

| model | opening | whose render | mean option mass | correct |
|---|---|---|---|---|
| qwen3.5-0.8b | `assistant\n<think>\n\n</think>\n\n` | **its own** | **0.9979** | 7/10 |
| qwen3.5-0.8b | `assistant\n` | none | 7.9e-09 | 6/10 |
| qwen3.5-0.8b | `assistant\n<think>\n` | the thinking branch | 5.3e-07 | 5/10 |
| qwen3.5-9b | `assistant\n<think>\n\n</think>\n\n` | **its own** | **0.9999** | 10/10 |
| qwen3.5-9b | `assistant\n` | none | 1.7e-08 | 10/10 |
| qwen3.5-9b | `assistant\n<think>\n` | the thinking branch | 9.1e-07 | 4/10 |
| gemma-4-e2b | `<\|turn>model\n` | **its own** | **1.0000** | 9/10 |
| gemma-4-e2b | `<\|turn>model\n<\|channel>thought\n\n<channel\|>` | the 12b's | 6.6e-08 | 5/10 |
| gemma-4-12b | `<\|turn>model\n` | the e2b's | 8.9e-07 | 9/10 |
| gemma-4-12b | `<\|turn>model\n<\|channel>thought\n\n<channel\|>` | **its own** | **1.0000** | 9/10 |

**every model reads about 1.0 on the prompt its own template renders with
`enable_thinking=false`, and about 1e-7 on any other.** that is the whole
result, and it is a much simpler one than first recorded here.

this section previously claimed the two gemma-4 models ship the same template
and need different openings, and concluded that the opening could not be
derived from the template. that was an error: only the e2b template was ever
fetched, and the 12b was assumed to match. it does not. diffed from
`/props.chat_template` on the serving build:

| | bytes | lines | sha256 |
|---|---|---|---|
| gemma-4-12b-it | 18921 | 389 | `aa3185dfc6505104` |
| gemma-4-e2b-it | 18807 | 385 | `74a88f94c57c14e2` |

the 12b carries a block the e2b has no trace of, in the generation prompt:

```jinja
{%- if not enable_thinking -%}
    {{- '<|channel>thought\n<channel|>' -}}
{%- endif -%}
```

so the two no-think generation prompts are `<|turn>model\n<|channel>thought\n<channel|>`
for the 12b and `<|turn>model\n` for the e2b, and the measurements above are
each model scored once on its own prompt and once on the other model's. both
models gate thinking, both gates are in the template, and the gates differ:
the 12b appends a pre-closed thought channel, the e2b relies on omitting the
`<|think|>` system turn that `enable_thinking=true` would add.

qwen3.5 fits the same rule. its no-think branch renders
`<|im_start|>assistant\n<think>\n\n</think>\n\n`, which is the row that scores
0.9979 and 0.9999.

the formatter is therefore **per model, because templates are per model**, and
it is derivable rather than searched.

one tolerance note: the 12b string measured here is
`thought\n\n<channel|>` where its template renders a single newline, and it
still read 1.0000. a near miss at the opening is survivable, so a healthy
option mass does not prove a byte-exact match. the spec uses the exact render
regardless.

### this is why option mass is a requirement, not a nicety

look at the qwen3.5-9b rows. with the plain opening it scores **10/10, the same
as the correct opening**, on an option mass of 1.7e-08. the renormalised
distribution is computed over labels that together hold a hundred-millionth of
the model's probability, and it still ranks them correctly.

accuracy cannot detect a broken prompt. option mass detects it immediately.
FR5 stands as a hard requirement, and the gate belongs in the library rather
than in a user's checklist.

the same effect explains the first gemma-4-12b run, which reported 9/10 at a
mass of 1e-7 and looked like a working configuration until the mass was read.
the cause there was mundane: it was being fed another model's generation
prompt. that is exactly the class of mistake option mass is there to catch,
and it caught it.

### can the templates be gathered automatically

yes, and the opening comes with them.

the template is mechanically available from the same place llama.cpp reads it:
the gguf header. over http it comes back in `/props.chat_template`; on device
it is in the gguf the plugin already loads. the base repo's
`chat_template.jinja` or `tokenizer_config.json` is the wrong file to trust,
because gguf conversions get patched on the way through and the patched copy is
what actually runs. that source selection is aimbot's finding and verdict
adopts it.

the assistant opening is just that template's generation prompt rendered with
`enable_thinking=false`. no search, no per-family knowledge.

decision: **the formatter is a derived artifact: the model's own template,
rendered.** the pipeline is:

1. take the chat template from the gguf header
2. render it with sentinel messages and diff the renders to recover the
   affixes: system prefix and suffix, user prefix and suffix, and the
   assistant opening from `add_generation_prompt=true, enable_thinking=false`
3. **verify** by measuring mean option mass over a fixed probe set
4. pin the result as `spec/formatters/<model>.json`, with the measured mass,
   the template sha256 and the build that measured it

step 3 is an assertion, not an objective function. it was nearly written as a
search, which would have worked and would have hidden the fact that the answer
was in the template all along.

both implementations load a small declarative affix table and neither needs a
jinja engine, which is what keeps python and dart byte-identical.

a formatter whose mass falls below a floor is a load-time error, not a warning.
the library refuses a model it cannot steer. given the measurements above the
floor can be set high: healthy is ~1.0 and broken is ~1e-7, six orders of
magnitude apart, with nothing observed in between.

### prior art on the fetch step

`../aimbot` already solves the gathering half and its approach should be
followed rather than reinvented. it keeps 164 templates on disk, deliberately
prefers huggingface's parse of the gguf header over any `chat_template.jinja`
in the same repo, and records that 17 gguf repos carry a template their base
repo has no trace of. it executes templates with real jinja2 under the
transformers environment, including the `{% generation %}` extension, and
probes thinking knobs for their accepted values and defaults.

verdict needs its own fetch, because it needs the affixes rather than aimbot's
role-acceptance facts, but the source selection and the renderer setup are
settled questions and the answers are there.

aimbot is considering emitting the rendered no-think generation prompt as a
registry field, which would cover the one piece verdict most wants. that is
their product owner's call. verdict derives and pins its own copy either way,
because the formatter has to travel with the spec rather than with a
dependency.

### smoke accuracy, for context only

10 hand-picked cases, not an evaluation:

| model | correct | mean option mass at the fitted opening |
|---|---|---|
| qwen3.5-0.8b | 7/10 | 0.9979 |
| qwen3.5-9b | 10/10 | 0.9999 |
| gemma-4-e2b | 9/10 | 1.0000 |
| gemma-4-12b | 9/10 | 1.0000 |

the one case every model misses is the three-level urgency rubric scored `med`
where the state says the report is wrong but a workaround exists until next
week. every model answers `low`. that is a rubric wording problem, not a
formatter problem, and it belongs to phase 3.

### how many options, and does confidence mean anything

date: 2026-09-19
probe scripts: `scripts/probe_label_alphabet.py`, `scripts/probe_option_scaling.py`

**the label ceiling is 62 and identical on both tokenizers.** usable means a
single token that is still its own token after the assistant opening, which is
the position being scored:

| alphabet | qwen3.5 | gemma-4 |
|---|---|---|
| `A`-`Z` | 26/26 | 26/26 |
| `a`-`z` | 26/26 | 26/26 |
| `0`-`9` | 10/10 | 10/10 |
| `10`-`99` | 0/90 | 0/90 |

every multi-digit number splits into digits, so numeric labels are out for any
list longer than ten. decision: **the alphabet is `A`-`Z`, extended with
`a`-`z` only when more than 26 options are declared.** digits are not used;
mixing them with letters buys 10 labels and invites confusion with option text.

**option mass does not degrade with option count.** scored on a browser element
table, one correct target among plausible distractors, at counts from 2 to 52:

| model | option mass, n=2 | n=26 | n=52 |
|---|---|---|---|
| qwen3.5-0.8b | 0.9962 | 0.9947 | 0.9932 |
| gemma-4-e2b | 1.0000 | 1.0000 | 1.0000 |

so the mechanism scales to a long list. accuracy is a different story:

| n | qwen3.5-0.8b | gemma-4-e2b |
|---|---|---|
| 2 | 2/4 | 4/4 |
| 4 | 4/4 | 4/4 |
| 8 | 3/4 | 4/4 |
| 16 | 2/4 | 2/4 |
| 26 | 1/4 | 3/4 |
| 52 | 1/4 | 1/4 |

**the finding that matters: gemma-4-e2b reports confidence 1.000 on answers
that are wrong.** at n=52 it was wrong on three of four placements, every one
at a top probability of 1.000 to three decimals, on an option mass of exactly
1.0000. its distribution is degenerate: it puts essentially all mass on one
label and is simply mistaken about which.

qwen3.5-0.8b behaves the opposite way. its mean top probability falls from
0.79 at n=4 to 0.41 at n=16 as its accuracy falls, so its confidence tracks its
competence.

two consequences.

first, **confidence is not comparable across models and is not safe to gate on
untuned.** a threshold fitted on qwen would pass everything gemma emits. this
is what NFR1 was reserving judgement about, and it is now a measured fact
rather than a caution. calibration is not an optional refinement for every
model; for gemma-4-e2b a raw confidence gate is worse than useless because it
fires hardest exactly when the model is wrong.

second, **option mass and confidence are independent and neither substitutes
for the other.** mass at 1.0 with confidence at 1.0 and the wrong answer is an
observed state. option mass proves the prompt is steering the model to the
labels. it proves nothing about whether the model is right. this bound belongs
in the docs wherever option mass is described, so it is not oversold.

position bias is present and is worth the order-averaging budget the plan sets
aside. qwen3.5-0.8b answered the first-listed option correctly at every count
from 16 up and missed most other placements; it also missed at n=2 whenever the
correct answer was second.

caveat on the design: the distractor list grows by prefix, so larger n also
means more confusable distractors, not just more of them. at n=2 the only
distractor is a site logo; by n=16 it includes "return date field", which is a
genuinely hard neighbour for "departure date field". accuracy across n is
therefore confounded and should not be read as a clean scaling curve. the
confidence findings do not depend on that, since they compare models at equal n.

### prior correction is harmful here, and the reason generalises

date: 2026-09-19
runner: `scripts/eval_interventions.py`, results in `eval/results/`

element selection with ground truth, one correct target among plausible page
distractors, at 8/16/26/52 options by four positions each. qwen3.5-9b:

| intervention | correct | P(correct) | mean ms |
|---|---|---|---|
| baseline | **16/16** | 0.991 | 1521 |
| order averaged, 2 orders | 16/16 | 0.952 | 3393 |
| **prior corrected** | **3/16** | **0.094** | 3267 |
| both | 2/16 | 0.062 | 5034 |

**prior correction, which the plan proposes as a debiasing step, destroyed the
result.** accuracy fell from 16/16 to 3/16.

the reason is not a bug. contextual calibration divides out what the model
answers when the state is empty, on the assumption that this residue is
prejudice about position and wording. **that assumption fails whenever the
option descriptions are themselves informative.** asked "which element opens
the departure date picker?" with no page at all, the model reads the option
list and correctly says "Departure date field". that is the right answer, so
the "prior" is concentrated on it, and dividing it out is precisely a
subtraction of the signal.

so the technique is sound only where the options carry no information and the
state carries all of it: a boolean whose criteria are "Yes." and "No.", a score
whose levels are a generic rubric. it is actively wrong for choice questions
whose descriptions are the evidence, which is every interesting case in the
browser and android demos.

decision: **prior correction stays off by default and is not promoted as a
general debiasing step.** where it is offered, the documentation has to say
which question shapes it suits. the plan's phase 7 wording, which presents it
as a generic option with a known cost, is too generous and this entry
supersedes it.

order averaging cost 2.2x the latency and changed nothing at this accuracy,
which is the expected result at ceiling rather than evidence against it. it
needs a benchmark with headroom to be judged, and qwen3.5-9b does not provide
one here: it answers 16/16 including at 52 options and at every position, which
also means the position bias measured earlier on smaller models is not present
in this one.

### the same benchmark on a weak model

qwen3.5-0.8b has the headroom qwen3.5-9b lacks:

| intervention | correct | first | quarter | middle | last | mean ms |
|---|---|---|---|---|---|---|
| baseline | 7/16 | 4/4 | 1/4 | 0/4 | 2/4 | 1977 |
| order averaged | **9/16** | 4/4 | 1/4 | 0/4 | **4/4** | 2923 |
| prior corrected | 1/16 | 0/4 | 1/4 | 0/4 | 0/4 | 2837 |
| both | 1/16 | 0/4 | 0/4 | 1/4 | 0/4 | 4127 |

the baseline shows the position bias plainly: 4/4 when the answer is listed
first, 0/4 when it is in the middle.

**order averaging helped, and helped exactly where the mechanism predicts.**
the whole gain is at `last`, 2/4 to 4/4. with two orders the second is the
reverse, so an option listed last becomes the option listed first and collects
the same first-position advantage. `middle` stayed at 0/4 because reversing
does not move the middle anywhere. lifting that would need three or more
permutations, at three or more times the cost.

so order averaging is a real but narrow fix: +2 of 16 for 1.5x the latency on a
model that needs it, and nothing at all on a model that does not. it stays off
by default and is worth enabling only for a model measured to have the bias.

**prior correction failed again, harder**, 7/16 to 1/16. two models, same
direction, same cause. the entry above stands.

worth noting across every row of both tables: **option mass never dropped below
0.992**, including where accuracy collapsed to 1/16. the readout stayed
perfectly healthy while the answers became worthless. that is the independence
of mass and correctness demonstrated a third time, now under a deliberate
intervention rather than by accident.

### truncation retry, confirmed necessary

the first cross-model run used a flat `n_probs: 64` and produced a label that
was simply absent from the returned list. the ladder now widens to 4096 and
then past the vocabulary, and stops early when the server returns fewer
entries than requested, which proves the whole distribution was seen. two of
the four models needed a retry on at least one case even though this build does
not cap `n_probs`.

### still open

carried into later phases, not blocking:

- flutter binding survey and plugin build approach (plan phase 0) is deferred
  until there is a desktop path worth binding. no flutter work was done.
- sequence copy support for the chosen architectures is a phase 4 question
  against the pinned `llama.h`; nothing here depends on it yet.
- larger models for phase 3 are available on this server (`qwen3.5-9b`,
  `qwen3.6-27b`). qwen3.5-4b from the plan is not present.
- llama.cpp is not pinned as a submodule yet; the build above is a remote
  server we do not control, so `b11028-972d2313b` is an observation, not a pin.
