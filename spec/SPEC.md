# verdict specification

version `0.1.0`.

this is the source of truth. an implementation that disagrees with this
document or with `spec/fixtures/` is wrong, including the first one written.
if an implementation needs to deviate, change the spec and the fixtures first.

everything here is designed so that two implementations in different languages
produce **byte-identical prompts and identical token sequences** for the same
input.

## 1. the operation

given a state, a question and an ordered set of options, verdict runs one
forward pass and reads the probability the model assigns to each option's
label token at a single position. no token is sampled. no text is generated.

## 2. question types

three types. they exist to map one-to-one onto the jev wire protocol, so an
adapter is a rename rather than a translation. see `docs/JEV_API.md`.

questions are declared as a **named map**, not a list. the name is the caller's
key for the answer.

### 2.1 choice

```json
{"type": "choice",
 "instructions": "Which team should handle this?",
 "criteria": {"billing": "Payments, invoices, refunds",
              "technical": "Bugs, outages, integrations"}}
```

`criteria` maps option id to a description. a null or empty description means
the option id is used as its own description.

**a description may be a string, object or array**, and a non-string is
serialised by the rule in section 3. real clients use this: browser-use's jev
agent describes each candidate element as an object carrying its label, current
value and aria role. the same applies to `instructions`, which is a string,
object or array for the same reason.

**option order is declaration order.** json object key order is preserved by
every parser this targets, and the order reaches the prompt, so it affects the
result. this is not an implementation detail; see section 9 on position bias.

result: a probability per option id.

### 2.2 boolean

```json
{"type": "boolean",
 "instructions": "Does the sender threaten to cancel?",
 "criteria": {"true": "Yes.", "false": "No."}}
```

`criteria` is optional and defaults to `{"true": "Yes.", "false": "No."}`.
sugar over a two option choice with `true` first. result: `P(true)`.

the jev wire name for this type is `noul`. verdict does not adopt that name
internally; the adapter translates.

### 2.3 score

```json
{"type": "score",
 "instructions": "How urgent is this?",
 "criteria": ["Not urgent", "Should be handled soon", "Blocking, handle now"]}
```

`criteria` is an **ordered** array of level descriptions. levels are addressed
by their index as a string, `"0"`, `"1"`, `"2"`.

result: a probability per level index, plus `score`, the expected level:

```
score = sum over i of ( i * p_i )
```

and `legend`, the index to description map, echoed so a caller can render the
answer without holding the request.

### 2.4 a single option is legal

a question may declare exactly one option. real clients do this: browser-use's
jev agent builds one target question per operation, and on most pages only one
element supports a given operation.

the answer is that option at probability 1.0, with `confidence` and
`concentration` both 1.0, and the `single_option` flag set so that a caller can
see no choice was actually made. **option mass is still measured and still
meaningful**: it says whether the model was steered to the label at all, which
is the one thing worth knowing when there is nothing to choose between.

zero options is an error.

### 2.5 multi-label

expressed as several `boolean` questions over one state. there is no
multi-label type, and adding one would break the single-position readout.

## 3. serialisation

`state`, `instructions` and each option description are a string, object or
array. a string is used verbatim. anything else is serialised as json with
**sorted keys and ascii escaping**, in one of two forms:

| form | used for | layout |
|---|---|---|
| block | `state`, `instructions` | two space indent |
| inline | option descriptions | no indent, one line |

descriptions must be inline because the option list is one line per option,
and that column of labels is what the model is being asked to read. indenting
a description would spread one option across several lines.

sorted keys matter: the state is the cached prefix, and an unstable
serialisation silently destroys prefix reuse while still returning correct
answers, which is the hardest kind of performance bug to notice.

## 4. formatters

a formatter is a declarative table of affixes. it is **derived, never
hand-written**, from the model's own chat template.

### 4.1 where the template comes from

the gguf header, which is what llama.cpp actually loads. over http that is
`/props.chat_template`; on device it is the gguf the plugin already opened. a
base repo's `chat_template.jinja` or `tokenizer_config.json` is the wrong file:
gguf conversions get patched on the way through, and the patched copy is what
runs.

### 4.2 how it is derived

render the template with sentinel messages and diff the renders to recover each
affix. the assistant opening is the generation prompt rendered with
`add_generation_prompt=true, enable_thinking=false`.

formatters are **per model, not per family**. two models in one family
routinely ship different templates, and the difference is often exactly the
generation prompt. `gemma-4-12b-it` and `gemma-4-e2b-it` differ by a single
block that changes the opening, and each model reads about 1.0 option mass on
its own opening and about 1e-7 on the other's.

### 4.3 derivation happens at runtime

**the formatter is derived when a model is first used, not shipped per model.**

a pinned artifact can only cover models someone thought to pin, which fails the
case that matters most: a user pointing the library at an arbitrary gguf on
their own device. derivation is mechanical and needs nothing but the template
the model already carries, so there is no reason to require it in advance.

- derive on first use, keyed by the template's sha256
- cache the result, so it is once per model rather than once per process
- run the option mass verification at derivation time and cache it with the
  formatter
- a template whose sha256 is already cached skips straight to the cached table

the single-implementation rule from section 0 is what makes this safe. exactly
one jinja engine renders templates, inside the rust core; dart and python reach
it through the c abi rather than each shipping an engine of their own. two
engines would mean two sets of whitespace and filter semantics and therefore
two prompts, which is precisely the failure this spec exists to prevent.

the engine should be the one llama.cpp already carries, since the core links
llama.cpp regardless and that engine is what the serving path applies. adding a
second independent engine to render the same templates differently is a cost
with no benefit.

### 4.4 what `spec/formatters/` is for

pinned formatters are **golden regression fixtures, not the runtime source**.
they record what derivation produced for a known model on a known build, so
that a change to the engine, the derivation, or the template is visible as a
diff rather than as a silent change in behaviour.

conformance asserts that deriving a pinned model's formatter today reproduces
the pinned affixes byte for byte.

the python reference client is the exception: it is http only, does not link
the core, and renders with jinja2. that makes it a genuine second
implementation of derivation, which is useful precisely because the fixtures
catch it disagreeing.

### 4.5 artifact schema

the same shape whether it was pinned to `spec/formatters/<model>.json` or
derived at runtime and cached by template sha256:

```json
{
  "spec_version": "0.1.0",
  "model": "qwen3.5-0.8b:Q8_0",
  "model_alias": "unsloth/Qwen3.5-0.8B-MTP-GGUF:Q8_0",
  "template_sha256": "74a88f94c57c14e2...",
  "derived_from": "llama-server /props chat_template",
  "affixes": {
    "system_open": "<|im_start|>system\n",
    "system_close": "<|im_end|>\n",
    "user_open": "<|im_start|>user\n",
    "user_close": "<|im_end|>\n",
    "assistant_open": "<|im_start|>assistant\n<think>\n\n</think>\n\n"
  },
  "verification": {
    "build_info": "b11028-972d2313b",
    "cases": 10,
    "mean_option_mass": 0.9979,
    "min_option_mass": 0.9948,
    "split_clean": true,
    "labels_single_token": true
  }
}
```

this paragraph used to read "deriving happens once, offline; every runtime
loads the table", which is the design section 4.3 replaced and which
contradicted it for as long as both were in the file. derivation happens on
first use, keyed by the template's sha256 and cached; a pinned table is a
fixture to diff against, and an optional fast path, never the only way to
reach a model.

exactly one jinja engine renders templates per implementation, per section
4.3. the python reference client is the stated exception and renders with
jinja2, which is imported lazily so that a model whose formatter is already
cached or pinned never needs it.

## 5. prompt layout

designed so everything shared across questions comes first.

`system_open` carries the model's bos token when its template emits one, so
there is no separate bos affix to forget to prepend.

```
prefix = system_open + SYSTEM_INSTRUCTIONS + system_close
       + user_open + <serialised state> + STATE_DELIMITER

suffix = <instructions>
       + "\n" + <option lines>
       + ANSWER_INSTRUCTION
       + user_close + assistant_open
```

constants:

```
SYSTEM_INSTRUCTIONS = "You are a precise classifier. Read the material, then
                       answer the question with exactly one option label.
                       Reply with the label only."
                      (one line, no wrapping, exactly as in spec/constants.json)
STATE_DELIMITER     = "\n---\n\n"
ANSWER_INSTRUCTION  = "\n\nAnswer with one label."
```

each option renders as one line:

```
{label}) {description}
```

the prefix is prefilled once per state. each question is a short suffix.

### 5.1 the boundary must tokenise cleanly

prefix and suffix are tokenised **separately** and their token ids
concatenated. this is what keeps the cached prefix stable.

it is only safe where the boundary does not sit inside a token merge.
`STATE_DELIMITER` ends with a blank line and the suffix begins directly with
the instruction text, which was verified clean; the earlier candidate of a
delimiter ending `\n---\n` against a suffix starting `\n` mismatched in 18 of
18 cases.

this is tokenizer dependent, so conformance asserts per model:

```
tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)
```

a mismatch is a hard error, not a warning.

### 5.2 labels

labels are assigned to options in declaration order from:

```
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z   (26)
then a b c d e f g h i j k l m n o p q r s t u v w x y z   (26 more)
```

digits are not used. they are single tokens, but they add only ten and are
easily confused with numbers appearing in option text.

a label must be a single token **and** must remain its own token when appended
to `assistant_open`, which is the position being scored. both conditions are
verified per model at formatter derivation time and recorded in the artifact.

results are always mapped back to option ids. the option text is never scored.

## 6. probability extraction

1. request the next-token distribution at the position immediately after
   `assistant_open`, with sampling disabled and post-sampling probabilities off
2. locate each label **by token id**, never by token text: the same letter
   exists as `"A"` and `" A"` with different ids
3. `option_mass` is the sum of the raw probabilities on the label tokens
4. `probs` is those raw probabilities renormalised to sum to 1

### 6.1 truncation

a label absent from the returned candidates is a truncation artifact, not a
zero. widen the readout and retry:

```
n_probs ladder: 64, 4096, then past the vocabulary
```

stop early when the server returns fewer entries than requested, which proves
the whole distribution was seen. if a label is still absent at full vocabulary
its probability is genuinely negligible; record it as zero and set `truncated`
to false. if the ladder is exhausted without a full-vocabulary read, set
`truncated` to true.

treating a missing label as zero without retrying produces a renormalised
distribution over whichever labels happened to survive. it looks like a
confident answer and is not one.

## 7. the result object

per question:

| field | meaning |
|---|---|
| `probs` | renormalised probability per option id |
| `top` | option id with the highest probability |
| `confidence` | the top probability |
| `margin` | top probability minus second |
| `concentration` | `1 - H(p)/ln(n)`, 0 when uniform, 1 when all on one option |
| `option_mass` | raw probability on the label tokens before renormalising |
| `score`, `legend` | score questions only |
| `calibrated` | false unless a calibration was applied |
| `calibration_id` | set when `calibrated` is true |
| `truncated` | true when the readout could not be widened enough |
| `flags` | see section 8 |
| `timing` | prefill ms, score ms |
| `provenance` | model, build, template sha256, prompt sha256, spec version, backend, formatter id |

`confidence` and `concentration` are both reported because they are different
numbers and the jev protocol means the second one by `confidence`. reporting
one and calling it the other mis-gates any client tuned on the other.

`build` is per model, not per server: llama-swap runs a different llama-server
binary per model, and two models on one endpoint were observed on builds
`b11028` and `b11056`.

## 8. health gates

### 8.1 option mass

option mass is the correctness check for the mechanism, not a diagnostic.

- at **formatter derivation**, a mean option mass below `0.9` is a hard failure
  and the formatter is not written
- at **load**, a formatter whose recorded mass is below `0.9` is refused
- at **request**, `option_mass < 0.5` sets the `low_option_mass` flag

the floor can be set this aggressively because healthy and broken are six
orders of magnitude apart, about 1.0 against about 1e-7, with nothing observed
in between.

### 8.2 what option mass does not tell you

**option mass proves the prompt is steering the model to the labels. it proves
nothing about whether the model is right.**

a model was measured answering wrongly at an option mass of exactly 1.0000 and
a confidence of 1.000. mass and confidence are independent, and neither
substitutes for the other. this bound is repeated wherever the metric is
documented, because the metric is otherwise easy to oversell.

## 9. position bias and order averaging

option order reaches the prompt and biases the answer. a model was measured
answering the first-listed option correctly at every option count from 16 up
while missing most other placements.

`order_averaging: n` scores the question under `n` permutations of the option
order and averages the per-option probabilities.

- default is 1, meaning off, because the cost multiplies
- permutations are derived deterministically from the prompt hash, so a result
  is reproducible
- the first permutation is always declaration order
- `order_averaging` is recorded in provenance

## 10. long option lists

up to **26** options: one pass, labels `A`-`Z`. this is the supported path.

**27 to 52** options: one pass, labels extended into `a`-`z`, and the
`wide_alphabet` flag is set. the labels are verified single tokens, but mixed
case has not been quality tested.

**more than 52** options: rejected by default. the caller must shortlist, or
opt in to one of the two paths below.

### 10.1 the wide alphabet, opt-in only

the label alphabet does not have to stop at ascii. a model's vocabulary carries
thousands of single-character tokens, and asking the model which ones it has is
the argument that moved formatters to runtime in section 4.3: a table someone
wrote down covers only the models they thought to write down.

- candidates are **letters only**, the same rule `A`-`Za`-`z` already follows.
  a digit reads as part of the option text it labels and `)` is the option line
  delimiter, so neither is a label however singly it tokenises
- the pinned 52 come first, so a list of 52 or fewer options is labelled
  **byte-identically** whether or not a wide alphabet is available. every
  fixture in `spec/fixtures/` stays valid
- past those, order is token id ascending. the reason is **determinism**: it is
  a tokenizer property and cannot drift, while probabilities are not
  bit-reproducible under load, so an alphabet ordered by probability could
  differ between two derivations and build different prompts for one question.
  an implementation must not "improve" the order by filtering characters that
  look like ascii letters -- measured, that costs 16/16 down to 11/16 and
  option mass 0.7780 down to 0.3146
- a candidate must be one token **after the assistant opening**, not merely in
  isolation, since that is the position being read. candidates are checked in
  order and the walk stops once enough have passed
- results carry the `extended_alphabet` flag
- resolving the alphabet costs one whole-vocabulary read, so it happens on
  demand and is cached in the formatter entry beside the affixes
- a model whose vocabulary cannot supply enough labels is refused with the
  alternatives named, never served a short alphabet. where the caller has also
  enabled the tournament, that refusal becomes a fall back to it: they named
  the approximation as acceptable, and it is still flagged as one
- **a resolved alphabet is then VERIFIED by scoring with it**, exactly as a
  formatter is in section 4. an unambiguous question is labelled entirely from
  the wide alphabet and its option mass is read; below the request floor the
  alphabet is refused, or falls back to the tournament where one is enabled.
  every label being wide is the point -- a list long enough to reach the wide
  labels is also half ascii, so its mass would stay healthy on the familiar
  half and say nothing about the other

usability is **measured, never predicted from what the characters look like**.
two predictions were tried and both failed: excluding characters confusable
with ascii letters made the best model worse, and ranking by how readily the
model emits a character puts the failing alphabet above the passing one. the
property is cheap to measure directly and has so far resisted every proxy.

the result is an exact readout at one position, which is what the tournament is
not. the cost is that the labels are **verified single tokens and nothing
more**: whether a model reaches for an unfamiliar character as readily as `A`
is a quality question, measured in `docs/EVALS.md` rather than assumed here.

### 10.2 tournament, opt-in only

`tournament(options, group_size)` partitions the options, scores each group,
then runs a final pass among the group winners. the reported probability for an
option is its within-group probability multiplied by its group winner's
probability in the runoff.

results carry `approximate: true`. the number is a hierarchical approximation,
not a distribution read from one position, and it is sensitive to how the
options were partitioned. a caller has to ask for this by name; it is never a
silent fallback.

### 10.3 the honest warning

the label alphabet is not the binding constraint. both models tested held
option mass above 0.99 at 52 options while accuracy fell to 1 in 4. **a long
option list is a quality problem before it is an encoding problem.** shortlist
if you can.

this bears directly on 10.1: removing the encoding limit does not remove the
quality limit, and a caller who reads "exact readout" as "usable answer" has
been misled by a true statement.

## 11. calibration and gating

`calibrated` is false unless a calibration artifact was fitted on the caller's
own labelled data and applied. verdict never claims calibrated probabilities
out of the box.

`gate(result, threshold)` returns act or escalate. it **requires** either a
loaded calibration or an explicit `assume_uncalibrated` acknowledgement from
the caller.

this is deliberate friction, and it is there because of a measurement: one
model reports a top probability of 1.000 on answers that are wrong, while
another's confidence tracks its accuracy closely. a threshold fitted on the
second passes everything the first emits. raw confidence is not comparable
across models, and for some models a raw gate fires hardest exactly when the
model is wrong.

## 12. conformance fixtures

### 12.0 what is exact and what is not

**prompts are exact. probabilities are not.** the fixtures compare the
rendered prefix, suffix and split point, all of which are byte-exact and
required to be so by FR3. no fixture carries a probability, and none should.

a probability read from the same model for the same prompt is NOT reproducible
run to run, and this is a property of the serving stack rather than a defect
in an implementation. measured on llama.cpp `b11056`:

| | two identical passes over 16 prompts |
|---|---|
| quiet server | 0 of 16 differed |
| server under sustained load | 5 of 16 differed, up to 3.2e-02 |
| `cache_prompt: false` | 4 of 16 differed, 2e-03 to 7e-03 |

the cause is batch composition. `tools/server/server-context.cpp` packs pending
prompts into a shared batch while `cont_batching` is on:

```c
if (params_base.cont_batching || batch.size() == 0) {
    ...
    if (!add_ok || batch.size() >= n_batch) return;  // batch is full
```

so how a prompt is split into ubatches depends on what else the server is
doing at that moment, and matmul reduction order follows the shape. the same
file halves the batch under kv pressure and retries, which changes the split
again:

```c
// retry with half the batch size to try to find a free slot in the KV cache
if (!try_clear_idle_slots()) { n_batch /= 2; }
```

**a fixed seed does not address this and must not be used to try.** nothing is
sampled -- `n_predict: 1` with `post_sampling_probs: false` reads raw logprobs
and discards the sampler's choice -- so the rng never touches these numbers. a
seed that changed them would mean something is sampling, which would be a bug.

### 12.1 comparing probabilities

per NFR5, comparisons are probability-tolerant. `spec/constants.json` carries:

| constant | value | meaning |
|---|---|---|
| `probability_abs` | 0.05 | two probabilities agree within this |
| `stable_choice_margin` | 0.10 | below this margin the CHOICE may differ |
| `observed_max_drift` | 0.032 | largest difference measured |

NFR5 also says comparisons are argmax-exact, and that holds **only when the
margin between the top two options exceeds the drift.** a case measured at
0.487 against 0.432 is a 0.055 margin and flipped between runs. so an
implementation comparison asserts:

- the chosen option matches, WHEN `margin >= stable_choice_margin`
- otherwise, that both implementations put the same options in the top two
- probabilities within `probability_abs`
- option mass within `probability_abs`

a benchmark built on these numbers inherits the same floor: a difference of
one or two cases in sixteen is noise and must not be explained. see
`docs/EVALS.md`.

### 12.2 the fixtures

`spec/fixtures/*.json` carry inputs, the expected rendered prefix and suffix,
and the expected split point. token ids are model specific and live in
`spec/fixtures/tokens/<model>.json`, generated once and reviewed.

every implementation's test suite loads these. coverage required:

- all three question types
- string state, object state, array state
- 2, 3, 8, 26 and 52 options
- a long state of several thousand tokens
- an option with a null description
- a state whose serialisation would differ under unsorted keys
- a boundary case where naive joint tokenisation differs from the split
