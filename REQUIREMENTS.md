# requirements

drafted from the product owner's plan. this file is the acknowledged product
requirement set; the plan document remains the source for phasing and rationale.
changes here need product owner approval.

## what verdict is

verdict answers typed questions about a piece of text without generating any
text. one forward pass, read the probabilities the model assigns to each
declared option, renormalise over them.

given a state, a question and a set of options, it returns a probability
distribution over the option ids, plus the confidence signals needed to decide
whether to act on the answer or escalate.

## functional requirements

### FR1 question types

- `choice`: two or more options with an id and optional description. returns a
  distribution over option ids.
- `boolean`: sugar over a two-option choice. returns `P(true)`.
- `score`: an ordered rubric of levels. returns the distribution over levels
  and the expected level.
- multi-label classification is expressed as several `boolean` questions over
  one state. no separate type.

### FR2 no text generation

no free-text output, explanations, tool execution or chat. exactly one token
position is scored per question.

### FR3 identical prompts across implementations

the python and dart implementations MUST produce byte-identical prompts and
identical token sequences for the same input. the spec in `/spec` is the source
of truth and is enforced by shared golden fixtures.

### FR4 prefix reuse

the prompt layout MUST place everything shared across questions first, so that
a state is prefilled once and each question is a short suffix. the
prefix/suffix boundary MUST tokenise cleanly: separate tokenisation of prefix
and suffix MUST equal joint tokenisation, asserted per model.

### FR5 option mass is reported

every result reports the total raw probability that landed on the option label
tokens before renormalisation. this is the primary health signal. a low value
means the prompt is not steering the model to the labels and the distribution
should not be trusted.

### FR6 result contents

per question: `probs` by option id, `top`, `confidence`, `margin`,
`option_mass`, `expected` (score questions only), timing, and provenance
(model identity, prompt hash, spec version, backend, calibration id).

### FR7 two backends, one api

- a python reference client over http to a llama-server, which doubles as the
  evaluation harness
- a flutter library that loads a gguf on device and runs inference through
  llama.cpp via ffi, plus an http backend so the same dart code can target a
  server during development

### FR8 the library never bundles a model

the on-device backend takes a file path to a gguf already on the device.

### FR9 confidence gating

a `gate(result, threshold)` helper returning act or escalate. calibration is an
explicit optional step fitted on the user's own data, never implied.

## non-functional requirements

### NFR1 honest claims

calibrated probabilities are NOT claimed out of the box. documentation states
claim boundaries explicitly, including which model sizes are fit for which
kinds of question.

### NFR2 llama.cpp contact is isolated

all llama.cpp api contact lives in the c shim. the shim exposes a narrow stable
c surface so upstream churn does not reach dart.

### NFR3 reproducibility

llama.cpp commit, flutter and dart sdk versions, model revisions and file
hashes are pinned. raw evaluation output is committed alongside summaries.

### NFR4 third party data is fetched, not vendored

external datasets are pulled by script with checksums and recorded provenance,
unless a licence check says copying is fine.

### NFR5 numeric drift is expected

backends and batch shapes are compared argmax-exact and probability-tolerant,
with the tolerance documented.

## explicit non-goals

- no training or fine-tuning
- no free-text output, explanations, tool execution or chat
- no web target in the first version
- no claim of calibrated probabilities without a fitted calibration
