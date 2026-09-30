# jev wire protocol

notes for the optional jev-compatible server. verdict is independent and not
affiliated with or endorsed by typesafe. this describes a public api surface so
that an existing jev client can change its base url and nothing else.

sources, read 2026-09-19:

- <https://docs.typesafe.ai/api> (official api reference)
- <https://docs.aimlapi.com/api-references/decision-models/typesafe/jev> (mirror
  with a full worked request and response)
- <https://vercel.com/kb/guide/typesafe-jev-and-ai-sdk> (ai sdk mapping, which
  renames fields; useful for gating guidance, not for the wire shape)
- <https://github.com/dddanielliu/semif-serve> (prior art: a drop-in jev
  endpoint backed by semif)

## endpoints

| method | path | purpose |
|---|---|---|
| POST | `/v1/systemone` | the decision call |
| GET | `/v1/models` | served model plus `jev-latest` / `jev-preview` aliases |
| GET | `/health` | liveness |

headers: `content-type: application/json`, and `authorization: bearer <key>`
when a key is configured.

errors: 401 missing or invalid key, 422 body failed validation, 429 rate
limited, 529 overloaded and retryable.

## request

```json
{
  "model": "typesafe/jev",
  "state": "Help! My payments have been failing for 3 days and nobody answers support.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?"
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

`state` is a string, object or array. `questions` is a map from a caller chosen
name to a typed question. the three types are:

- `choice`: `criteria` is a map from option id to a description or null
- `noul`: yes/no. `criteria` is optional, `{"true": "...", "false": "..."}`
- `score`: `criteria` is an ordered array of level descriptions

note that the state is shared across every question in one request. that is
exactly the prefix/suffix split verdict already builds around, so a jev request
maps onto one prefill plus one suffix per question.

## response

```json
{
  "model": "typesafe/jev-1.13-20260917",
  "answers": {
    "is_urgent": {"type": "noul", "noul": 0.96},
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.97,
      "probabilities": {"billing": 0.98, "technical": 0.02, "sales": 0}
    },
    "frustration": {
      "type": "score",
      "score": 1.3,
      "confidence": 0.55,
      "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
      "probabilities": {"0": 0, "1": 0.7, "2": 0.3}
    }
  },
  "usage": {"input_tokens": 403, "output_tokens": 73}
}
```

`score` is the expected level, the probability weighted mean of the level
indices. `legend` echoes the rubric so a caller can render the answer without
holding the request. `probabilities` for a score are keyed by the level index
as a string.

## where this disagrees with our spec

three points to settle rather than paper over.

**confidence means something different.** the plan defines `confidence` as the
top probability. jev defines it as how concentrated the distribution is, "from
0 (spread evenly across options) to 1 (all on one option)", which is a
normalised entropy. these are different numbers and a client tuned against one
will mis-gate on the other.

decision: verdict's native result keeps the plan's `confidence` (top
probability) and adds `concentration` (normalised entropy,
`1 - H(p)/log(n)`). the jev adapter emits `concentration` as its `confidence`.
both are reported so neither client is silently wrong.

the score confidence is documented only as a "formula-based statistic" and the
exact formula is not public. we use normalised entropy over the levels and say
so. our score confidence is therefore not numerically comparable to jev's.

**`output_tokens` is nonzero in jev's example** (73), despite no sampling.
whatever it counts, we cannot reproduce it. verdict reports `0` and documents
that, matching what semif-serve does.

**jev calls the boolean type `noul`.** verdict's own api calls it `boolean`.
the adapter translates; the spec does not adopt `noul` as an internal name.

**questions are a named map, not a list.** the plan's fixtures use a list of
questions with ids. a map is the better shape, is what jev uses, and keeps the
adapter thin. the spec adopts the map.

## conformance target: browser-use/jev-ultrafast

a real jev client worth aiming at, because it points its base url at whatever
you tell it. source read 2026-09-19 from `jev_ultrafast/model.py`,
`jev_ultrafast/questions.py` and `.env.example`.

it posts to `{base}/v1/systemone` with `authorization: bearer <key>`, where the
base comes from `TYPESAFE_BASE_URL` or `TYPESAFE_API_URL`, and it sets
`TYPESAFE_MODEL=jev-latest`. so serving the alias is not cosmetic.

verdict's side of the pairing is `make serve`, which takes the backend, port
and key from the `.env` at the repo root. the client keeps its own `TYPESAFE_*`
names and changes only the base url.

what it requires of a response, and will raise `ValueError` without:

- `answers[name].choice`, a string
- `answers[name].probabilities`, numeric, each in 0-1, **summing near 1.0**
- `answers[name].confidence`, numeric, in 0-1
- top level `model` and `usage`

it retries on 429, 529 and 503 with backoff at 0.5s and 1s, three attempts,
on a 25 second timeout. our error codes must match those or the client will
fail fast instead of retrying.

not every client retries: jevbench never does. so the backend waits out a
model that is still loading itself, retrying the 502 from llama-swap and the
503 from llama-server with backoff for about 15 seconds before surfacing 529.

a 400 from the backend is the opposite case: llama-server refusing the prompt
as sent, usually a state longer than its context. that surfaces as 422 with
the backend's reason, because a client that retried would send the same
prompt into the same refusal.

two structural facts matter for us:

**it fans out speculatively.** one request carries `operation` plus several
`<operation>_target` questions over one shared state, and it uses only the
target matching the chosen operation. that is exactly one prefill and n
suffixes, which is the shape verdict is built around. it also means the
server's slot count sets the latency: on a `total_slots = 1` server the
suffixes serialise.

**option ids are element indices.** a page element table can carry dozens of
entries, and every one becomes an option. verdict's label alphabet is
currently `A`..`H`, eight labels. this use case needs far more, and the labels
must still be single tokens at the assistant position. **open issue**, see
`WORKING.md`.

### the text model is not a decision model

`.env.example` sets `TEXT_MODEL=inception/mercury-2.5` against
`https://openrouter.ai/api/v1`, and `model.py` posts it to
`{base}/chat/completions` and reads
`choices[0].message.content`. the comment in the file is explicit: "Required
for TYPE_TEXT."

so mercury is the free text generator for one operation, deciding what string
to type into a field. jev picks the operation and the element; mercury writes
the characters. it is doing the job verdict deliberately does not do.

mercury is a diffusion language model, and its advantage is generation
throughput. **that advantage does not transfer to verdict, which generates
nothing.** one forward pass, read the logits, stop. what verdict wants from a
backend is prefill speed and a well shaped next-token distribution, not fast
decoding.

running jev-ultrafast against verdict therefore still needs some text model for
TYPE_TEXT, and it can be any instruct model on the same llama-swap by pointing
`TEXT_MODEL_BASE_URL` at its `/v1`.

### telemetry must be off

browser-use ships anonymised telemetry on by default and announces it:

```
INFO [service] Using anonymized telemetry, see https://docs.browser-use.com/development/monitoring/telemetry.
```

disable it before any run:

```
ANONYMIZED_TELEMETRY=false
```

set in the environment or in the `.env` the run loads. the point of verdict is
that the decision never leaves the machine, and a run that phones home about it
defeats the exercise.

## where verdict is weaker than jev, and why

this section exists so that nobody reads our numbers as comparable to theirs.
**we have not measured jev.** everything attributed to it below is its own
public claim, and at least one source notes those claims are self-reported and
not independently reproduced. what follows compares our measurements against
their claims, which is not a like-for-like comparison.

### option cardinality: 52 against 255

| | ceiling | above it |
|---|---|---|
| jev | **255 options per choice** | two stage: score candidates independently, then choose. "slower", by their own account |
| verdict | **52** on ascii labels | the wide alphabet in spec 10.1, one exact pass, or the tournament in 10.2, approximate and slower |

sources for the 255 figure: truefoundry's writeup and flaviocopes' deep dive
agree on it independently.

the shape of the answer above the ceiling used to be the same on both sides:
both fell back to a two-stage scheme and both admitted it costs latency. with
the wide alphabet that is no longer true for us. labelling from the model's own
vocabulary reads 160 options at one position in one pass, so **161 options is
now an ordinary pass for verdict too** -- which is precisely the gap the hacker
news demo fell into.

**that closes the encoding gap and not the quality gap**, and the two are easy
to conflate. measured with list length held fixed, an unfamiliar label costs
qwen3.5-9b nothing in accuracy and costs minicpm5-2b 16/16 down to 9/16. we
have no comparable figure for jev, which does not publish one and which we have
not measured, so this is a statement about verdict alone.

the reason for the gap is architectural. our ceiling is 52 because that is how
many single characters are single tokens in a general instruct model's
vocabulary and still survive the assistant opening. it is a property of a
tokenizer we did not design, borrowed for a purpose it was not built for.
255 is one byte minus a reserved value, which is what an option index budget
looks like when it is designed rather than borrowed. typesafe does not disclose
the architecture, so treat that reading as inference, not fact.

### why our answers are worse

four distinct causes, worth separating because only some are fixable by us.

**1. model class.** jev is a purpose-built decision-only model. we read letter
logits off a general instruct model that was never trained to answer this way.
semif's ladder on its own authored decisions shows how much this dominates:
qwen3-0.6b 0.440, minicpm5-2b 0.686, qwen3.5-4b 0.813 balanced accuracy. the
technique does not rescue a model that cannot do the task.

**2. calibration is their training objective and we have none.** typesafe
describes training with "reinforcement learning for calibrated decisions",
optimising for probabilities being honest rather than for human preference.
we ship raw renormalised logits and say so. the measured consequence on our
side: gemma-4-e2b reports a top probability of 1.000 on answers that are wrong.
spec section 11 exists because of that measurement.

**3. option binding.** their options appear to live in an index space scored
against the state. ours are letters written into the prompt, and the model has
to hold the letter-to-description mapping by reading a list. that binding
weakens as the list grows: both models tested fell to 1 in 4 correct at 52
options while option mass stayed at 1.0.

**4. latency.** jev reports most calls at about 100 ms, in a 70-500 ms band.
our 161-option decision took 48 s. that is three orders of magnitude, and it
is mostly ours to fix rather than inherent: grouped scoring serialised eight
passes on a server with `total_slots = 1`. raising the slot count lets the
groups score in parallel.

### where we match

the type safety claim is identical in kind, and for the same reason. jev's "0%
type-error rate" is described as structural rather than empirical. ours is too:
an option id that was not declared cannot be returned, because only declared
label tokens are ever read. neither of us is being clever, and neither of us
can regress on it.

## what is deliberately not implemented

- no calibrated probabilities are claimed. jev advertises "calibrated
  confidence"; verdict's numbers are raw unless a calibration is fitted and
  named in the response provenance.
- no `jev-latest` behaviour guarantee. the aliases are served so clients do not
  break, and they resolve to whatever model the server was started with. the
  response `model` field always reports the real model.
