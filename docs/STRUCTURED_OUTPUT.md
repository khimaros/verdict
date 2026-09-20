# structured output

whether constrained decoding, gbnf grammars or json schema output help verdict.

short answer: **not for the decision path, and one form of it would actively
break the project's main safety signal.** there is one job it is right for, and
it is the job verdict does not do.

## it is a no-op on our readout, measured

phase 0 ran the same prompt with and without a grammar restricting output to
the labels:

```
grammar: root ::= "A" | "B"

plain   top="\n\n"   labels {" B": 0.102096, " A": 0.08757}
grammar top="\n\n"   labels {" B": 0.102096, " A": 0.08757}
```

**byte-identical.** `completion_probabilities` are computed from the logits
before sampling, so a grammar changes what gets sampled and we discard what
gets sampled. it costs nothing and buys nothing.

## it is the alternative to verdict, not a part of it

| | verdict | structured output |
|---|---|---|
| forward work | one prefill, one decode step | one prefill, N decode steps |
| tokens produced | **zero** | one per output token |
| result | a distribution over declared options | one sampled answer |
| invalid output | impossible, only declared labels are read | impossible, the grammar forbids it |

both are structurally type safe, and for the same reason: neither can emit an
option that was not declared. the difference is cost and what you get back.
semif measured the same 21 binary criteria at 1.023 s read directly against
5.332 s generated as a json array, about 5x.

for several fields, structured output generates them in one pass. verdict asks
several questions against one shared prefix, which is cheaper and returns a
probability per field rather than a sample.

## the decisive argument: it would destroy option mass

the tempting version is to let a grammar restrict output to the labels and read
the post-sampling distribution, so the server renormalises for you. **do not.**

option mass is the raw probability the labels hold *before* renormalising, and
it is the only thing that catches a prompt which has stopped steering the
model. a grammar renormalises over the allowed tokens at every step, so the
answer always looks like a clean split between the labels.

what that would have hidden, both measured on this project:

- gemma-4-12b answering 9/10 correctly on an option mass of **1e-7**, because
  it was being fed another model's generation prompt and really wanted to emit
  `<|channel>`
- qwen3.5-9b scoring **10/10 on an option mass of 1.7e-08** with a broken
  assistant opening

under a grammar both of those look perfect. the signal exists only because the
distribution we read is unconstrained. `post_sampling_probs: true` has the same
problem and is already rejected for it in `docs/DECISIONS.md`.

## where it is the right tool

**free text.** browser-use's jev agent needs a string to type into a field, and
it calls a separate text model to write one. that is generation, verdict
declines to do it, and a json-schema-constrained generation is exactly the
right mechanism. if verdict ever grows a text field it should be constrained
output, clearly separated from the decision path and never reported as a
probability.

**possibly, multi-token option labels.** the cardinality ceiling is 52 because
that is how many single characters are single tokens. scoring options as
multi-token continuations, summing log-probabilities over each, would lift it,
and a grammar could enforce that only declared continuations are reachable.
that is the plan's stretch goal, it costs N decode steps per option, and it
should be measured against the tournament before being believed.

## what we do instead

nothing. the decision path sends `temperature: -1.0`, `post_sampling_probs:
false`, and no grammar, and reads the unconstrained next-token distribution.
