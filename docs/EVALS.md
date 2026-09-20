# evals

every measurement, what instrument produced it, and what it does not show.
raw output is under `eval/results/`.

results that turned out to be invalid are kept here rather than deleted, with
the reason. a result set with the failures removed is not a record of what was
measured.

## how much of this is noise

**the benchmark is not bit-reproducible, and every number below carries about
one case of slop.** same process, same served model, same 16 prompts, run
twice back to back: 5 of the 16 returned different probabilities (0.5655 vs
0.5976, 0.9381 vs 0.9448) and the total moved 10/16 to 11/16.

**the cause is not known, and the obvious suspect was tested and cleared.**
`cache_prompt: true` was the hypothesis -- a partly reused prompt prefilling
in a different batch shape -- and the measurement says the opposite:

| | two identical passes of 16 cases |
|---|---|
| `cache_prompt: true` | **0 of 16 differed** |
| `cache_prompt: false` | 4 of 16 differed, by 2e-3 to 7e-3 |

so reuse makes the readout MORE reproducible, not less, which makes sense in
hindsight: a reused prefix is prefilled once and read back, while a cold prompt
is re-split into batches every time.

**the cause is batch composition, found in the llama.cpp source.**
`tools/server/server-context.cpp` packs pending prompts into a shared batch
while `cont_batching` is on, so how a prompt is split into ubatches depends on
what else the server is doing at that instant, and matmul reduction order
follows shape. the same file halves `n_batch` under kv pressure and retries,
which changes the split again. both are load-dependent, which is why the drift
appeared on a server that had been under sweep load for hours and not on a
quiet one, and why `cache_prompt: true` is steadier -- it prefills a handful of
new tokens rather than re-chunking six thousand.

it is a property of the serving stack and the client cannot fix it. a server
run with `--no-cont-batching` and enough kv headroom to avoid the halving
should be reproducible; that has not been tested here.

the tolerance this implies is now normative: `spec/SPEC.md` section 12 and
`numeric_tolerance` in `spec/constants.json`, applied by `extract.agree()`.

the practical guidance does not change, and does not depend on the mechanism.

so:

- a difference of 1-2 cases out of 16 is NOISE. do not explain it.
- a difference of 4+ (order averaging on a biased model, 16/16 vs 10/16)
  is larger than the observed spread and worth reading.
- a POSITION COLUMN at 0/4 is not a reliable "never": g9v3-3b read first 0/4
  once and 1/4 on re-run, and the case that flipped was a near-tie, correct
  at 0.487 against the best distractor at 0.432.
- whole-run behaviour (a false completion, a run that never acts) is not a
  marginal probability and does not carry this caveat.

the phase 0b determinism claim -- cold and warm probabilities bit-identical for
the same prompt -- was measured on one small prompt inside one loaded instance
and should not be read as covering a varied workload.

**a fixed seed does not help and is the wrong instinct here.** nothing is
sampled: `n_predict: 1` with `post_sampling_probs: false` reads the raw model
logprobs at one position, and whatever token the sampler would have picked is
discarded. a seed governs the sampler's rng, which never touches those numbers.
what varies is floating point accumulation -- a different prefill batch shape
reduces the same matmuls in a different order -- and no rng is involved. if
setting a seed ever DID change the readout, that would mean something is
sampling where this design says nothing is, and it would be a bug rather than a
fix.

**this is a benchmark problem, not a conformance one.** all 29 fixtures compare
`prefix`, the rendered prompt text; none carries a probability. FR3 asks for
byte-identical prompts and identical token sequences and is untouched. NFR5
already names the mechanism -- "backends and BATCH SHAPES are compared
argmax-exact and probability-tolerant, with the tolerance documented" -- so the
requirement anticipated this. the gap is that the tolerance was never
documented, which is why the floor above had to be measured rather than cited.

## the instruments, cheapest first

| instrument | measures | needs | cost |
|---|---|---|---|
| `scripts/survey_option_mass.py` | can this model be read at all | a served model | ~1 min/model |
| `scripts/smoke_models.py` | decision quality, ground truth | a served model | ~1 min/model |
| `scripts/probe_wide_labels.py` | is a vocabulary label as usable as `A` | a served model | ~3 min/model |
| `scripts/eval_interventions.py` | does a debiasing step help | a served model | ~4 min/model |
| `demos/reliability.py` | does the agent loop complete a task | browser or device | ~3-5 min/run |
| `scripts/sweep_models.py` | the above, across models | browser or device | hours |

**run them in that order.** an agent run against a live site measures the
model, the harness, the network and a page that moves underneath it, and a bad
number does not say which. two model sweeps, about four hours, produced four
harness bugs and no ranking; the benchmark produced the ranking in ten minutes.
`eval_interventions.py` has said so in its own docstring since it was written:
*"this is the benchmark the browser demo cannot be, because a live site changes
underneath every run."*

## 1. can the model be read at all

option mass is the raw probability on the label tokens before renormalising.
below the floor the distribution is not trustworthy whatever the accuracy, and
the formatter is refused at fit time. see `DESIGN.md`.

| model | gib | option mass | note |
|---|---|---|---|
| gemma-4-e4b-it | 7.7 | 1.0000 | |
| gemma-4-e2b-it | 4.8 | 1.0000 | |
| granite-4.2-3b | 3.6 | 1.0000 | |
| g9v3-3b | 3.0 | 1.0000 | |
| minicpm5-2b | 2.5 | 1.0000 | |
| qwen3.5-9b | 9.1 | 0.9996 | |
| qwen3.5-4b | 4.3 | 0.9996 | |
| qwen3.5-0.8b | 0.8 | 0.9979 | |
| lfm2.5-2.6b | 2.9 | 0.9569 | only after its open think block is closed |
| minicpm5-1b | 1.1 | 0.9217 | closest any model came to the floor |
| gpt-oss-20b | 11.3 | 0.9909 | only with an explicit harmony opening |
| **functiongemma-270m-it** | 0.3 | **0.6936** | **REFUSED** |

two models needed their assistant opening repaired and one is refused outright.
both repairs are template faults rather than model deficiencies; see
`docs/DECISIONS.md`.

**a single two-option check is not sufficient.** lfm2.5 clears this at 0.9569
and collapses to 0.1535 by 52 options, which section 2 caught and this did not.

## 2. decision quality: element selection

16 cases per model: 4 option counts (8, 16, 26, 52) x 4 positions for the
correct answer, fixed distractors, no browser and no device. deterministic.

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
| g9v3-3b | 3.0 | 10-11/16 | 0.9390 | 390 | weakest FIRST, 0-1 of 4 |
| qwen3.5-0.8b | 0.8 | 7/16 | 0.9932 | 535 | weak MIDDLE, 0 of 4 |
| minicpm5-1b | 1.1 | 6/16 | 0.5200 | 327 | weak FIRST, 0 of 4 |

`qwen3.5-0.8b` scored 7/16, reproducing its phase 0b figure exactly across
months and a great deal of code change. that is what makes the rest of the
table worth reading.

**lfm2.5 scores 16/16 on a minimum option mass of 0.1535** -- perfect accuracy
on a readout that cannot be trusted, which is the case `DESIGN.md` argues for
the floor, now measured rather than hypothesised.

**for phase 8: minicpm5-2b.** 16/16 at 2.5 gib and 374 ms, the fastest model
that is also perfect. qwen3.5-4b matches the 9b at half the size.

## 2a. the wide alphabet: does an unfamiliar label still work

spec section 10.1 labels lists past 52 from the model's own vocabulary, so they
can be read at one position instead of bracketed into a tournament. the labels
are verified single tokens at the scored position. that is an encoding property
and says nothing about whether a model will REACH for one.

**asking this at 104 options would have answered nothing**, because length and
alphabet change together and section 10.3 already says a long list is a quality
problem by itself. so the question is put with LENGTH HELD FIXED: the same 16
cases from section 2, the same distractors, the same four positions, scored
twice, differing only in which characters label the options.

`scripts/probe_wide_labels.py`. `ascii` is `A`-`Za`-`z`; `wide` is non-ascii
single-character letters only, so every label is unfamiliar.

| model | ascii | mass | all labels unfamiliar | mass |
|---|---|---|---|---|
| qwen3.5-9b | 16/16 | 0.9996 | **16/16** | 0.7780 |
| minicpm5-2b | 16/16 | 0.9995 | **9/16** | 0.6991 |
| granite-4.2-3b | 15/16 | 0.9671 | **12/16** | **0.3754** |
| qwen3.5-2b | 14/16 | 0.9844 | **12/16** | **0.3577** |

**the cost is real, it is model-dependent, and only the largest model escapes
it.** qwen3.5-9b ranks identically under both alphabets. minicpm5-2b loses
nearly half its accuracy on questions it answers perfectly in ascii -- same
options, same order, same positions, only the label characters changed. and it
degrades with count: 4/4 at 8 options, 3/4 at 16, 2/4 at 26, 1/4 at 52.

**every model loses option mass and two fall below the request floor**, so
those answers arrive flagged `low_option_mass` rather than silently trusted.
accuracy alone would have reported granite at a respectable 12/16.

granite's ascii column reads 15/16 here against 14/16 in section 2. that is one
case of slop, which is the noise floor stated at the top of this file, and it
is why the alphabet comparison is run paired rather than against a number from
another session.

### the shipped path is not the stress arm, and the difference is large

the right column above is a deliberate worst case: **every** label unfamiliar.
that is not what the feature does. the pinned 52 come first, so a real 104
option question is labelled `A`-`Za`-`z` and then 52 vocabulary characters --
half of it familiar, and the correct answer is only sometimes in the tail.

scored that way, at 104 options:

| model | correct | min option mass |
|---|---|---|
| qwen3.5-9b | 4/4 | 0.9644 |
| granite-4.2-3b | 2/4 | 0.9856 |
| minicpm5-2b | 2/4 | 0.9262 |
| qwen3.5-2b | 1/4 | 0.9152 |

**option mass stays healthy on all four** -- 0.92 to 0.99, against 0.36 to 0.78
in the stress arm. so the readout does not collapse in the configuration
actually shipped. it collapses when EVERY option carries an unfamiliar label,
and on the shipped path that never happens: the first 52 labels are always
ascii, so the unfamiliar share is 0 at 52 options, half at 104, and approaches
one only for lists far longer than anything measured here.

the accuracy column is 4 cases per model and **confounded with list length**,
which section 10.3 warns about on its own. read it as "the 9b model still
answers a 104 option question and the small ones mostly do not", not as an
alphabet result.

**what none of this shows.** no model was tested whose vocabulary cannot supply
160 single-character letters. nothing here says what happens past 104 options,
where the familiar half runs out.

### it is not homoglyphs, and the obvious fix makes it worse

ordered by token id the alphabet begins `e`-acute, then cyrillic o, a, e, t --
and **cyrillic o is drawn identically to latin o while being a different
token.** a model shown `o) Site logo` cannot see which was meant, and answering
with the latin token it thinks it sees would put the mass outside every
declared label. that is the exact shape of the result above, mass falling while
accuracy holds, so it looked like the explanation.

it is not. a third arm drops every lookalike -- anything whose NFKD form is an
ascii letter, and anything latin, cyrillic or greek -- leaving 348 usable
labels of 600 on qwen3.5-9b. **it is worse on every axis**, on the one model
that had no accuracy problem to begin with:

| qwen3.5-9b | correct | min option mass | n_probs reached | median |
|---|---|---|---|---|
| ascii | 16/16 | 0.9996 | 4096 | 1033 ms |
| wide, lookalikes included | 16/16 | 0.7780 | 4096 | 2596 ms |
| **distinct, lookalikes dropped** | **11/16** | **0.3146** | **300000** | 2869 ms |

the obvious explanation was frequency: the lookalikes are the commonest
non-ascii characters a western-trained tokenizer has, so excluding them forces
the alphabet into arabic and hangul that the model is less willing to emit.
that was written here as a conclusion. **it was then measured, and it is
wrong.**

### frequency explains why ascii wins and nothing else

the whole-vocabulary readout taken during derivation carries a probability per
token at exactly the position a label is read from. that is each candidate's
base rate, and it was being discarded. `scripts/probe_label_rates.py` keeps it:

| group | n | min | median | geometric mean | scored |
|---|---|---|---|---|---|
| ascii | 52 | 1.46e-06 | 3.44e-05 | **4.21e-05** | 16/16, mass 0.9996 |
| wide, lookalikes in | 52 | 4.84e-09 | 1.01e-07 | 1.40e-07 | 16/16, mass 0.7780 |
| distinct, lookalikes out | 52 | 2.19e-09 | 2.83e-07 | **3.23e-07** | 11/16, mass 0.3146 |

**ascii is about 300x more emittable than either alternative**, which does
explain why it wins and why the pinned 52 belong first.

**it does not explain the rest.** `distinct` has a base rate roughly 2x HIGHER
than `wide` and performs much worse. base rate does not order those two, so a
usability floor built on it would not have flagged the group that actually
failed. that filter was designed and is not being built, because this is the
measurement that would have justified it.

what is left unexplained is why dropping lookalikes hurt. one untested
possibility: the filter removes the latin script entirely, not merely the
confusable characters, so an arabic-labelled row stops looking like a labelled
list in a western script at all and some of the labels are right-to-left. that
is a hypothesis with no measurement behind it and should be read as one.

**ascending token id stays, for the reason it was chosen: determinism.** it is
a tokenizer property and cannot drift, while probabilities are not
bit-reproducible under load (see the top of this file), so an alphabet ordered
by probability could differ between two derivations and produce different
prompts for the same question. the earlier claim here -- that id ordering is
also right for quality because it tracks frequency -- is withdrawn.

### so usability is measured, not predicted

three predictions of "will this model use these labels" have now been tried
from cheap properties, and all three failed:

| prediction | result |
|---|---|
| drop characters confusable with ascii | made the best model **worse**, 16/16 to 11/16 |
| rank by how readily the model emits a character | puts the failing alphabet **above** the passing one |
| option mass on one 26 option list | admits the model with the **worst** wide accuracy |

so the alphabet is verified the way a formatter is, by scoring with it. an
unambiguous question is labelled **entirely** from the wide alphabet at 8, 26
and 52 options, and both its option mass and its answer are read. every label
must be wide, because a list long enough to reach the wide labels is also half
ascii and its mass would stay healthy on the familiar half.

| model | worst wide mass | wide correct | probe accuracy | gate |
|---|---|---|---|---|
| qwen3.5-9b | 0.8976 | 3/3 | 16/16 | **served** |
| granite-4.2-3b | 0.8772 | 3/3 | 12/16 | **served** |
| minicpm5-2b | **0.9658** | **0/3** | **9/16** | **refused** |
| qwen3.5-2b | **0.3938** | 3/3 | 12/16 | **refused** |

**mass and accuracy come apart, and mass alone points the wrong way.**
minicpm5-2b holds the highest wide option mass of the four and answers the
question wrong at every count, having answered the same question 3/3 in
`A`-`Za`-`z`. so the gate is both conditions: mass above the request floor
**and** the verification answered correctly.

end to end on an 80 option question, the two models the gate admits both pick
the right element -- qwen3.5-9b at mass 0.9311 and granite at 0.9999 -- and the
two it refuses say why rather than returning a confident number.

qwen3.5-2b is refused on mass while scoring 12/16, the same as granite. that is
the intended behaviour and not a near miss: 60% of its probability lands
outside the declared labels, which is the condition option mass exists to
catch, whatever the ranking happens to do.

### the latency cost, which no accuracy or mass figure shows

the readout asks for the top 64 candidates and widens only when a label is
missing. a label the model would never have emitted is missing by construction,
so the ladder escalates:

| alphabet | n_probs reached | median decision |
|---|---|---|
| ascii | 4096 | 1033 ms |
| vocabulary labels | 4096 | 2596 ms |
| rare labels only | **300000**, the whole vocabulary | 2869 ms |

a 104 option question on the shipped mixed alphabet also reaches 300000, at
5112 ms median. **the wide alphabet roughly doubles decision latency before
the list is even long**, and that cost appears in no accuracy or option mass
number in this file.

## 3. order averaging

**it is not safe to apply by default.** measured as a paired run, baseline and
N=2 back to back in one session so server load could not confound them:

| model | baseline | N=2 | delta |
|---|---|---|---|
| qwen3.5-9b | 16/16 | 16/16 | 0 |
| gemma-4-e4b-it | 16/16 | 16/16 | 0 |
| qwen3.5-4b | 16/16 | 16/16 | 0 |
| minicpm5-2b | 16/16 | 16/16 | 0 |
| gemma-4-e2b-it | 10/16 | **14/16** | **+4** |
| qwen3.5-0.8b | 7/16 | 8/16 | +1 |
| granite-4.2-3b | 14/16 | **13/16** | **-1** |
| qwen3.5-2b | 14/16 | **12/16** | **-2** |

it helps only where a directional bias exists, does nothing at the ceiling,
and HURTS two models. granite-4.2-3b lost a case for the second time, which
makes that a reproducible cost rather than noise, and qwen3.5-2b lost two with
its 52-option column going 2 -> 0.

**so it is an intervention to apply where a bias is measured, never a default**
-- and the position profile from section 2 is how you know which models those
are. it also costs 2x latency, and the models that need it least are the ones
that can afford it.

### reversal maps first to last, which is its whole limitation

`qwen3.5-0.8b` at N=2 reads `[fir:4 qua:0 mid:0 las:4]` -- perfectly bimodal.
reversing a list swaps the two ENDPOINTS, so a model that prefers endpoints
keeps both of them and gains nothing in between. its +1 is that artifact
rather than an improvement, and the same geometry is why a middle-blind model
cannot be repaired by N=2 at all.

### the earlier single-arm numbers, and what N=4 adds

taken before the paired run above, on a busier server and against baselines
measured hours earlier, so read the paired table for the effect sizes and this
one for the mechanism. scoring each option list under N orders and averaging.
N=2 is forward and
reversed; N=4 adds seeded shuffles.

| model | baseline | N=2 | N=4 | the blind position |
|---|---|---|---|---|
| g9v3-3b | 10/16 | **14/16** | - | first 0 -> 3 |
| gemma-4-e2b-it | 10/16 | **14/16** | - | quarter 1 -> 3 |
| minicpm5-1b | 6/16 | 8/16 | 8/16 | first 0 -> 2 |
| qwen3.5-0.8b | 7/16 | 9/16 | 9/16 | middle 0 -> 0 -> **2** |
| granite-4.2-3b | 14/16 | 13/16 | - | - |
| qwen3.5-9b | 16/16 | 16/16 | - | - |
| minicpm5-2b | 16/16 | 16/16 | - | - |

the prediction was recorded before the run and held including its negative
half: reversal cancels a DIRECTIONAL bias and cannot touch a middle one. N=4
moved the middle and bought no accuracy -- 9/16 either way, with first 4->3,
last 4->3, middle 0->2. **the errors move; they do not go away.** on n=16 a
swing of +-2 is weak; the clean signal is a position column going 0 -> 3.

worth paying for trust rather than accuracy: calibrating a
position-contaminated distribution fits the contamination.

## 4. prior correction

measured 16/16 -> 3/16 on qwen3.5-9b. it divides out the model's answer to an
empty state, which is prejudice only when the option descriptions carry no
information. here the descriptions ARE the evidence, so it subtracts the
signal. off by default and should stay off.

## 5. agent loop: android, via mimic

11 models x 3 runs, emulator freshly booted, `pm clear com.android.settings`
before every run so no run inherits the answer screen. scored on the WITNESS --
text that actually appeared on a screen the agent reached.

| model | witnessed | claimed DONE | false completions | median ms |
|---|---|---|---|---|
| **qwen3.5-9b** | **2/3** | 2/3 | 0 | 6172 |
| g9v3-3b | 0/3 | **3/3** | **3** | 1570 |
| granite-4.2-3b | 0/3 | 2/3 | **2** | 1731 |
| minicpm5-2b | 0/3 | 1/3 | **1** | 1635 |
| the other seven | 0/3 | 0/3 | 0 | 1356-4583 |

**only qwen3.5-9b did the task.** scored on the agent's own status `g9v3-3b`
would rank FIRST -- three DONEs in one to three actions, having reached
nothing. six false completions across three models.

this is the strongest validation of ground-truth witnessing the project has:
without it the sweep would have recommended the model that does nothing and
says it has finished.

**decision quality and agent behaviour are different things.** `minicpm5-2b`
is 16/16 on section 2 and still produced a false completion here;
`gemma-4-e2b` is 10/16 and was the best browser performer. a model that can
choose correctly is not automatically a model that knows when to stop.

that holds, and it is not the whole story: on a goal it can actually finish,
the same `minicpm5-2b` is 3/3 with no false completions and knows it has
finished at 0.90-0.95. what defeats it is navigation depth, not stopping. see
"the confound, broken" below.

## 5a. android, with a verified reset

the sweep in section 5 ran with an incomplete reset and is superseded by this
one. `pm clear com.android.settings` clears the app under test and leaves
whatever else is open, so once a run tapped 'Phone' every later run started
inside the dialler with a popup focused, reading two rows. **the reset
verified its own action rather than the state it wanted**, which is this
project's recurring bug, and it is why `demos/mimic_reset.sh` now clears, goes
home, AND checks the screen before returning.

five models, three runs each, reset verified between every run:

| model | witnessed | false DONE | statuses |
|---|---|---|---|
| **qwen3.5-9b** | **2/3** | 0 | done, out-of-steps x2 |
| minicpm5-2b | 0/3 | 1 | out-of-steps x2, done |
| granite-4.2-3b | 0/3 | **3** | done x3 |
| qwen3.5-2b | 0/3 | 0 | out-of-steps x3 |
| qwen3.5-0.8b | 0/3 | 0 | blocked x3 |

the reset was worth it: `qwen3.5-9b` went from 1/3 with a false completion to
**2/3 with none**. the others did not improve, which is the useful half -- a
bad start state was hurting the model that could do the task and was not what
made the others fail.

### false DONE sits below true DONE, and the sample is confounded

the final decision of a run was the only one never recorded, because the loop
breaks before the history append. recording it, across both sweeps:

| | confidence |
|---|---|
| **true DONE** (witnessed) | 0.973, 0.934 |
| **false DONE** | 0.757, 0.716, 0.642, 0.575, 0.522 |

no overlap, and a gap of 0.177. **a DONE floor near 0.85 would have rejected
all five false completions and kept both true ones.** 0.85 is where this
reasoning first landed and it is **not** the number to use -- a later run
produced a true DONE at 0.8615, and the correction is two subsections down.

**but every true DONE came from qwen3.5-9b and every false one from a smaller
model**, so this could not separate "false completions are low confidence"
from "small models are low confidence about everything". under the second
reading a floor at 0.85 would also reject a small model's CORRECT completion,
trading false completions for false blocks.

so the floor went in as a **separate `--done-confidence`, defaulting to off**,
rather than by raising `--min-confidence`. those two thresholds answer
different questions -- is this the right target, versus is the task finished.

### the confound, broken

breaking it needed a true DONE from a small model, and the existing goals gave
none: minicpm5-2b and granite-4.2-3b were 0/3 witnessed on a goal that needs
navigation. so the goal was made single-screen instead -- "Open Settings and
find how much storage space is free", where one LAUNCH puts the answer on the
first screen and the `acted` rule still withholds DONE until something has been
done.

minicpm5-2b, three runs, reset and verified before each:

| | |
|---|---|
| task complete | **3/3** |
| witnessed | **3/3**, `Storage / 38% used - 9.96 GB free` |
| false completions | **0** |
| DONE confidence | **0.8993, 0.9387, 0.9543** |
| median decision | 1591 ms |

so a 2.5b model's TRUE completion sits at 0.90 to 0.95, inside the band the
9b model occupies and far above the false band. the separation survives:

| | n | range |
|---|---|---|
| **true DONE** | 5, two models | 0.899 - 0.973 |
| **false DONE** | 5, three models | 0.522 - 0.757 |

**the gap is 0.142 and no longer explained by model size.** the second reading
above -- that this only measures model size -- is ruled out.

### running the gate, and the threshold it corrected

the ten observations above came from runs taken WITHOUT the gate, so they only
show it would have fired. what the agent does with the steps it no longer
spends stopping is a different question, and needed its own sweep: the same
three models, the same goal and reset as 5a, `--done-confidence 0.85` the only
change.

| model | witnessed | false DONE | | witnessed | false DONE |
|---|---|---|---|---|---|
| | **baseline** | | | **gated** | |
| qwen3.5-9b | 2/3 | 0 | | 3/3 | 0 |
| minicpm5-2b | 0/3 | 1 | | 0/3 | **0** |
| granite-4.2-3b | 0/3 | **3** | | 0/3 | **0** |

**all four false completions are gone and no true DONE was blocked.** granite
stopped claiming success three times out of three and now runs out of steps
three times out of three.

**it bought trustworthiness, not capability, and the distinction matters.**
witnessed is unchanged on both weak models: a gate can stop a model lying about
finishing and cannot help it finish. qwen3.5-9b's 2/3 to 3/3 is not evidence of
improvement either -- n=3, and two runs finished in 4 actions where the
baseline took 40, which is device-state variance rather than anything the gate
could cause.

**the sweep also corrected the threshold.** it produced a true DONE at
**0.8615**, lower than any previously seen, which makes 0.85 the wrong number:
it clears that by 0.0115 against a drift of 0.032 (section "how much of this is
noise"). the same completion on a loaded server could have been blocked, which
is this gate's own failure mode inverted.

| | |
|---|---|
| highest false DONE | 0.7568 |
| lowest true DONE | 0.8615 |
| gap | 0.105 |
| drift to clear on each side | 0.032 |
| usable band | 0.789 to 0.8295 |
| **recommended** | **0.81** |

`test_the_threshold_clears_both_sides_by_more_than_the_drift` holds that,
so a future observation landing inside the band fails a test rather than
quietly outliving the claim.

it stays opt-in. twelve observations across two tasks is enough to make the
knob worth having with a number attached, and not enough to change a default
that every other measurement in this file was taken under.

## 5b. both agents, one model set

the same five models through both loops, three runs each. this is the only
like-for-like comparison of the two.

| model | params | element selection | browser stories | android witnessed | false DONE |
|---|---|---|---|---|---|
| qwen3.5-9b | 9.7b | 16/16 | **3/3 x3** | 1/3 | 1 |
| minicpm5-2b | 2.5b | 16/16 | **2/3 x3** | 0/3 | **3** |
| granite-4.2-3b | 3.7b | 14/16 | 0/3 | 0/3 | 1 |
| qwen3.5-2b | 2.3b | 14/16 | 0/3 | 0/3 | 0 |
| qwen3.5-0.8b | 0.9b | 7/16 | 0/3 | 0/3 | 0 |

**the benchmark gates both loops.** only the two 16/16 models reach anything
in either; nothing at 14/16 or below reaches anything in either. two cases out
of sixteen is the difference between a model that can drive an agent and one
that cannot.

**it does not tell you which loop a model can drive.** `minicpm5-2b` is second
only to the 9b on the browser and produces three false completions out of
three on android. deciding well and knowing when to stop are different
capabilities, and only the second is dangerous: a wrong choice costs a step,
a false completion ends the task while reporting success.

**android carries far more variance than the browser.** every browser model
was identical across its three runs; on android `qwen3.5-9b` went 2/3 in one
sweep and 1/3 in the next, because whether `Android version` sits above the
fold on arrival depends on device state. **three runs is enough to rank models
on the browser task and is not enough on android.**

### what the per-step record bought

recording `changed` and where each step landed turned four opaque rows into
named faults, and every one of them is an action-space defect rather than a
model one:

| model | what the record showed | the fix it implies |
|---|---|---|
| qwen3.5-9b | TAP 'Model' / BACK x4, all `changed=True` | `--retire-visited`, already written and off |
| minicpm5-2b | TAP a no-op then DONE, 1 step, x3 | DONE needs an action that CHANGED something |
| granite-4.2-3b | `mimic TAP: Path bounds must not be negative` | do not offer an element with negative bounds |
| granite-4.2-3b | LAUNCH 'Settings' twice, second `changed=False` | already covered by DeadEnds |

**the DONE gate is satisfied by a no-op.** it withholds DONE until `acted`,
and `acted` means an action was TAKEN rather than an action that did
anything. minicpm5-2b taps a search icon that changes nothing and is then
allowed to claim the goal is met. this was invisible until `changed` was
recorded per step.

## 6. agent loop: browser

### the intervention that worked

same task, same model, three runs each, before the sweeps below:

| | complete | stories/run | gated | status |
|---|---|---|---|---|
| baseline | 0/3 | mean 1.7 | 1, 1, 5 | blocked x3 |
| `--retire-read` | 3/3 | mean 3.0 | 0, 0, 0 | done x3 |
| `--retire-read`, later set | 2/3 | mean 2.7 | 0, 0, 2 | done x2 |
| `--trim-state` | 0/3 | mean 1.3 | 2, 1, 1 | blocked x3 |

`--retire-read` stops offering a link to a page already collected from. it is
the largest single intervention measured on either demo and it is an ACTION
SPACE change. `--trim-state` is 4.2x faster and loses the task, so it stays
off. see `docs/DEMOS.md`.

### the frontier sweep, valid

five models, three runs each, on a harness with four faults fixed and a
control in the run to prove it. scored on distinct `item?id=` pages actually
COLLECTED FROM -- see the scoring note below.

| model | params | real stories | actions | status | median ms |
|---|---|---|---|---|---|
| **qwen3.5-9b** | 9.7b | **3/3, 3/3, 3/3** | 7, 8, 5 | **done x3** | 21982 |
| **minicpm5-2b** | 2.5b | **2/3 x3** | 14 x3 | out-of-steps | **3761** |
| granite-4.2-3b | 3.7b | 0/3 x3 | 14 x3 | out-of-steps | 5628 |
| qwen3.5-2b | 2.3b | 0/3 x3 | 1 x3 | blocked | 3810 |
| qwen3.5-0.8b | 0.9b | 0/3 x3 | 0-1 | blocked | 3760 |

**qwen3.5-9b completed the task three times out of three**, the first reliable
completion this demo has produced. it is also the control: the harness works,
so every other row is a model result rather than a fault.

`minicpm5-2b` reaches two of three stories identically three times, at **a
quarter of the size and a sixth of the latency**. that is the pareto trade in
one line -- 3/3 at 9.7b and 22 s a step, or 2/3 at 2.5b and 3.8 s a step.

every model was internally consistent across its three runs, which is worth as
much as the ordering: no run-to-run randomness is hiding in these numbers.

### the score counts stories, not comment-shaped text

`qwen3.5-2b` first appeared to reach 1 story per run. it had clicked the
site-wide `comments` nav link, landed on `/newcomments`, and harvested five
bylines that satisfied the collect pattern. **the metric was counting pages
containing comment-shaped text rather than story pages**, which is this
project's recurring bug wearing yet another hat, this time in the scoring
rather than the agent.

the fix is offline, because the saved outcomes carry the urls: a story is a
distinct `item?id=` page. `browser_agent` should also take a `--collect-url`
pattern so collection only counts where the url matches, which no run so far
has had.

it is also arguably the GOAL being ambiguous rather than the model being
wrong. the instruction says "open a story comments" and the agent took a link
literally labelled `comments`.

### the model sweeps, both partly invalid

| sweep | runs | valid | what killed it |
|---|---|---|---|
| first | 33 | 16 | `settle()` raised while the page was navigating |
| second | 33 | 21 | `history.back()` raised while navigating |

both faults were in the harness and both were recorded as the agent failing to
act, which is indistinguishable from a model that cannot decide. the fix that
mattered was neither bug: it was recording `error` and `failed_operation` on
the outcome, without which the second cause would have been guesswork too.

the one result that survives the second sweep: **`gemma-4-e2b-it` completed the
task once (2/3, 2/3, 3/3), the only model to do so**, beating qwen3.5-9b at
1/3. one completion in three is weak evidence and hacker news moves underneath
every run, which is the whole reason section 2 exists.

`qwen3.5-4b`, `minicpm5-2b`, `g9v3-3b` and `minicpm5-1b` have NO valid browser
data -- all twelve of their runs died on `BACK`.

## what none of this measures

- **calibration.** confidence is not calibrated out of the box; gemma-4-e2b
  was measured reporting 1.000 on wrong answers. phase 7.
- **quantisation.** every number here is Q8_0. phase 8 runs a phone at about
  Q3, and no model on the eval server is served at two quants, so whether any
  of this is a model property or a rung property is unknown.
- **generation.** the element selection benchmark reads one token position. it
  says nothing about a model used to write text.
