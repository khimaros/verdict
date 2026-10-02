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
| `scripts/sweep_jevbench.sh` | decisions a third party wrote, through the jev endpoint | a served model | ~4-8 min/model |
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

the two models read through a head, on the same 16 cases. neither has an
option mass, since a head's distribution covers its options and nothing else:

| model | read as | correct | by option count (8, 16, 26, 52) | median ms |
|---|---|---|---|---|
| jev-omni | jev-omni (head) | 15/16 | 4, 4, 4, 3 | 742 |
| clef-flash | clef-flash (joint head) | 12/16 | 4, 4, 4, **0** | 8232 |

**both are right through 26 options and fall off at 52.** jev-omni's card
claims quality to 20 options. clef-flash's 16 cases are really 4: its author's
prompt lists options sorted by id, so the position of the right answer does
not reach the model and the four positions at each count return the same
probability to sixteen digits (0.951, 0.818, 0.778, then 0.291 and a miss at
52). read its row as 3 of 4, with no position bias to measure. its
verification question at 52 options passes, so the limit sits between an easy
question and this one and not in the reading. its 8 s is the transfer: every
position's hidden state crosses the vpn, since this script talks to the
backend directly.

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

## 5c. android, watched

the 5a goal and reset, three runs a model, through one verdict endpoint beside
the server. the difference from every sweep above is that the runs were
WATCHED: `scripts/sweep_models.py` now writes each step to
`/tmp/sweep-steps-<model>.log` as it happens, where a run's own output used to
be captured and dropped. nearly everything below was visible in that file
within a minute and in no summary at all.

### what the step log showed

first with the two models read through a head, then with the 5b set. each is
an action-space or bookkeeping fault, each has a test in
`tests/test_demo_gates.py`, and all but the last is on by default.

| seen in the log | what it was | the fix |
|---|---|---|
| `LAUNCH Settings` from inside Settings, every run, again after each scroll | a no-op is remembered per screen, and a scroll makes a new screen | the app already in front is not offered for LAUNCH |
| on the settings search screen TAP, SCROLL and BACK all "unchanged", then HOME / LAUNCH twenty times | with a keyboard up the first BACK only closes it; BACK was retired on one no-op | BACK gets a second try before it is retired |
| the search field tapped in every jev-omni run | the agent has no operation that types | a field with mimic's `edit` action is not a tap target |
| `no-elements` after tapping 'Security & privacy', twice | that window reads as `no active window` for as long as it is up | the agent goes BACK, then HOME, and only then gives up; the row that led there is used up |
| BLOCKED at 0.41-0.49 ending three runs with scrolling untried | BLOCKED ends a run as DONE does and nothing checked it | BLOCKED is held to DONE's floor |
| DONE at 0.45-0.59 with the answer below the fold | the gate of section 5a was opt-in | `--done-confidence` defaults to 0.81 here; 0 turns it off |
| granite DONE at 0.90-0.97 on a screen with TAP and LAUNCH withheld | removing options inflates DONE's share of what is left | a DONE reached with operations withheld is asked again with none withheld, and that answer is the one held to the floor |
| the same row tapped six times; scroll down / scroll up for thirty steps; LAUNCH Calendar / LAUNCH Settings ten times | every action changed the screen, so nothing was a no-op | a row tapped twice, or a scroll or launch made twice, from one screen is not offered a third time there; with every row used up TAP itself goes |
| a row tapped twice already, offered and taken twice more | the list came back with one more row in view and its signature no longer matched | a tap is remembered by the row's name on a screen sharing 60% of its rows |
| jev-omni: TAP at 0.85 whatever the target confidence, SCROLL_DOWN never above 0.17 | asked which row and THEN which operation, it tapped whatever the first question named | one question in place of two, below; the default since |
| every model: the About screen opens with the version below the fold and none scrolls down | the state lists the rows in view and says nothing of more | `--say-below`, below; opt-in, and it did not help overall |

### what that bought

witnessed is `Android version` actually on a screen the run reached; false
DONE is a run that ended `done` without it.

| model | before: witnessed | false DONE | fixed: witnessed | false DONE |
|---|---|---|---|---|
| qwen3.5-9b | 1/3 | 1 | 1/3 | 1 |
| minicpm5-2b | 0/3 | 3 | 0/3 | 0 |
| granite-4.2-3b | 0/3 | 1 | 0/3 | 0 |
| qwen3.5-2b | 0/3 | 0 | 0/3 | 0 |
| qwen3.5-0.8b | 0/3 | 0 | 0/3 | 0 |
| clef-flash | 1/3 | 2 | 1/3 | 0 |
| jev-omni | 0/3 | 1 | 0/3 | 0 |

before is `sweep-mimic-frontier` and `sweep-mimic-heads`; fixed is
`sweep-mimic-frontier-fixed` and `sweep-mimic-heads-fixed`, the two-question
agent with every default fix.

**false completions went from 8 runs of 21 to 1, and capability did not
move.** that is what section 5a found of the gate alone, now across seven
models: the fixes make a run trustworthy and do not make a model able. the one
left is qwen3.5-9b at 0.83, on the About screen with the answer below the
fold.

**the witnessed column is mostly luck at n=3.** whether `Android version` is
in view when About opens varies from run to run on a reset device. clef-flash
with the gate alone, before the later fixes, was witnessed 3/3 twice running
and then 1/3 on the final code, and the step logs show the gate behaving
identically in all nine: every DONE it blocked was on a screen without the
answer. the difference was which runs happened to open About with the row in
view. section 5b said three runs are not enough on this task; this is what
that looks like.

**no floor separates true from false across models.** with two questions and
every operation on offer, false DONEs here run 0.45 to 0.83 and true ones 0.81
to 0.99, and the overlap is between models: clef-flash is right at 0.81 to
0.93, qwen3.5-9b wrong at 0.83. one true DONE from clef-flash came in at 0.66
and the gate cost that run its ending. asked the one-question way jev-omni was
wrong at 0.815 and at 0.96. 0.81 is a default that removes most false
completions on most models, and section 7's calibration table is the reason
it cannot be more than that.

### one question instead of two

the agent now asks a single question whose options are every row, every app
and every move, and beside it, in the same request, whether to stop.
`--no-together` is the two-question agent, which asks which row is best and then which operation, telling
the second what the first chose. jev-omni cannot weigh "tapping would hit X,
confidence 0.4" against scrolling: it named the search row at 0.97 and then
tapped at 0.85, forty steps a run. asked once on the same screen, with "what
is needed is not among these rows" an option beside the rows, it put 0.988 on
scrolling.

| model | two questions: witnessed | median ms | one question: witnessed | median ms |
|---|---|---|---|---|
| qwen3.5-9b | 1/3 | 4758 | **3/3** | 3052 |
| clef-flash | 1/3 | 8010 | 2/3 | 4695 |
| jev-omni | 0/3 | 10300 | 1/3 | 5774 |

`sweep-mimic-heads-together`. ahead on all three, 6 of 9 against 2 of 9, at
about half the latency, with one false DONE (jev-omni, 0.815). the product
owner made it the default on that evidence, for both agents. it is three runs
a model on a noisy task, so the two-question form stays one flag away and the
"fixed" columns above, taken with two questions, are the comparison to rerun
when there are more runs.

how the stop is asked decided whether it worked at all, and two forms failed
before one held:

| the stop asked as | what happened |
|---|---|
| DONE, one option among the rows and moves | thinned out with the screen: true DONEs came in at 0.57 to 0.70 and never cleared the floor. clef-flash reached the answer and stopped in no run |
| a yes or no, "is the goal visible on this screen" | 0.90 on the settings list, where a row merely mentions the answer. three false DONEs in three runs, after three steps |
| DONE against the operations, in the same request | the form the floor was measured on. kept |

### saying that the screen continues

`--say-below` adds one line to the state when anything on screen scrolls:
more rows may be below, or, once a scroll down has done nothing here, that
this is the end. mimic reports that a container scrolls and not which way, so
that is all the harness can say.

| model | one question | one question, `--say-below` |
|---|---|---|
| clef-flash | 2/3 | 3/3 |
| jev-omni | 1/3 | 2/3, and a false DONE at 0.96 |
| qwen3.5-9b | 3/3 | **0/3** |

`sweep-mimic-hint-together`. it helps the two head models and breaks the one
model that was clean: qwen3.5-9b scrolled down, up, down, up on the settings
list, identically in all three runs, and never reached About. with two
questions it took the same model from 1/3 to 0/3 (`sweep-mimic-hint`, that one
model only). 5 of 9 against 6 of 9, so it is a flag and off.

### what is still wrong

- the About screen. it is the one failure every model shares and the hint
  did not fix it. a direction from mimic (`scroll_forward` against
  `scroll_backward`) would let the state say "the list continues below" as a
  fact and not a maybe.
- the unreadable window is mimic's to fix; the agent only routes around it.
- three runs a model. every table in this section needs more before a
  difference of one run is read as a difference.

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

### winnow-12b, with a control that did not reproduce

the frontier task and flags, three runs each, winnow-12b read with its own
layout and qwen3.5-9b as the control. scored on distinct `item?id=` pages.

| model | real stories | actions | status | median ms |
|---|---|---|---|---|
| winnow-12b | 4, 4, 6 | 14 x3 | out-of-steps x3 | 14832 |
| qwen3.5-9b (control) | 2, 2, 2 | 4 x3 | done x3 | 13256 |

**the control failed where it had completed 3/3**, declaring DONE after four
actions with two stories every time. hacker news and the harness both moved
since the frontier sweep, so this is not a clean model comparison. on the day:
winnow collected at least three real stories in every run and never stopped,
reading well past the goal until it ran out of steps; qwen3.5-9b stopped early
and was wrong about being done. winnow's runs also sent every prompt with the
doubled bos section 8 describes.

the same three runs again on current code -- pooled connections, a single
bos, checkpoint anchors, section 8:

| winnow-12b | median decision | mean | real stories | status |
|---|---|---|---|---|
| before | 14832 ms | 13950 ms | 4, 4, 6 | out-of-steps x3 |
| current | **9985 ms** | **9115 ms** | 0, 4, 3 | blocked, out-of-steps, blocked |

**a decision is a third faster.** the behaviour moved too, and is not
attributed: one run answered BLOCKED before acting, one oscillated between a
commenter's profile and the front page after four stories, one stopped
BLOCKED rather than DONE once it had three. the page had changed between the
two sets, three runs is a small sample, and of the three code changes only the
bos fix alters what the model reads -- section 7 measured winnow at 198
against 200 on jevbench with it, the same within noise.

### the two head models

the frontier task and flags (`--retire-read`), three runs each, through one
verdict endpoint beside the server that routes each request to its model.
`eval/results/sweep-browser-heads/`.

| model | read as | real stories | actions | status | median ms |
|---|---|---|---|---|---|
| **clef-flash** | joint head | **3, 3, 3** | 7 x3 | **done x3** | 42078 |
| **jev-omni** | head | **3, 4, 3** | 8, 14, 8 | done x2, out-of-steps | 75583 |

**both complete the task three times out of three**, which only qwen3.5-9b
had done on this demo. clef-flash is the cleanest run recorded here: the same
seven actions each time and DONE on the step the third story is read.
jev-omni reads its three stories in eight actions twice; in the third run it
does not stop, reads a fourth and wanders until the steps run out.

**they are the slowest models measured in this loop, for a reason that is not
the model.** a decision is 40 to 75 s against 22 s for qwen3.5-9b. an
embedding request reuses no prompt prefix, so every question of a step pays
for the whole page again, where a completion request pays for it once and
then for each question's tail. jev-omni asks one prompt per question and pays
three times a step; clef-flash asks all of a request's questions in one
prompt, which is why the larger head is the faster of the two.

**every run of both models spends one action on a commenter's profile.** after
the second story each clicks a username on the front page and comes straight
back. minicpm5-2b did the same in the frontier sweep. it costs two steps and
no run its completion, and it is an action-space fault: a profile link is
offered as a target for a goal that can never need one.

an earlier attempt at this sweep omitted `--retire-read` and is discarded:
clef-flash cycled between one story and the front page until the steps ran
out, twice, which is the behaviour that flag exists to stop.

### one question, and a stop the harness can check

section 5c's changes carried over where they apply: BLOCKED held to DONE's
floor, a DONE reached with operations withheld asked again with none, and
`--together`, one question whose options are every offered link and every
move with whether to stop asked beside it. three runs each, the same task.
one question and `--retire-read` are both the browser agent's defaults now, on
the evidence below and in "the intervention that worked"; `--no-together` and
`--no-retire-read` turn each off.

| model | questions | `--collect-pages` | real stories | actions | status | median ms |
|---|---|---|---|---|---|---|
| clef-flash | two | - | 3, 3, 3 | 7 x3 | done x3 | 42243 |
| clef-flash | one | - | 3, 3, 3 | 5 x3 | done x3 | **23928** |
| clef-flash | one | 3 | 3, 4, 4 | 5, 7, 7 | done x3 | 23214 |
| clef-flash | two | 3 | 3, 3, 3 | 8, 5, 8 | done x3 | 36801 |
| jev-omni | two | - | 3, 4, 3 | 8, 14, 8 | done x2, out-of-steps | 75993 |
| jev-omni | one | - | 7, 5, 6 | 14 x3 | out-of-steps x3 | 50364 |
| jev-omni | one | 3 | 3, 3, 7 | 6, 6, 14 | done x2, out-of-steps | **30814** |
| qwen3.5-9b | two | - | **2, 2, 2** | 3 x3 | done x3, all false | 10641 |
| qwen3.5-9b | one | - | **2, 2, 2** | 3 x3 | done x3, all false | 9381 |
| qwen3.5-9b | one | 3 | 3, 3, 3 | 5 x3 | done x3 | 8895 |
| qwen3.5-9b | two | 3 | 3, 3, 3 | 6 x3 | done x3 | 11886 |

`sweep-browser-heads`, `-heads-together`, `-heads-control`,
`-pages-together`, `-pages`.

**one question is about half the latency on the head models and loses
nothing.** clef-flash goes from seven actions at 42 s to five at 24 s, and
the wasted click on a commenter's profile is gone: with "go back, this page
has nothing more" an option beside the links, it stops choosing a link for
the sake of choosing one. on qwen3.5-9b, whose requests reuse their prefix,
the saving is small.

**no confidence floor can decide when this task is finished.** qwen3.5-9b
claims DONE after two stories of three, six runs of six in both question
forms, at 0.41 to 0.44. clef-flash is right to stop after three at 0.39 to
0.41. the android default of 0.81 would block both; nothing between them
divides them.

**the harness already knew.** it keeps a ledger of the pages it collected
comments from, which is the task's output. `--collect-pages 3` does not offer
DONE until that ledger holds three, the same rule as not offering DONE before
anything has been done. with it every run of all three models completes, in
both question forms that were run, and jev-omni, which never stopped with one
question, stops on the step after its third story in two runs of three. its
third run reads seven stories without stopping, which costs steps and not the
task.

that rule is a task parameter and not a trick: the goal says three stories,
the ledger counts them, and what is left to the model is the navigating. it
has no android counterpart, where the answer is one row and whether it is on
screen is the whole question.

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

## 7. jevbench's public items

jevbench's 231 public decisions (easy 48, original 72, hard 111), sent through
verdict's own `/v1/systemone` by jevbench's unchanged `typesafe` adapter, one
request at a time, on the eval server over the vpn. it is the one benchmark here that
verdict did not write: the items, the scoring and the client are a third
party's. raw per-item results are under `eval/results/jevbench*/`, and every
run is in `eval/results/export.json` with the weights it was measured on.

**public items only.** the board also scores sealed items verdict never sees,
and runs on datacenter gpus, so these numbers are not comparable one-to-one
with board rows. the hard tier is 111 items, so a difference of a few items is
noise, as section "how much of this is noise" says of every table here.

### each model read the way it was trained

| model | read as | correct | hard | hard ece | p50 |
|---|---|---|---|---|---|
| jev-omni | jev-omni (head) | 204/231 | 0.775 | 0.093 | 0.52 s |
| winnow-12b | winnow | 200/231 | 0.739 | 0.102 | 0.69 s |
| gemma-4-12b-it | chat | 200/231 | 0.757 | 0.199 | 0.74 s |
| jevk5-4b | semif | 199/231 | 0.739 | 0.096 | 1.01 s |
| decider-4b (v2.1) | decider-plain | 192/231 | 0.658 | 0.201 | 1.46 s |
| clef-flash | clef-flash (joint head) | 190/231 | 0.640 | 0.114 | 1.23 s |
| qwen3.5-4b | chat | 182/231 | 0.622 | 0.176 | 1.01 s |
| qwen3.5-9b | chat | 182/231 | 0.631 | 0.132 | 1.29 s |
| gemma-4-e4b-it | chat | 180/231 | 0.586 | 0.354 | 0.58 s |
| decider-2b (v11) | decider-plain | 176/231 | 0.577 | 0.221 | 0.66 s |
| standardone-8b | standardone-native | 173/231 | 0.522 | 0.196 | 0.45 s |
| gpt-oss-20b | chat | 161/231 | 0.513 | 0.320 | 0.68 s |
| granite-4.2-3b | chat | 157/231 | 0.451 | 0.446 | 0.42 s |
| gemma-4-e2b-it | chat | 153/231 | 0.396 | 0.548 | 0.48 s |
| qwen3.5-2b | chat | 149/231 | 0.460 | 0.229 | 0.64 s |
| minicpm5-2b | chat | 144/231 | 0.504 | 0.336 | 0.37 s |
| qwen3.5-0.8b | chat | 131/231 | 0.387 | 0.231 | 0.64 s |

**the layouts reproduce their authors.** decider-4b under `decider-plain`
scores 0.658 on the hard tier; its card reports 0.649 for v2.1 on the same
items, one item apart. that is the strongest check available that the layout
sends decider the bytes it was trained on.

**a model read through its decision head leads the table.** jev-omni does not
answer through its vocabulary: verdict takes the final hidden state of the
last prompt token from a llama-server in embedding mode and applies the
author's 256-way linear head to it (`spec/SPEC.md` 5.4). it scores 204/231,
the same 204 on three runs across two server builds, with the best hard tier
and the best hard-tier calibration measured here. the card reports 87.45%
micro on these 231 items for the bf16 torch model, which is 202, so the Q8_0
gguf read this way is within two items of its author. on the four decisions
the author publishes reference probabilities for, verdict's worst absolute
difference is 0.032, held by `tests_e2e/test_live_head.py`.

two things that number does not say. the row has **no option mass**: a head's
distribution covers the options and nothing else, so the health check every
other row passed does not exist for it, and it was admitted on answering the
verification question instead. and the author's interface takes plain option
strings, so `name: description` for a described option is verdict's choice;
the agreement with the card is the evidence that it is the right one. on
element selection it scores 15/16, missing at 52 options, where its card
claims quality only to 20.

**pooling `last` is the same vector as pooling `none`, and 15 times
cheaper.** the author's adapter asks the server for every position and keeps
the last. measured on one 73 token prompt, the last row is bit-identical
under both, and a warm request takes 270 ms against 4203 ms: the difference
is 74 rows of 3840 floats as json over the vpn, for one row used.

**clef-flash is read through its joint head, and its hard tier is UNVERIFIED.**
the head reads the final hidden state of every prompt position plus the
backbone's output embedding rows for each option's tokens, and decides all
questions of a request in one prompt (`spec/SPEC.md` 5.5). run through a
verdict beside the server, because every position travels as json: 76 kb a
token. it answers easy 48/48 and original 71/72 at about a second, and the
hard tier at 71/111 with a p50 of 12 s, those prompts being thousands of
tokens long.

that hard tier, 0.640, is level with its base model read as chat: qwen3.5-9b
scores 0.631. it is a different reading and not the same answers, since the
two predict alike on 75 of the 111, clef-flash wins 17 the base misses and
loses 16 the base gets (multi_hop 15/18 against 11, temporal_numeric 4/15
against 7). the gain over the base is on the two easier tiers, 119/120 against
112/120, and in calibration, hard ece 0.114 against 0.132. the card publishes no reference probabilities and no per-tier figure
for these items, so there is no number to hold 0.640 against. what IS checked:
the prompt is byte and span identical to the author's `encode_record` on four
records; llama-server tokenizes all 155 distinct pieces of 24 of these prompts
(17082 tokens) exactly as the author's tokenizer does; and the numpy port is
the author's head, at toy size in the unit tests and at full size by
`scripts/check_joint_head_parity.py`, which loads the published weights into
the author's torch class and feeds both the hidden states the live server
returns: 11 questions over 6 prompts up to 52 options and 1372 tokens, the
same answer on all, worst probability gap 1.2e-07. what is not: the Q8_0
gguf's hidden states against the bf16 torch backbone on a real record. until
that runs, read the row as the gguf through the author's head and not as the
author's model.

**reading a model in its own format pays where it was trained on one.**
jevk5-4b goes from 190 under chat to 199 under `semif`, winnow-12b from 196
to 200 under `winnow` with its ece falling from 0.157 to 0.102, and standardone-8b
cannot be read under chat at all -- its template's generation prompt is empty
and derivation refuses it. decider-2b reads the same either way, 176 against
175, so for it the layout is fidelity rather than a gain.

**four general instruct models decide as well as the dedicated ones.**
gemma-4-12b-it, qwen3.5-4b, qwen3.5-9b and gemma-4-e4b-it clear 0.75 overall
and 0.55 on the hard tier, the level of the weaker dedicated decision models,
and the model registry marks them as decision models on this evidence.

**confidence is not calibration.** gemma-4-12b-it ties for the most accurate
model and reports a mean top probability of 0.95 on the hard tier at 0.757
accuracy. winnow-12b, a decision fine-tune of the same gemma 4 12b, scores the
same 200 at ece 0.102, and jevk5-4b, whose author fitted a temperature, 0.096:
the two models whose confidence tracks their accuracy are the two trained for
it.

### calibration: can the confidence be acted on

the same runs, each model's default reading, all 231 items. confidence is the
top probability of an answer. generated into `README.md` by `make export` and
carried per run in the export (`ece`, `mean_confidence`, `overconfidence`,
`confident_share`, `confident_accuracy`, `confidence_auroc`).

| model | read as | accuracy | mean conf | over | ece | at 0.9+ | right there | auroc |
|---|---|---|---|---|---|---|---|---|
| jevk5-4b | semif | 0.862 | 0.875 | +0.013 | 0.030 | 65% | 0.960 | 0.848 |
| jev-omni | jev-omni | 0.883 | 0.913 | +0.030 | 0.042 | 77% | 0.972 | 0.931 |
| clef-flash | clef-flash | 0.823 | 0.843 | +0.020 | 0.059 | 61% | 0.979 | 0.885 |
| winnow-12b | winnow | 0.857 | 0.919 | +0.062 | 0.062 | 81% | 0.947 | 0.916 |
| standardone-8b | standardone-native | 0.775 | 0.802 | +0.027 | 0.064 | 42% | 0.990 | 0.890 |
| qwen3.5-9b | chat | 0.788 | 0.853 | +0.065 | 0.065 | 62% | 0.944 | 0.849 |
| qwen3.5-4b | chat | 0.788 | 0.848 | +0.060 | 0.076 | 56% | 0.930 | 0.805 |
| decider-4b | decider-plain | 0.831 | 0.903 | +0.072 | 0.085 | 72% | 0.940 | 0.861 |
| qwen3.5-0.8b | chat | 0.567 | 0.663 | +0.096 | 0.096 | 13% | 0.903 | 0.732 |
| decider-2b | decider-plain | 0.762 | 0.865 | +0.103 | 0.103 | 61% | 0.907 | 0.831 |
| gemma-4-12b-it | chat | 0.870 | 0.976 | +0.106 | 0.109 | 94% | 0.903 | 0.804 |
| qwen3.5-2b | chat | 0.645 | 0.780 | +0.135 | 0.148 | 45% | 0.874 | 0.790 |
| gemma-4-e4b-it | chat | 0.779 | 0.946 | +0.167 | 0.176 | 84% | 0.830 | 0.833 |
| gpt-oss-20b | chat | 0.697 | 0.890 | +0.193 | 0.207 | 67% | 0.838 | 0.778 |
| granite-4.2-3b | chat | 0.680 | 0.929 | +0.250 | 0.255 | 78% | 0.746 | 0.779 |
| minicpm5-2b | chat | 0.623 | 0.891 | +0.268 | 0.273 | 66% | 0.745 | 0.746 |
| gemma-4-e2b-it | chat | 0.675 | 0.965 | +0.289 | 0.289 | 89% | 0.717 | 0.769 |

`over` is mean confidence minus accuracy. `ece` is the expected calibration
error, ten equal-width bins, jevbench's rule. `at 0.9+` is the share of items
answered with a top probability of 0.9 or more and `right there` the accuracy
on those. `auroc` is the chance a right answer carries more confidence than a
wrong one, ties counting half.

**every model is overconfident, and by how much is the difference between
them.** the gap runs from +0.013 to +0.289. the five best calibrated are the
five decision models read their own way, and the three gemma-4 chat models,
granite and minicpm report 0.89 to 0.98 mean confidence whatever their
accuracy.

**calibration and ranking are different properties, and the table needs both.**
ece and the gap say how far the number is from the truth, which a temperature
fitted afterwards repairs. auroc says whether a lower number means a likelier
mistake, which nothing fitted afterwards repairs. gemma-4-e4b-it has the
fourth worst ece and an auroc of 0.833, level with decider-2b: its confidence
is badly scaled and still informative. qwen3.5-0.8b is the reverse case, a
middling ece of 0.096 and the worst ranking measured, 0.732.

**what a gate at 0.9 buys depends on the model far more than on the
threshold.** jev-omni answers 77% of the items that confidently and is right
on 0.972 of them. standardone-8b clears 0.99 and only on 42% of the work.
gemma-4-12b-it, the second most accurate model here, says 0.9 or more on 94%
of items and is right on 0.903 of those, so the gate removes almost nothing
and what passes it is three points better than no gate. gemma-4-e2b-it passes
89% of items at 0.717. a caller who gates on raw confidence gets a guarantee
from the first group and a false one from the last, which is why
`--min-confidence` has no default.

**what this does not measure.** the 231 items include 48 easy ones nearly
every model answers at high confidence, so the all-item figures flatter each
model against the hard-tier ece in the first table; both are in the export.
`right there` rests on as few as 30 items (qwen3.5-0.8b). nothing here is
calibrated after the fact: a temperature per model, fitted on held-out
decisions, is roadmap work and would move ece and the gap and leave auroc
where it is. and a head's distribution covers its options and nothing else,
so for jev-omni and clef-flash there is no option mass beside the confidence
to say the reading itself was sound.

### debiasing, per model

correct items / hard tier, with each option alone:

| model | as served | order averaging 2 | prior correction |
|---|---|---|---|
| gemma-4-12b-it | 200 / 0.76 | 200 / 0.75 | 193 / 0.68 |
| qwen3.5-4b | 182 / 0.62 | 184 / 0.65 | 174 / 0.54 |
| qwen3.5-9b | 182 / 0.63 | 182 / 0.64 | 179 / 0.59 |
| gemma-4-e4b-it | 180 / 0.59 | 178 / 0.57 | 168 / 0.50 |
| gpt-oss-20b | 161 / 0.51 | 161 / 0.50 | 170 / 0.54 |
| granite-4.2-3b | 157 / 0.45 | 157 / 0.44 | 139 / 0.33 |
| gemma-4-e2b-it | 153 / 0.40 | 154 / 0.41 | 152 / 0.41 |
| qwen3.5-2b | 149 / 0.46 | 149 / 0.47 | 145 / 0.41 |
| minicpm5-2b | 144 / 0.50 | 141 / 0.45 | 148 / 0.41 |
| qwen3.5-0.8b | 131 / 0.39 | 140 / 0.40 | 132 / 0.39 |

**order averaging leaves accuracy where it was and improves calibration on
every model**, hard-tier ece down by 0.02 to 0.10, for twice the scoring
passes. the one accuracy change past the noise floor is qwen3.5-0.8b, +9,
the model section 2 already found blind in the middle positions. use it where
confidence is acted on, or on a model with a known position bias.

**prior correction splits the models.** it costs six of the ten, worst on the
hard tier, where the answer to an empty state is not only prejudice:
granite-4.2-3b loses 18 items and gemma-4-e4b-it 12. it helps two past the
noise floor. gpt-oss-20b gains 9, and minicpm5-2b gains 4 by breaking a named
bias: it answers "other" to all twelve intent items at 0.97-1.00, order
averaging does not move that, and prior correction takes it from 2/12 to 7/12.
so it is a per-model setting, measured before it is turned on, and never a
default.

**tournament and the wide alphabet** cannot engage here: no jevbench item has
more than six options. both are recorded in the export's variants for when a
long-list benchmark runs; sections 2 and 2a have the evidence for them.

### what it found about the setup

- the eval server served the decision models at `--ctx-size 4096`, and standardone's
  native prompt overflowed it on three long-policy items. the run is kept
  under `eval/results/jevbench-invalid/ctx4096/`; the registry now sizes each
  model's context from its author, and the rerun at 8192 answers all 231.
- a verdict server reported a prompt that does not fit as `529 backend
  unavailable`, which a jev client retries. it now answers 422 with the
  backend's reason, since sending the same prompt again cannot fit it.

## 8. where a decision's time goes

measured on jevbench's 48 easy items, one request at a time, with
llama-server's own per-request `timings` to split the wall time. raw runs are
under `eval/results/perf-connection/`.

| minicpm5-2b | p50 | mean | min |
|---|---|---|---|
| a new connection per backend call | 0.346 s | 0.359 s | 0.322 s |
| a connection per server thread | 0.320 s | 0.358 s | 0.310 s |
| **a pool shared across threads** | **0.173 s** | **0.179 s** | **0.165 s** |

**a fresh tcp connection per call was half of a small model's decision.** it
costs a round trip before the request is sent, and the eval server is ~150 ms away. a
connection per thread did not help, because the jev server answers every
client connection on a new thread and clients open one per request; only a
pool shared across threads reuses anything.

| model | wall p50 | llama-server compute p50 | new tokens | prefix reused |
|---|---|---|---|---|
| minicpm5-2b | 170 ms | 13 ms | 1 | 114 |
| gemma-4-12b-it | 397 ms | 222 ms | 119 | 0 |
| qwen3.5-4b | 753 ms | 336 ms | 116 | 0 |

- **what remains for a small model is the network.** minicpm spends 13 ms
  computing and ~157 ms travelling. verdict beside its llama-server would
  answer in tens of milliseconds.
- **prefix reuse did not happen on sliding-window or recurrent models**
  (gemma-4, winnow, qwen3.5 and the models built on it), so the state was
  prefilled again for every question, which is the cost the prefix and suffix
  split exists to avoid. llama.cpp cannot roll such a cache back to an
  arbitrary position; it resumes from a checkpoint, and places one at the
  start of every user span it finds by matching `message_delimiters` against
  the prompt's tokens. verdict now sends one at the end of the shared state,
  chosen per model at derivation from the tokens that really end the rendered
  state (`---\n\n` on qwen, whose tokenizer folds whitespace into a following
  newline, `\n---\n\n` on gemma, `Question:` on decider, ` "criterion":` on
  semif). a follow-up question over a ~4.5k token page, after one with a
  ~1.5k token option list -- the browser agent's operation after its targets:

  | model | without the anchor | with it |
  |---|---|---|
  | winnow-12b | 4,025 new tokens, 4.98 s | 45, 1.04 s |
  | gemma-4-12b-it | 4,001, 5.10 s | 35, 0.72 s |
  | qwen3.5-4b | 3,997, 2.92 s | 33, 1.02 s |
  | decider-4b | 3,960, 2.82 s | 23, 1.03 s |
  | jevk5-4b | 4,017, 2.80 s | 48, 1.10 s |
  | standardone-8b | 16, 1.28 s | 16, 0.53 s |

  no server flag and no extra memory. standardone reused without it, its
  architecture being neither. after a short option list both cases reuse,
  because llama.cpp also checkpoints ~4 and ~516 tokens before every prompt's
  end; the anchor is what covers a list longer than that.
- **on a sliding-window model, reuse depends on what came before.** the
  eval server's own request log for winnow shows it exactly: after a question
  whose option list was ~1,560 tokens long, the next question over the same
  page re-prefilled all ~5,000 tokens; after one with ~100 tokens of options,
  the next reused 4,969 and prefilled 313. llama.cpp keeps only the last window
  of a cached prompt, so the shared state is reusable only while the previous
  prompt's tail after it fits the window. verdict now scores a request's
  questions shortest first, which makes every question after the first reuse
  the state when a client asks them together. it cannot help the browser
  agent, which sends targets and then the operation as two requests, so its
  long list is never the last prompt on a page; `--swa-full` on the server
  removes the condition for everyone.
- **gemma-family prompts reached the model with two bos tokens.** gemma's
  template renders `<bos>` into the system opening, and llama-server adds the
  model's bos to every text prompt, which its log flags as "the final prompt
  starts with 2 BOS tokens". derivation now records the bos the server adds
  and the decider drops it from the front of a text prompt. the six models it
  touched were run again on jevbench with a single bos
  (`eval/results/jevbench-single-bos/`):

  | model | doubled bos | single bos |
  |---|---|---|
  | standardone-8b | 173, hard 0.523, ece 0.196 | **179, hard 0.559, ece 0.151** |
  | gemma-4-12b-it | 200, hard 0.757 | 201, hard 0.766 |
  | winnow-12b | 200, hard 0.739 | 198, hard 0.721 |
  | gemma-4-e2b-it | 153, hard 0.396 | 156, hard 0.396 |
  | gemma-4-e4b-it | 180, hard 0.586 | 180, hard 0.586 |
  | minicpm5-2b | 144, hard 0.505 | 144, hard 0.505 |

  only standardone moved past the noise floor: its mistral-format prompt
  opens with `<s>`, and the doubled one cost it six items and some
  calibration. the gemma models and minicpm read the same either way. these
  reruns also came after the connection pool, so their latencies are lower
  for that reason and are not comparable with the rows above them.
- **qwen3.5-4b carries ~250 ms that neither compute nor the network explains.**
  its gguf is an MTP build served with a draft setting, and its vocabulary is
  248k pieces sorted for the top 64. not yet isolated.

## what none of this measures

- **calibration, fixed.** confidence is measured now, as hard-tier ece in
  section 7, and it is not calibrated out of the box: gemma-4-12b-it reports
  0.95 at 0.757 accuracy, and gemma-4-e2b was measured reporting 1.000 on wrong
  answers. fitting a calibration is phase 7.
- **quantisation.** every number here is Q8_0. phase 8 runs a phone at about
  Q3, and no model on the eval server is served at two quants, so whether any
  of this is a model property or a rung property is unknown.
- **generation.** the element selection benchmark reads one token position. it
  says nothing about a model used to write text.
