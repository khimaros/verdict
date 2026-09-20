# demos

concrete acceptance scenarios. each one is a real agent loop driven by verdict
answering typed questions, with no text generated for the decisions.

both share a shape: a changing state (a page, a screen), a fixed set of
operations, and a target chosen from whatever the state currently offers. that
is one prefill and several suffixes per step, which is what verdict is built
around.

## demo 1: hacker news

**task: browse to hacker news and fetch the top 5 comments from the top 3
stories.**

two harnesses run this task, and they answer different questions:

| | `demos/run_hn_demo.py` | `demos/browser_agent.py` |
|---|---|---|
| driver | browser-use/jev-ultrafast, unmodified | ours, cdp over one websocket |
| answers | is the wire protocol right? | can verdict drive an agent? |
| verdict | yes: a third-party jev client works unchanged | see below |
| action space | scroll up, scroll down, wait | + BACK, HOME, and page-wide targets |
| task outcome | 0/5 runs, no variance | measured below |

the first is the CONFORMANCE TEST and stays exactly as it is: an unmodified
third-party client pointed at verdict with `TYPESAFE_BASE_URL` and nothing else
changed is the only real evidence the protocol is right. see `docs/JEV_API.md`.

it is a poor capability demo for reasons that are properties of its harness
rather than of verdict, documented below, so the capability claim rests on the
second.

why this task is a good test rather than a friendly one:

- **it is multi-step and stateful.** three stories, each needing navigate in,
  read comments, navigate back. the agent has to know which stories it has
  already done, which is exactly the "don't repeat satisfied steps" criterion
  in the client's own NEXT_ACTION rubric.
- **hacker news has a large, flat element table.** a front page is dozens of
  links with near-identical shapes. this lands squarely on the option-count
  problem: the spec caps a single pass at 52 options and both models tested
  fell to 1-in-4 accuracy at 52 while option mass stayed above 0.99. **this
  demo will need shortlisting to work at all**, and it is the right place to
  find out what shortlisting has to do.
- **it distinguishes verdict's job from the text model's.** navigating and
  choosing stories is decisions. extracting comment text is not a decision, and
  verdict must not pretend otherwise. the comments come from the page, not from
  a model.

a `TEXT_MODEL` is still required for the client's TYPE_TEXT operation. point
`TEXT_MODEL_BASE_URL` at the same llama-swap `/v1` rather than openrouter.

### first run, 2026-09-19

`demos/run_hn_demo.py` against `qwen3.5-9b:Q8_0` on a one-slot llama-swap,
headed chrome over CDP. the run completed without error and **did not complete
the task**.

| decision | latency | options | outcome |
|---|---|---|---|
| 1 | 48.4 s | operation 5, click_target **161** | CLICK, opened a story's comments |
| 2 | 17.6 s | operation 6, click_target 41, type_text_target 1 | SCROLL_DOWN |
| 3 | 11.8 s | operation 6, click_target 43 | BLOCKED |

it navigated from the front page to `item?id=49763697` and then gave up. the
goal asked for the top 10 comments from the top 5 stories; it reached one story.

what the run established, which is worth more than a green tick:

- **the loop is real.** a local gguf model chose every action. nothing about
  the page or the goal left the machine, and both telemetry systems were off.
- **161 options on the front page**, against a 52 label ceiling. the
  tournament path was not a hypothetical; the demo cannot run without it.
- **option mass held throughout**, 0.9965 to 0.9990 including through grouped
  scoring. the encoding stayed healthy at every step.
- **the answers did not.** `operation` confidence was 0.373, 0.714 and 0.440.
  the model was unsure and correctly signalled it, then chose BLOCKED at 0.440.
- **`single_option` showed up immediately**, on a `type_text_target` with one
  candidate. verdict had been rejecting that as invalid; a real client sends it
  on any page where one element supports an operation.
- **latency does not fit the client's budget.** jev-ultrafast fixes a 25 s
  http timeout against a hosted service. the first decision took 48 s, so the
  demo only runs with that timeout raised. a deployment that matters has to fit
  25 s, and grouped scoring over 161 options on one slot does not.

the honest reading: the mechanism works and the agent does not. option mass
says the prompt is steering the model perfectly; confidence says the model does
not know what to do. those are exactly the two independent signals the spec
insists on reporting separately, and this run is why.

the next thing to try is shortlisting, not a bigger model. 161 candidates in
one decision is a bad decision at any size.

**later finding, and it does not change that conclusion.** the 52 label
ceiling has since lifted: spec 10.1 labels a list this long from the model's
own vocabulary and reads it in one exact pass, so 161 options no longer needs
the tournament. the ENCODING constraint was the one that moved. measured with
list length held fixed, unfamiliar labels take minicpm5-2b from 16/16 to 9/16
and drop granite's option mass to 0.3754, below the request floor. "a bad
decision at any size" still stands, and shortlisting is still the fix.

### second run: shortlisting

`demos/shortlist.py` ranks candidates by lexical overlap with the goal, keeps
the top 26 and preserves page order among the survivors. no model calls.

**it fixed the latency completely.**

| decision | no shortlist | shortlisted | options |
|---|---|---|---|
| 1 | 48.4 s | **17.5 s** | 161 -> 26 |
| 2 | 17.6 s | **11.1 s** | 42 -> 26 |
| 3 | 11.8 s | **8.6 s** | 38 -> 26 |

every decision now fits inside the 25 s budget the client allows, without
raising the timeout. the tournament path stopped being needed at all.

**it did not fix the agent.** the run still ended blocked. a later run with a
reworded goal got five decisions deep and four actions in, then wandered onto
hacker news's comment submission page and blocked there.

### shortlisting cannot be transparent behind the jev api

the first shortlisted run failed validation client-side. browser-use checks
`set(probabilities) == set(ids)` against **the full option set it sent**, and
rejects the entire response otherwise. a proxy that quietly drops candidates
breaks that contract.

so `restore_answers` puts the dropped ids back at probability `0.0`. that keeps
the contract, and it hides something: **zero here means "not considered", which
is indistinguishable in the response from "considered and found unlikely"**.
that is the strongest argument for shortlisting being the caller's explicit
decision rather than a server's quiet optimisation, and it is why
`demos/shortlist.py` is in `demos/` and not in the library.

### the real cause: the way home scrolls off the page

the missing BACK operation is worse than it first looks, because the one
substitute for it is not reliably present either.

`snapshot.js` excludes any element whose centre point falls outside the
viewport:

```js
if (!rname || r.width<=0 || r.height<=0 || x<0 || y<0
    || x>=innerWidth || y>=innerHeight) continue;
```

so the candidate list is what is on screen right now, not what is on the page.
captured from a real run:

| request | page | "Hacker News" among candidates |
|---|---|---|
| 002 | comments page, unscrolled | **yes** |
| 003 | same page, after one SCROLL_DOWN | **no** |

the agent is asked to read comments, which requires scrolling down, and
scrolling down deletes the only route back to the front page. `SCROLL_UP`
becomes available at the same moment, so recovery exists, but taking it means
planning two moves ahead: scroll up to reveal a link, then click it. a
decision-only model chooses one action from what it can see. it will not do
that, and it did not: it clicked a username instead and never came back.

this is not a scoring failure and no threshold or debiasing fixes it. the task
is unreachable from the action space the client offers, for any model.

### the action space has no way back

worth recording because it cost a run. jev-ultrafast's `snapshot.js` emits only
three non-element controls, `scroll_down`, `scroll_up` and `wait`, and
`choose()` adds `DONE` and `BLOCKED`. the complete vocabulary is:

```
CLICK  TYPE_TEXT  SCROLL_DOWN  SCROLL_UP  WAIT  DONE  BLOCKED
```

**no back, no new tab, no history navigation.** a goal saying "go back to the
front page" asks for an operation that does not exist, and the model answered
BLOCKED at 0.804 confidence, its most confident decision of that run. it was
right. the only way home is clicking the site logo, and the goal has to say so.

this is the client's action space, not verdict's. a real jev deployment hits
exactly the same wall, because the operation set is built by the client.

### where the time actually went

`scripts/profile_decision.py` replays a captured request and attributes every
round trip. running it first was worth more than any guess: **the model was not
the bottleneck, the client was.**

one decision, two questions, warm prefix cache throughout:

| | wall | `/tokenize` | `/completion` | round trips |
|---|---|---|---|---|
| as first written | 16889 ms | 11476 (68%) | 5412 | **31** |
| label ids pinned in the formatter | 6984 ms | 1533 (22%) | 5450 | 5 |
| plus state trimmed to the shortlist | 5077 ms | 949 (19%) | 4127 | 5 |
| plus prompts sent as text | **4302 ms** | **0** | 4301 | **2** |

**3.9x, with no change to the model, the prompt or the answers.** the client
now contributes nothing measurable: every millisecond left is the model.

26 of the original 31 round trips were one `/tokenize` call per label letter.
label token ids are a per-model constant, so they are now derived once and
pinned into `spec/formatters/<model>.json`. the first two rows return
byte-identical answers; that change is pure waste removal.

trimming the state to the elements a question can still choose cut it from 161
elements and 10.6k tokens to 26 and 3.2k. **it also improved the answers**:
`operation` confidence rose from 0.496 to 0.642 and the target from 0.781 to
0.932, and the chosen operation changed from SCROLL_DOWN to CLICK, which is the
more goal-appropriate move. 135 elements nobody could act on were costing
prefill and attention both.

the last three `/tokenize` calls went away by sending the prompt as text rather
than as separately tokenised halves. that is safe for exactly one reason: the
spec already requires `tokenize(prefix) + tokenize(suffix)` to equal
`tokenize(prefix + suffix)` and treats a mismatch as a hard error. where that
holds, the two forms are the same token sequence and the split buys nothing but
round trips. `Decider(pretokenize=True)` restores the explicit split, and
`tests_e2e/test_live_server.py` asserts the two agree against a live model, so
the day the assertion stops holding the suite says so rather than the
probabilities quietly moving.

a correction worth recording: an earlier note here claimed the native path
avoids these round trips in a way the http client could not. that was wrong.
the native path avoids them because it links llama.cpp and tokenises in
process, which is a property of being **co-located with the tokenizer**, not of
being rust. a rust client speaking to a remote llama-server would make exactly
the same calls python was making. the fix belonged here all along.

what is left is real work: 100% is now `/completion`, two calls that serialise
because the server has `total_slots = 1`. raising it is the next win and costs
nothing but a server flag.

### no, it does not need a planner

a reasonable guess is that the browser failures need a planning model, and that
jev-ultrafast already has one in `inception/mercury-2.5`. neither holds.

**mercury does not plan.** `field_text` is called from `agent.py` only when
`action["kind"] == "fill"`. it writes the string to type into a field. that is
the whole job.

**nothing plans.** `goal = "\n".join(plan)` concatenates the human's goal list
into one string that is passed unchanged to every decision, and `plan_index` is
only ever set to `int(selected == "DONE")`, so it is a done flag rather than a
cursor. the sole memory between steps is `history[-10:]` with four fields. the
architecture is deliberately plannerless: the human supplies the plan.

two hypotheses were tested against that, in order.

**hypothesis 1: it cannot track progress.** plausible, because the recorded
actions are labelled "178 comments", which does not say which story.
`demos/progress.py` accumulates visited story ids in the proxy and states them
in the shared prefix. it worked mechanically and reported 1/3 correctly. **the
agent still failed**, and differently: it clicked a username, then clicked
"comments" four times, and ended on `/newcomments`.

**hypothesis 2: it already knows it is lost and nobody reads the signal.**
this one held. target confidence across that run:

| decision | target confidence | outcome |
|---|---|---|
| 1 | **0.959** | clicked the right comments link |
| 5 | 0.464 | wandered |
| 6 | 0.269 | wandered |
| 7 | 0.378 | wandered |
| 8 | 0.385 | wandered |

option mass was 0.999 throughout, so the prompt was never the problem. and the
candidates on `/newcomments` were `upvote` x3, `parent` x3, `context` x2,
`0 minutes ago` x3. **no model can pick correctly from those. 0.27 is the
correct answer**, and a planner would not change it: no plan makes "upvote" the
right link.

`demos/gating.py` blocks a step whose chosen target falls below a threshold:

| | decisions | actions | outcome |
|---|---|---|---|
| ungated | 8 | 8 | wandered to `/newcomments`, **5 wrong actions** |
| gated at 0.6 | 4 | 3 | stopped on the user page, **0 wrong actions** |

**this buys trustworthiness, not completion.** it converts wrong actions into
no action. the task still does not finish, and saying otherwise would be
dishonest about what a threshold can do.

one failure survives the gate and is worth naming: decision 3 clicked the
username `prometheus1992` at **0.873** confidence. confidently wrong, so no
threshold catches it. that is a model-quality limit, and the fix is better
candidate descriptions or a better model, not a planner and not a threshold.

so the ranked answer to "what improves reliability here": gate on the
confidence that already exists, make candidate descriptions discriminative, and
keep the agent on pages where a correct answer is present at all. a second
model earns its place only for generated text, which is the one thing verdict
does not produce.

### reliability, over five runs

one run of an agent against a live site is an anecdote. `demos/reliability.py`
repeats it and reports the ground truth the task actually asks for: distinct
story pages reached, out of three.

qwen3.5-9b, shortlist 26, progress tracking on, gated at 0.6:

| run | stories | actions | decisions | gated | status | wall |
|---|---|---|---|---|---|---|
| 1 | 1/3 | 3 | 5 | 1 | blocked | 42 s |
| 2 | 1/3 | 3 | 4 | 1 | blocked | 25 s |
| 3 | 1/3 | 3 | 5 | 1 | blocked | 35 s |
| 4 | 1/3 | 3 | 4 | 1 | blocked | 33 s |
| 5 | 1/3 | 3 | 6 | 1 | blocked | 43 s |

```
task complete (3/3 stories): 0/5
reached at least one story:  5/5
stories per run:             mean 1.0, median 1, range 1-1
decision latency:            median 7704 ms over 24 decisions
```

**0 of 5, with no variance at all.** every run reached exactly one story, took
exactly three actions, fired the gate exactly once, and blocked.

that uniformity is the most useful part of the result. the readout is
deterministic, because logits are read and nothing is sampled, so a structural
wall produces an identical failure every time. this is not variance that a
better prompt or a few more samples would average away. it is the same wall in
the same place, five times, and it is the one described above: reading the
comments scrolls the only route home off the page.

a reliability number is worth having precisely because it distinguishes those
two cases. 1 of 5 would have meant a flaky agent worth tuning. 0 of 5 with zero
spread means stop tuning and change the action space.

**caveat on that table: `--target-first` was inert.** the flag was parsed,
documented and then dropped at the call site, so `demos/target_first.py` never
ran in the jev harness no matter what was passed. no jev-harness run has
measured target-first ordering, and the 0.132 -> 0.9955 figure quoted below
comes from the android agent, where the ordering is native rather than proxied.
fixed, with `python/tests/test_demo_gates.py` asserting the flag reaches the
proxy. a flag that is advertised and dropped is worse than a missing one: the
run reads as evidence the technique does not help.

### our own harness, over three runs

`demos/browser_agent.py` on the same task, qwen3.5-9b, gated at 0.5, scored on
distinct pages actually READ FROM rather than pages reached:

| run | stories | actions | decisions | gated | status | wall |
|---|---|---|---|---|---|---|
| 1 | 2/3 | 4 | 5 | 1 | blocked | 162 s |
| 2 | 1/3 | 2 | 3 | 1 | blocked | 63 s |
| 3 | 2/3 | 12 | 13 | 5 | blocked | 380 s |

```
task complete (3/3 stories): 0/3
reached at least one story:  3/3
stories per run:             mean 1.7, median 2, range 1-2
decision latency:            median 24702 ms over 21 decisions
```

still 0/3 complete, but the failure has a different shape from
jev-ultrafast's. that one was 0/5 with zero spread -- a wall. this one reaches
one or two stories, varies run to run, and always ends `blocked`, which means
the agent ran out of targets it was confident enough to click rather than
hitting something structurally unreachable.

run 3 says what is actually wrong. twelve actions, two distinct pages read:

```
CLICK '8 comments'    BACK    CLICK '231 comments'   CLICK 'Hacker News'
CLICK '698 comments'  SCROLL_DOWN   BACK   CLICK 'Hacker News'
CLICK '699 comments'  SCROLL_DOWN   BACK   BACK
```

`698 comments` and `699 comments` are the SAME story. the count ticked up
between the two visits, so the label changed while the destination did not --
which is why `--retire-read` compares resolved hrefs and not link text. the
agent has no way to know which stories it has already opened, and it does not
have to: the harness knows exactly which urls it collected from.

### `--retire-read`: 0/3 to 3/3

stop offering a link whose page has already been collected from. same task,
same model, same gate, three runs each:

| | complete | stories/run | actions | gated | status |
|---|---|---|---|---|---|
| baseline | **0/3** | mean 1.7, range 1-2 | 4, 2, 12 | 1, 1, 5 | blocked x3 |
| `--retire-read`, pretokenizing server | **3/3** | mean 3.0, range 3-3 | 5, 5, 5 | 0, 0, 0 | done x3 |
| `--retire-read`, text server | **2/3** | mean 2.7, range 2-3 | 5, 5, 6 | 0, 0, 2 | done x2, blocked |

**5 of 6 over both sets**, and the first set of three is NOT evidence of zero
variance -- that was claimed here on three runs and the next three broke it.
three runs distinguishes a wall from a flake; it does not establish a rate, and
a run of three identical outcomes is the most tempting moment to forget that.

the five successful runs took the identical path -- five actions, six
decisions, no gate fired:

```
CLICK '14 comments'  BACK  CLICK '231 comments'  CLICK 'Hacker News'  CLICK '702 comments'
```

and returned 15 comments, five from each of three distinct stories. across the
three runs the newest story's top comment differs (`abraxas`, `zem`,
`mauvehaus`) because hacker news moved underneath the run, which is what
confirms the text is read live rather than replayed.

the readout is deterministic, so every difference between runs comes from the
site. that is what makes a small sample worth anything at all: jev-ultrafast's
0/5 with zero spread is a wall, not a flake, and no amount of resampling was
going to move it. it is NOT licence to read three identical successes as a
rate -- done here, and wrong within three more runs.

this is the largest single intervention measured on either demo, and it is
again an ACTION SPACE change rather than a prompt one. the model was never
going to reason its way to "I have already read that story" from an action
history; it did not have to, because the harness already knew.

### where the time actually goes, measured rather than guessed

`scripts/count_upstream.py` sits between verdict and llama-server and records
every call: rung, prompt size, response size, duration, and how much of the
prompt it shares with the one before. a proxy rather than client
instrumentation, because the client is the thing under measurement.

one browser run, 7 steps, 22 upstream calls:

| | measured | what it rules out |
|---|---|---|
| calls per step | 3.1 | two jev requests, three scoring passes, as designed |
| rungs used | **64 only, all 22 calls** | the n_probs ladder never escalates |
| exact duplicate prompts | **0** | a response cache would hit nothing today |
| response size | median 32 KB | payload size is not a factor |
| prompt size | **median 23,122 chars, max 39,309** | this is the whole cost |

three predictions worth recording as wrong. the ladder was the prime suspect
and never fires. duplicate requests were the proposed cache and there are
none -- `recent_actions` and `progress` change every step, so a state is never
repeated. and `-np > 1` was called the biggest free lever in `WORKING.md`; it
does nothing alone, because the client issues sequentially.

what remains is prefill of a ~6k token prompt, three times a step. a hacker
news front page put **190 rows into the state where 26 were choosable**, and
`build_state` lists every one of them -- then the question criteria lists the
choosable 26 again.

`shortlist_state` already fixed this and was already used by
`demos/run_hn_demo.py`. `demos/browser_agent.py` never called it. worse, it
keys on `state.elements` and the browser state says `state.links`, so
importing it alone would have returned the body untouched and reported
success. this is the second time in this document a fix existed and the newer
harness did not pick it up; `--target-first` was the first.

`--trim-state`, one step, before and after:

| | prompt | per call | step |
|---|---|---|---|
| before | 23,030 c | 10,009 / 9,117 / 1,895 ms | 21,038 ms |
| after | 7,564 c | 3,689 / 2,868 / 1,706 ms | **8,270 ms** |

**-69% prompt, -68% per call**, and target confidence went UP, 0.95 to 0.99.

and the reliability a/b, three runs each with the navigation race below fixed
on both arms, says **do not turn it on**:

| | complete | stories/run | gated | decision latency |
|---|---|---|---|---|
| `--trim-state` | **0/3** | mean 1.3, range 1-2 | 2, 1, 1 | **5,051 ms** |
| no trimming | 2/3 | mean 2.7, range 2-3 | 0, 0, 2 | 21,174 ms |

**4.2x faster and it loses the task.** `shortlist_state`'s own docstring said
why before the measurement did: the discarded rows are still page context, and
a model that cannot see a row cannot reason about what surrounds the row it
picks. the gate counts say where it goes: trimming raises target confidence on
the FIRST decision, where the goal words match a row plainly, and costs it
later, once the agent is back on the front page choosing the next story from a
shortlist that no longer holds the useful candidates.

so the flag stays off. a 3x speedup that costs completions is not a speedup,
and the latency table is not allowed to decide this on its own.

the redundancy remains worth attacking, because it is free: link text is
listed once in the state and again in the question criteria. removing rows
loses context; removing the SECOND COPY of a row loses nothing. that is the
next thing to measure.

### labelling browser links with their row did not help

the android demo was fixed by labelling rows from their contained text, and the
browser's candidates looked like the same problem: a front page offers twenty
options reading `14 comments`, `238 comments`, `712 comments`, differing only
in a number. three runs blocked on the first decision with the page fully
loaded and the witness satisfied, which looked exactly like mass spreading over
indistinguishable options.

so each candidate now carries the row it sits in, verified against the live
page:

```
'16 comments'  in: '1. How Hacker News ranking works: scoring, controversy,
                    and penalties (2013) (righto.com) | 45 points by thean...'
```

**0/3, and every run now fails identically**: one story, two actions, blocked
at the third decision. the ambiguity hypothesis was wrong, or at least the
context does not resolve what the model is actually unsure about. kept,
because it costs little and the state already carried the text, but it is
recorded here as an intervention that did nothing rather than quietly dropped.

the failure that remains is specific and reproducible: the agent opens story
one and returns to the front page, and then cannot commit to story two. the
goal says "the top 3 stories" and nothing in the candidate set says which story
is second. that is an action-space problem, not a prompt one, and it is where
the next attempt should go.

### four harness faults, all recorded as the agent failing

a model sweep is a poor place to find these, because every one of them
produced a plausible-looking agent failure. in order of discovery:

| fault | looked like | actually |
|---|---|---|
| `settle()` raised while navigating | 17/33 runs "action-failed" | the condition settle exists to wait for |
| `history.back()` raised on its own navigation | 12/33 runs "action-failed" | the navigation had already happened |
| `close()` shared one try block | nothing at all | 13 leaked about:blank tabs |
| BACK reachable from the start page | 3 runs dead after 1 action | about:blank has no elements |

**the fix that mattered was none of them.** it was recording `error` and
`failed_operation` on the outcome. before that the summary said only THAT a
run ended in the harness, so a harness fault and a model that could not decide
were the same row, and the second cause would have been guesswork too.

two of these also show why a gate must be positional rather than temporal.
withholding BACK until the agent had acted was not enough -- measured, CLICK
then BACK then BACK walked front page, item, front page, about:blank. on the
start page there is nowhere useful behind us however we arrived, so the rule
is about WHERE we are.

### "finished with" is collected-from or came-back-from

`--retire-read` keyed its ledger on pages it had COLLECTED from, which meant a
page yielding nothing was never retired. measured on a 2.5b model:

```
CLICK 'RohanAdwankar' -> BACK -> CLICK 'RohanAdwankar' -> BACK   ... x7
```

a hacker news user page has no comments, so nothing was collected, so nothing
was retired -- and `DeadEnds` saw nothing either, because every one of those
actions genuinely changed the page. it is the same two-state oscillation the
android agent hit as TAP 'Model' / BACK, and `Exhausted` was already written
for it there and never ported.

the ledger is now the union: a url is finished with when it was collected from
OR navigated away from with BACK. that also repairs a quieter bug --
`--retire-read` did nothing at all unless `--collect` was also passed, because
the collect ledger was the only thing feeding it.

### the benchmark predicts WHETHER, not HOW

this section first said a benchmark score does not predict agent behaviour,
written from one observation: `minicpm5-2b` scores 16/16 and clicks a USERNAME
link at 0.95 confidence. the full sweep says that was too strong.

| element selection | reached a story on hacker news |
|---|---|
| qwen3.5-9b 16/16 | **3 of 3** |
| minicpm5-2b 16/16 | **2 of 3** |
| granite-4.2-3b 14/16 | 0 |
| qwen3.5-2b 14/16 | 0 |
| qwen3.5-0.8b 7/16 | 0 |

**a sharp threshold.** both 16/16 models reached stories; nothing below did.
two cases out of sixteen on the benchmark separates a model that can drive
this task from one that cannot, which is far more predictive than expected.

what the benchmark does NOT predict is HOW a model fails. the two 16/16 models
behave completely differently -- the 9b finishes in 5 to 8 actions, the 2.5b
wanders for 14 and reaches two of three -- and the failures below the
threshold are each their own shape: granite loops on a self-link, qwen3.5-2b
takes the site-wide `comments` nav link, qwen3.5-0.8b blocks before acting.

so the cheap gate earns its place as a GATE. it says who is worth running, and
it says nothing about what running them will look like.

### the speedup exposed a navigation race that was always there

the first trimmed a/b came back **0/3, zero actions, blocked on step one**,
which looked like trimming destroying the decision. it was not. the tell was in
the timing: 4.6 s per decision where a loaded page takes 8.1 s. a smaller
prompt means fewer elements, and fewer elements meant THE PAGE HAD NOT LOADED.

```python
def settle(self, tries=20, pause=0.3):
    if self.evaluate("document.readyState") in ("interactive", "complete"):
        return
```

`readyState` describes whatever is loaded right now. a fresh tab holds
`about:blank`, which is already `complete`, so `navigate()` returned before the
target page existed. this is the same bug as `settled_screen` on android and
the same bug as scoring a run on `am force-stop`: **a check that passes because
of what was already there.**

it does not crash. the agent finds a handful of elements, decides against them
at low confidence and blocks, which reads as a model failure. it had presumably
been losing this race occasionally all along; trimming made the agent fast
enough to lose it every time. `settle` now waits for the document it navigated
to, and the a/b was re-run against the fix.

### the latency figures above were measured against a stale server

the verdict server serving every run in this document started at 11:50 and
`decide.py` was last edited at 12:18. a python process holds the code it
imported, so it went on sending PRETOKENIZED prompts for four hours after the
source on disk had switched to text -- visible only as a token array in the
llama-swap request log, with nothing anywhere saying which form was in use.

so **the per-step numbers here include a `/tokenize` round trip the current
code does not make.** the ordering they support is unaffected, because every
arm ran against the same server; the absolute milliseconds are an upper bound
on what current code costs.

measured directly against the same llama-swap, one scoring pass, repeated
state so the prompt cache hits:

| | median | option mass | answer |
|---|---|---|---|
| text | 610 ms | 0.9992 | BLOCKED |
| pretokenized | 890 ms | 0.9992 | BLOCKED |

identical answer and identical option mass, which is the boundary assertion
holding: the two forms are the same token sequence. text is what the serving
path uses, because it costs no round trip and because a request is legible in
a proxy log, which is where these get debugged. `pretokenize=True` remains as
a verification switch and `python/tests/test_wire_format.py` asserts nothing
in the serving path turns it on.

the server now prints its wire format and load time at startup:

```
prompts   text, one /completion per scoring pass (loaded 15:47:16)
```

### how the comment data comes back

for a long time the honest answer was **it did not**. the hacker news task was
scored on `stories_visited` -- pages *reached* -- and nothing in either harness
ever read a comment. jev-ultrafast had the text the whole time and never looked
at it: its own state carried 3,382 characters of comment body per page, unused.

`demos/browser_agent.py --collect REGEX` closes it, on the same division of
labour as the android demo. the agent decides which page to open; the comments
are then plain text sitting in that page, and a regex reads them.

```
--collect '^\s*[a-z0-9_-]{2,15} [0-9]+ (?:minute|hour|day)s? ago \|[^\n]*\n+[^\n]+'
```

three things about that pattern were each a wrong run:

- **it is written against `document.body.innerText`**, not the html and not
  what the page looks like. hacker news puts a byline and its comment on two
  lines with blank lines between, so the pattern crosses a newline and the
  match is squeezed back to one line. `scripts/page_text.py URL` dumps exactly
  what the agent sees, which is how to write one of these without guessing.
- **`^` must mean the start of a line.** `re.finditer` without `MULTILINE`
  anchors at position zero, so the first version returned nothing -- and a
  `--collect` that silently returns nothing reads as "the page had no
  comments" rather than as a broken pattern. the run that found this reported
  `witnessed 2/2` and collected 0.
- **matches are kept per url.** "five comments from each of three stories" is
  three pages of five, and flattening them loses which story each came from.

that per-url ledger doubles as the progress the decider is told about, and it
is true by construction because it is what was read off a page rather than what
a step claimed:

```
collected 5 from https://news.ycombinator.com/item?id=49765348
  1. johnfn 3 hours ago | next It's a tale as old as time -- people don't ...
  2. calebkaiser 1 hour ago | parent | next This is also a really common ...
collected 5 from https://news.ycombinator.com/item?id=49764791
  1. ajjenkins 10 hours ago | next A lot of people in the comments say ...
```

### DONE has to clear the same bar as a decision to act

the first three-story run with collection working read two stories and stopped,
on a DONE held at **0.32**. the model was not claiming the goal was met; it was
merely less unsure of DONE than of anything else, and `--min-confidence` only
gated operations that needed a target.

the android agent had already been fixed for this and the browser agent had
not, which is the whole hazard of porting a lesson by hand. both now fall back
to the best non-DONE operation when DONE sits below the floor, and
`python/tests/test_demo_gates.py` holds the two of them to it together.

### a live site is not a safe playground

in the reworded run the agent clicked "add comment" and landed on hacker news's
comment submission page. nothing was posted, because the browser profile is not
logged in, and that is the only reason.

an agent with click authority on a live public site can take public,
irreversible actions. the demo profile must stay logged out, and anything
beyond a demo needs an allowlist of permitted targets rather than trust in the
model's judgement. a decision model that reports 0.4 confidence and acts anyway
is not a safety mechanism.

**set `ANONYMIZED_TELEMETRY=false` before any run.** browser-use ships
anonymised telemetry on by default. a local decision that phones home is not a
local decision.

## demo 2: android, via mimic

**an on-device agent loop that never leaves the phone.**

`../mimic` is an android accessibility service that already exposes the two
halves this needs:

- **view**: the active window's accessibility tree as json, with per-node
  bounds, tap coordinates and supported actions, filterable **on the device**
  (interactive-only, text-only, by text/id/class/desc, regex)
- **interact**: tap, long-press, swipe, click or set-text on a node, scroll
  until a node appears, back / home / recents / notifications, launch an app

over three token-gated surfaces on localhost: intents, http, and mcp.

this is a better fit for verdict than the browser case, for three reasons.

**the element table already exists and is already typed.** mimic hands back
nodes with their supported actions attached, so the operation question and the
target question can both be constrained to what the screen actually offers,
rather than inferred from rendered html.

**mimic already solves the problem the spec flags.** its on-device filtering
exists "so an agent sends and receives the minimum context", which is precisely
the shortlisting that a 52-option ceiling demands. the interactive-only filter
is the shortlist.

**the whole loop can be local.** mimic binds to `127.0.0.1`, and verdict's
flutter plugin runs the gguf on the same device. that closes the loop with no
network, no api key and nothing to disable: the phone decides what to tap using
a model on the phone. the browser demo still reaches a server; this one need
not. that is the strongest argument this project has, and it should be the
demo that gets built carefully.

### shape of the loop

per step, one prefill of the current screen and three suffixes:

```
state      the filtered accessibility tree, serialised per spec section 3
operation  choice over TAP, SET_TEXT, SCROLL, BACK, HOME, DONE, BLOCKED
target     choice over the node indices that support the chosen operation
done       boolean, is the goal satisfied by what is on screen now
```

the operation and target questions are the speculative fan-out the jev clients
use: ask both, keep the target matching the chosen operation.

### it works, 2026-09-19

`demos/mimic_agent.py` against a pixel 8 pro, qwen3.5-9b deciding and planning,
goal "open the Settings app and navigate to the screen showing the Android
version":

```
 1  TAP  conf=0.99  'About phone / ARES'
 2  SCROLL_DOWN
 3  TAP  conf=0.98  'About phone / ARES'    -> planned: 'TAP About phone / ARES'
 4  SCROLL_DOWN
 5  TAP  conf=0.61  'Android version / 17'
 6  TAP  conf=0.94  'Android version / 17'  -> screen unchanged; TAP retired
 7  DONE conf=0.90

status done after 6 steps, 3 planning calls
```

it reached the right screen and stopped. the browser demo never finished its
task in ten attempts across two configurations.

**none of what fixed it was model tuning.** in the order the failures appeared:

| change | effect |
|---|---|
| offer LAUNCH | opened Settings directly rather than hunting an icon inside a launcher folder |
| ask for the target first, then name it in the operation question | TAP went from 0.132 to **0.9955** on the launcher screen |
| retire operations that leave the screen unchanged | ended an 11-scroll loop; fired again at step 6 |
| label a clickable row with the text it contains | 25 indistinguishable rows became 12 named ones |

the last was decisive and is worth stating precisely, because it looked like a
model failure for two runs and because the first explanation written here was
too broad.

`--filter interactive` on the settings home screen returned 25 clickable
`LinearLayout` rows carrying no text and no id, because each row's label lives
in a `TextView` child the filter dropped. that left
`main_content_scrollable_container` as **the only candidate with a name**, and
a tap on a scroll container is delivered to its centre point, which lands on
whichever row is mid-screen. hence Security & privacy, repeatedly.

**the model was never confused. it was choosing the only option it could name,
at 0.99 confidence, which was the only rational move available to it.**

the correction, from the mimic session after they instrumented it: the
container being present was not itself the fault. focusable-only nodes turned
out to be zero on that screen, so excluding them changed nothing there; the
unlabelled rows were the whole problem. a scroll container belongs in the table
because `scroll` is a real action on it. it must simply not be the only row
with a label.

the two filters do different jobs and both earn their place. mimic answers
"what can be acted on at all"; this agent additionally requires `click` in a
node's actions before offering it as somewhere to **tap**, which removes the
ScrollView from tap candidates while leaving it available to scroll.

mimic has since fixed the labelling at source: an interactive node inherits its
descendants' text, stopping at any descendant that is itself clickable, capped
at four parts. the same screen now returns 13 rows, every one named, no
negative centres, and `About phone / ARES` is exactly what the title-plus-summary
join produces. `--filter` is now comma-combinable and ANDed, so
`--filter interactive,visible` is the recipe. a borrowed label is reported as
`label` and a node's own text stays in `text`, so a consumer can tell which it
got.

the client-side bounds join in `demos/mimic_agent.py` is transitional and comes
out once it can be run against the fixed build.

### navigating is a decision; reading the answer is not

a demo that navigates correctly and returns nothing has not answered the
question it was asked, and for a while this one did exactly that. "find the
android version" was scored as *reached the screen*, with no answer artifact
anywhere in the run.

the agent cannot produce the value by construction. it emits typed choices and
never generates a token, so it can reach the screen holding `Android version /
15` and it cannot say `15`. three ways to close that:

| | works when | cost |
|---|---|---|
| read it from the structured data | the value is on screen in the tree | a substring match |
| a typed `choice` over candidates | the candidates are known in advance | one scoring pass |
| a generative extraction call | neither of the above | a generation, and the thing verdict declines |

**the first is the right answer here and it needs no model at all.** the value
is already structured data in the accessibility tree, in the row the agent
navigated to, rendered by mimic as `Android version / 15`. putting a model on
that would be slower, dearer and less reliable than string handling.

so the division of labour is: **verdict decides where to look, the device
supplies what is there, and a generative model is needed only when the answer
has to be composed rather than read.** jev-ultrafast draws the identical line:
its TEXT_MODEL exists solely to invent a string to type into a field and never
to decide anything.

`--require TEXT` is how a run is scored. each required string is captured the
first time it appears on a screen the agent actually reached, together with the
step, and the whole row is kept because the value travels with its label.

### the pattern across both demos

every intervention that worked changed **what the agent was allowed to choose
between**. every intervention at the prompt or scoring layer did nothing or
made things worse:

| layer | intervention | result |
|---|---|---|
| action space | retire links to pages already read | **0/3 -> 3/3** |
| candidate quality | label browser links with their row | 0/3, no change |
| action space | offer LAUNCH | opened the app directly |
| action space | offer BACK and HOME | made the task reachable at all |
| action space | retire no-op operations | broke a scroll loop |
| question design | ask target first, name it in the operation | 0.132 -> 0.9955 |
| candidate quality | label rows from contained text | task completed |
| prompt | reword the goal to name SCROLL_UP | 0/5 before, 0/5 after, plus false DONEs |
| scoring | prior correction | 16/16 -> 3/16 |
| scoring | order averaging | no change at ceiling, +2/16 on a weak model |

that is the single most useful thing either demo produced.

### open questions

- how large a labelled tree gets on a busy screen, and whether it clears 52
  nodes often enough to need the tournament path. the settings home screen
  yields 12.
- whether an on-device model in the 1-2b class can drive this at all. the
  measured ladder is not encouraging: qwen3.5-0.8b scored 7/16 on the element
  selection benchmark where qwen3.5-9b scored 16/16. this is the honest test of
  the project's central claim and a negative result is worth publishing.
- whether the planner earns its keep. `--planner-model` can be omitted for a
  clean a/b on the same task.

## what neither demo is allowed to do

neither demo may quote an accuracy number without also quoting option mass and
the model that produced it. a model was measured answering wrongly at an option
mass of exactly 1.0000 and a confidence of 1.000; a demo that reports only the
happy path is not evidence.
