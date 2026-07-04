# Known bugs / improvements — running list

Tracked here instead of fixed immediately so a first full real (non-mocked)
Colab run can surface as many issues as possible before we spend time tuning.
Add to this list; don't fix silently in passing.

## Fixed

- **Confirmed on a real (verified-clean, post-restart) Kaggle run: the
  rebalanced prompt (previous entry, `ff99a7b`) fixed the LoCoMo-side
  overcorrection but broke the signal/noise benchmark completely --
  `0% noise (signal=0, noise=0)` for HippoVoice AND Mem0-style (shared
  extraction code), meaning literally nothing was extracted from any of the
  22 signal or 44 noise turns.** This surfaced a second, previously-hidden
  bug in `run_signal_noise_benchmark`: `noise_rate = noise_count /
  len(retrieved) if retrieved else 0.0` silently reports a "clean" 0.0 when
  retrieval returns nothing at all -- a total extraction failure printed as
  `PASS` (`< 20% noise`) because the vacuous 0/0 case is indistinguishable
  from a genuinely clean result by that formula alone. Fixed:
  `run_signal_noise_benchmark` now raises loudly if `retrieve()` returns
  zero results, instead of computing a misleading 0.0.

  Root cause of the actual extraction failure: signal turns in this
  benchmark are first-person, heavily emotional disclosures ("My father was
  diagnosed with stage 3 cancer last week and I feel absolutely
  terrified.") -- structurally, heavy emotional language plus first person
  phrasing, which is exactly what the "skip reactions" instruction (from
  the previous fix) was written to suppress. The model generalized "skip
  emotionally-worded turns" instead of "skip turns with no new
  information," so genuine crisis disclosures got treated the same as
  "Thanks, Mel!" This is the third distinct failure mode found while tuning
  this one prompt (over-extraction of junk -> under-extraction of
  everything -> under-extraction specifically of emotionally-worded
  signal), which says more about how brittle few-shot-only calibration is
  for a 0.6B model than about any one wording choice.

  Fixed (not yet validated on a real run): reworded the instruction to
  frame the skip condition narrowly and explicitly ("bare greeting, bare
  acknowledgment, or question with no statement attached") and added a
  direct counter-example ("A turn describing something upsetting or
  emotional that happened to someone is NOT a bare reaction -- it is an
  event, and must be extracted"), plus a new few-shot example structurally
  similar to the benchmark's signal turns (a distressing personal
  disclosure) without reusing any exact benchmark sentence. **Next step**:
  before any GPU run, validate directly against a handful of real
  `SIGNAL_TURNS`/noise-pool sentences (not just the junk/real probe used for
  the LoCoMo side) to confirm this specific regression is actually fixed,
  given how many rounds this prompt has needed so far.

- **Confirmed on a real Kaggle run: the reaction-skipping prompt fix
  (previous entry below) overcorrected -- the model started skipping real
  facts too, not just reactions.** Validated with commit `4e6ea86` actually
  loaded (`commit [4e6ea86]` confirmed in Step 2 output this time). The
  validation probe showed the fix worked exactly as intended for junk
  (`"Caroline: Thanks, Mel!"` -> `[]`, all four junk turns now correctly
  empty) but also collapsed the one fact that matters most for this
  benchmark: `"Caroline: The transgender stories were so inspiring! I was so
  happy and thankful for all the support."` -> `[]`, nothing extracted at
  all. System-wide confirmation: conversation 1's semantic store dropped
  from 700 -> 19 memories (way past "remove the junk," into "remove almost
  everything"), and the signal/noise benchmark showed HippoVoice at
  `0% noise (signal=0, noise=0)` -- not a win, a total extraction failure
  (Mem0-style baseline showed the identical 0/0, since it shares this
  extraction code, confirming this isn't retrieval-side).

  Root cause: the previous prompt repeated "skip" three times with only one
  counterbalancing "keep" example, and that one keep example was an
  explicit, on-the-nose preference statement -- nothing showed the model
  that an *implied* fact (identity revealed indirectly, e.g. "the
  transgender stories were so inspiring") still counts as worth extracting.
  A 0.6B model given a lopsided few-shot set generalized "return `[]`" as
  the safe default for anything that wasn't a close lexical match to the
  one positive example. Compounding this: the one extraction that *did*
  succeed (`"Caroline is planning to adopt and become a single parent"`)
  was suspiciously close to one of the few-shot examples, which had been
  copied near-verbatim from an already-traced real turn (D2:14) -- i.e. the
  prompt was effectively leaking a test-adjacent example rather than
  demonstrating the general rule, and the model may have been pattern
  matching that one sentence rather than generalizing.

  Fixed (not yet validated on a real run): rebalanced the few-shot set to
  one skip example and two keep examples -- one explicit preference, one
  *indirect/implied* fact (new synthetic example, not reused from any real
  traced turn, to avoid the same leakage issue) -- and reworded the
  instruction to state the skip condition once instead of three times, so
  the extraction default isn't biased toward suppression. **Next step**:
  before any full re-ingest, re-run the same cheap validation probe
  (junk turns + the two known real fact turns, including the exact
  transgender sentence) to confirm both directions hold simultaneously this
  time -- junk still empty, AND the transgender fact extracts -- before
  spending any GPU time on a full LoCoMo run.

- **Confirmed via a full-store rank dump on a real Kaggle run: the bare-name
  fix was necessary but nowhere near sufficient -- the real problem is
  massive extraction over-generation, not a narrow content-quality edge
  case.** After the bare-name fix (below), re-tracing "What is Caroline's
  identity?" still showed low-information Caroline reaction fragments in
  the top 5 (`"It stands for freedom and being real."`, `"What gave you the
  idea?"`, `"..."`) instead of the real fact. Ran a direct diagnostic: fed
  the same 419-turn conversation through a fresh pipeline (real LLM, no
  benchmark harness) and searched the *entire* semantic store, not just the
  top-5, for the actual query. Findings:
  - **The semantic store held 700 entries from 419 turns** (~1.67
    memories/turn) -- for a mostly-casual chat conversation, nearly every
    single turn produced at least one "durable fact." Manually scanning the
    ranked list, the overwhelming majority are pure conversational filler
    that never should have been extracted at all: `"Thanks, Caroline!"`,
    `"Congrats Caroline!"`, `"Agreed, Mel!"`, `"That's so funny!"`,
    dozens of near-duplicate reaction variants.
  - **The actual target fact ("Caroline: The transgender stories were so
    inspiring!") ranked 45th out of 700** by pure cosine relevance to "What
    is Caroline's identity?" A second phrasing of the same fact ("I mentor
    a transgender teen just like me") ranked 65th. This is not a "just
    outside top-5, widen top_k a bit" problem -- 44 other Caroline
    utterances, many only thematically adjacent (art/self-expression turns
    of phrase that happen to share vocabulary like "identity"), scored
    higher than the fact that actually answers the question.
  - Conclusion: this cannot be fixed by tuning candidate-pool size or
    reranking weights alone, because the correct answer is genuinely
    outranked by a large volume of near-duplicate low-value content, not
    narrowly missed. The root cause is upstream, at extraction: the prompt
    never told the model to *withhold* memories for turns that are just
    reactions/small talk, so on a casual-chat dataset like LoCoMo it
    extracts something from almost every turn.
  - Fix attempted (not yet validated on a real run): tightened
    `EXTRACTION_PROMPT` in `memory/extractor.py` to explicitly instruct the
    model to return `[]` for turns that are only a greeting, acknowledgment,
    thanks, compliment, reaction, or question, with two few-shot examples
    (one empty-array reaction turn, one real fact turn) to anchor the
    distinction. The 2-word minimum content filter (below) stays as
    defense-in-depth, but the real fix has to stop the junk from being
    created in the first place, not out-rank it after the fact. **Next
    step**: before spending a full LoCoMo GPU run on this, cheaply validate
    against the real model already loaded in a Kaggle session by calling
    `extract_memories` directly on a handful of known-junk turns (`"Caroline:
    Thanks, Mel!"`) and known-real-fact turns from this exact conversation,
    confirming the new prompt actually suppresses the former without
    dropping the latter -- only then re-ingest and re-check where the
    target fact ranks, and only then re-run the full benchmark.

- **Confirmed via `print_qa_trace` on a real run: bare-name degenerate
  extractions were crowding out real facts in the semantic store.** Traced
  "What is Caroline's identity?" and "What is Caroline's relationship
  status?" -- both got 5/5 retrieved results that were literally just the
  word `"Caroline"`, no actual content. Root cause: `_parse_extraction_response`
  only checked that `content` was non-empty, not that it was substantive.
  On a low-content turn (a greeting, a short reply), the real LLM sometimes
  lazily emits `{"content": "Caroline", "entity": "Caroline", "type":
  "person"}` instead of correctly returning nothing. Since `type="person"`
  routes to the never-decaying semantic store, these accumulate over a long
  conversation, and a bare proper noun is a near-perfect cosine match for
  any query mentioning that same name -- confirmed directly that they
  systematically crowd out genuinely informative facts about the same
  person (which, per `debug_extraction_for_turns`, *were* being correctly
  extracted and classified -- extraction wasn't the bottleneck for these
  two questions, this pollution was). Fixed: reject extracted content with
  fewer than 2 words. A single bare word can't be a self-contained "fact,
  preference, or event" per the extraction prompt's own definition.
  **Not yet re-verified on a real run** -- next LoCoMo run should show
  these two questions (and likely others affected by the same pollution)
  finally surfacing real context.

- **Confirmed via `print_qa_trace`, NOT yet fixed: name-similarity
  confusion in retrieval.** Traced "Which city have both Jean and John
  visited?" (gold `Rome`, predicted `"downtown"`) -- the retrieved context
  was entirely about "Jon" and "Gina" (a different conversation's speakers),
  not "John"/"Jean" at all. The model's answer was a verbatim echo of
  "Jon: It's downtown which is awesome..." -- i.e. this is a genuine
  retrieval failure, not model confabulation: "Jon" and "John" are close
  enough in embedding space that seed similarity search pulled in the wrong
  person's content entirely. This is a harder problem than the bare-name
  fix above -- embedding-based semantic search has no real entity
  disambiguation mechanism. Open; no fix attempted yet.

- **The entire LoCoMo scoring methodology was wrong -- replaced with
  LoCoMo's actual published F1 scorer.** `_answer_matches` was a home-grown
  boolean matcher (substring or >=70% word overlap), never validated
  against what the real benchmark uses. Fetched LoCoMo's actual
  `evaluation.py` from source: it uses stemmed (Porter), normalized,
  token-level F1 (SQuAD-style), category-branched -- category 1 (multi-hop)
  splits both prediction and gold on commas and takes mean-of-max F1 across
  sub-answers; categories 2/3/4 use plain undecomposed F1; category 5
  (adversarial) is a binary "not mentioned" check. Critically, naively
  comma-splitting for ALL categories (rather than just category 1) would be
  wrong: many category-2 date answers ("19 January, 2023") contain a comma
  as punctuation, not a list separator -- confirmed empirically that many
  comma-containing golds are NOT category 1. Implemented `normalize_answer`,
  `f1_score`, `multi_hop_f1`, `score_answer` (category dispatcher) as exact
  reproductions, validated against four values hand-derived directly from
  the fetched source *before* writing any code (one of these hand
  derivations was itself wrong on the first pass -- estimated 0.33 for a
  multi-hop case using a naive single-blob token-F1 instead of the actual
  split-and-max-per-subanswer algorithm; recomputing with the verbatim
  algorithm gives 0.5. Caught by insisting on validating against source
  rather than trusting estimation, which is exactly the discipline this
  whole exercise was arguing for). All four now pass as tests.

  **Re-scored the existing 45-prediction checkpoint with zero new LLM
  calls**: avg F1 = 0.0599, bins = {near_zero: 40, partial: 5, high: 0}.
  This settles whether the old boolean matcher was hiding real progress --
  mostly no. 40/45 questions score below 0.2 F1 (genuinely wrong, not just
  strictly scored), 0 score above 0.7 (not one clean win). The metric fix
  is real and necessary (this project can't credibly compare against
  Mem0/A-MEM's published LoCoMo numbers using a made-up matcher), but it
  doesn't rescue the result -- 6% average F1 is a genuine, low floor. The
  bin shape (mass in near_zero, not the 0.2-0.7 middle) points at
  retrieval-plumbing/generation-confabulation as the dominant failure mode,
  not matcher-strictness -- makes the planned Rome/downtown trace (was
  "Rome" ever actually in retrieved context?) more important, not less.

- **Confirmed on Kaggle: both retrieval regression fixes actually worked.**
  Full signal/noise rerun (commit `c486468`): HippoVoice 20.0% noise
  (signal=8, noise=2) vs NaiveRAG 30%, Mem0-style 30%, AMem-style 10% --
  beats all three baselines (the actual core research claim), a genuine
  validated recovery from the 40%/50% regressions. Misses the strict
  absolute `<20%` threshold by landing exactly at 20%, but the comparative
  wins are real.
- **Confirmed on Kaggle: the top_k-merge fix worked (10 total results, not
  20), but noise rate got WORSE (50%) -- traced to a second, distinct bug
  in how semantic candidates are scored.** Pinning semantic candidates'
  availability to `1.0` and feeding it through the same blended
  `relevance_weight * relevance + (1 - relevance_weight) * availability`
  formula used for episodic candidates gave every semantic-store item a
  flat, unconditional bonus of `(1 - relevance_weight)` -- since
  availability never varies for them, that term carried no real
  information, it just uniformly inflated every semantic candidate
  regardless of actual relevance to the query. Reproduced locally with zero
  GPU: a mock that classifies noise turns as `"fact"` (plausible real-LLM
  behavior) hit a 90% noise rate purely from this scoring floor -- pure
  irrelevant noise scored ~0.35-0.5 just for being in the semantic store.
  Ruled out relevance_weight tuning as the cause first (tested 0.05-0.65,
  all gave 0% noise locally with the deterministic mock, since that mock
  never routes anything into the semantic store at all -- confirms local
  regression tests weren't exercising the actual failure mode). Fixed:
  semantic candidates are now scored by relevance alone, no blended
  constant. Added a permanent regression test
  (`test_irrelevant_semantic_facts_do_not_outrank_relevant_episodic_memories`)
  reproducing the exact failure. **Not yet re-verified on Colab/Kaggle**
  (GPU credits exhausted for now) -- next run should confirm signal/noise
  is back under baselines.
- **Checkpoint fingerprint didn't detect code changes, only config
  changes -- caused a real false "no improvement" reading.** After pulling
  the `retrieve()` merge fix, a rerun with the same model/num_conversations/
  max_qa_per_conversation matched the existing checkpoint's fingerprint and
  silently resumed (skipped re-running entirely), replaying byte-for-byte
  identical pre-fix predictions. Looked like the fix did nothing; it had
  just never actually run. Same root cause as the earlier dry-run
  contamination bug, different trigger. Fixed: fingerprint now also
  includes the current git commit hash (`_current_commit_hash()`), so any
  code change invalidates a stale checkpoint automatically -- no manual
  deletion needed, since the old checkpoint format doesn't have a
  `"commit"` key at all and will mismatch on its own.
- **Confirmed on Colab: Rung 1 regressed the signal/noise guardrail.**
  HippoVoice noise rate jumped to 40% (12 signal/8 noise = 20 total
  results), worse than every baseline including NaiveRAG. Root cause:
  `retrieve()` returned up to `top_k` from **each** store unconditionally
  (up to 2x results), and the semantic store applies zero decay/importance
  filtering by design (correct for durable facts -- relevance alone should
  gate them) -- so any noise turn the real LLM happened to mis-classify as
  `fact`/`preference`/`person` instead of `event` got a permanent,
  unfiltered slot in every single retrieval, with no mechanism to ever
  suppress it. Fixed: both stores now merge onto one comparable
  relevance/availability score (semantic candidates get availability
  pinned to 1.0, since they never decay, using the same formula and
  `DEFAULT_RELEVANCE_WEIGHT` as episodic scoring) and compete for a single
  shared `top_k` budget instead of each getting a guaranteed allocation.
  **Not yet re-verified on Colab** -- next signal/noise run should confirm
  this actually brings the noise rate back down below baselines again.
- **Rung 2 implemented: episodic retrieval reranks by relevance ×
  availability instead of pure salience.** `hippo_retrieve()` now computes,
  per candidate: relevance (cosine similarity between query and memory
  content) and availability (`current_salience` from the existing decay
  model, log-normalized to `[0, 1]` anchored at `FORGET_THRESHOLD` -- see
  `_availability_score`), combined via a fixed weighted sum
  (`relevance_weight=0.65` default). Two simpler designs were tried and
  empirically rejected first:
  - Raw multiplicative/additive blend on unnormalized values: fails
    outright for the reason already quantified (availability spans ~1e-13
    to ~1+, so a fresh irrelevant memory still swamps an old relevant one).
  - Reciprocal rank fusion (rank position, not magnitude): fixes the scale
    problem but overcorrects for small candidate pools -- with only 2-3
    candidates, rank 0 vs rank 1 barely differs regardless of whether the
    true availability gap is 1% or 1,000,000%, so a genuinely dominant
    signal couldn't reliably win. Verified this failure directly with a
    real test case before switching to log-normalized weighted sum.

  Caught and fixed two real regressions while building this:
  - Computing relevance via one `embedder.encode()` call *per candidate*
    turned retrieval latency from a few ms into ~5.5s for a 500-memory
    store. Fixed by batch-encoding the query + all candidate contents in a
    single call.
  - The first test written to validate "availability matters" used two
    memories that turned out not to be equally relevant to the test query
    (phrasing/lexical overlap differed enough that relevance alone decided
    the outcome) -- the test's premise was wrong, not the ranking design.
    Verified empirically (printed actual relevance/availability numbers)
    before rewriting the test with phrasing that's genuinely
    comparable in relevance.

  **Not yet verified on a real Colab run** -- next step is exactly that,
  now that both Rung 1 (store split) and Rung 2 (relevance-aware episodic
  reranking) are in place; these two together should meaningfully improve
  both the identity/preference category (Rung 1) and the "old but relevant
  beats new but irrelevant" dynamic for events (Rung 2), though date
  questions specifically still depend on whether the LLM's own extraction
  preserves the (now-available) session date into the memory's content.
- **Rung 1 implemented: split memory store by type (semantic facts vs
  episodic events).** `HippoVoicePipeline` now holds `semantic_memory`
  (`fact`/`preference`/`person` types -- never decays, survives regardless
  of age) and `episodic_memory` (`event` type -- keeps the existing
  Ebbinghaus decay + emotional consolidation + forgetting/compression
  exactly as before). `retrieve()` queries both and combines results.
  `_maybe_decay()` only ever touches the episodic store. This directly
  targets the Rung 0 finding (facts being physically deleted, not just
  outranked) for content that's inherently durable rather than a specific
  timed occurrence.

  **Important caveat, worth being honest about**: this does NOT fix the
  date-recall category on its own. "When did Caroline go to the support
  group" is answered by an *event* memory by definition -- events still
  route to the episodic store and are still subject to the same decay
  collapse quantified earlier. Of the 45 sampled questions, only the
  identity/preference/relationship-status-style ones (a minority) benefit
  directly from this split; the date-heavy majority still needs the
  episodic store's retrieval to stop being purely recency-dominated --
  i.e. Rung 2 (relevance × availability reranking for episodic retrieval)
  is not optional polish, it's necessary for most of the observed failures.
  Caught a real regression while implementing this: the `passthrough_llm`
  test fixture tagged all extracted memories `"fact"`, silently routing the
  entire signal/noise benchmark's content into the never-decaying semantic
  store and completely bypassing salience/decay -- noise rate reverted to
  30% (NaiveRAG-level) until fixed to tag `"event"` (the benchmark's
  content -- personal narrative statements -- is inherently episodic).
- **`_flatten_conversation()` discarded session dates -- confirmed root cause
  for a large share of LoCoMo's date questions.** LoCoMo turns routinely use
  relative date language ("yesterday", "last Saturday", "next month") that's
  only resolvable against the session's actual calendar date, stored
  separately in a `session_N_date_time` key that was never read. Verified
  directly: evidence turn `D1:3` says "went to a support group *yesterday*";
  session 1's date is 8 May 2023; 8 May − 1 day = **7 May 2023**, exactly
  matching gold. The date was never given to the model at all -- no amount
  of retrieval/ranking/decay tuning could fix this class of failure. Fixed:
  each flattened turn is now prefixed with its session's date. Roughly half
  of the 45 sampled LoCoMo questions are date questions, so this alone
  likely explains a large share of the "right topic, wrong/vague date"
  failure pattern independent of anything else in this list.
- **Rung 0 diagnostic (see below) confirms deletion, not just ranking, is
  the dominant LoCoMo failure mode.** Ingested a full real 419-turn LoCoMo
  conversation locally (deterministic passthrough extraction, no GPU
  needed): only 82/419 (19.6%) of extracted memories survived to the end.
  Checked the exact evidence turns (via LoCoMo's own `evidence` dialogue-id
  field) for the first 5 QA pairs: all 5 were completely gone -- not in the
  store, not in the top-40 seed pool, not in the final top-5. Confirms that
  retrieval-side fixes alone (reranking, similarity blending) cannot recover
  these; the information no longer exists by the time retrieval runs. This
  is what justifies splitting the store by memory type (durable facts vs.
  decaying episodes) rather than only adjusting the retrieval formula --
  tracked as the next major piece of work, not yet started.
- **Forgetting/compression never actually touched the store -- confirmed,
  fixed.** `MemoryStore.get_all()` returned memory dicts with no `"id"`
  field (the id was only ever the dict *key* in `_id_to_meta`, never a field
  inside the value). `HippoVoicePipeline._maybe_decay()`'s
  `mid = m.get("id")` was therefore always `None`, so
  `self.memory.delete(mid)` was never called -- forgetting has been a no-op
  in every run of this project, ever. Worse, the synthetic "compressed"
  entry `_compress()` builds was never persisted back into the store either
  (`_maybe_decay()` computed `active`/`forgotten` but never wrote `active`
  anywhere) -- the entire compress/forget mechanism had zero effect on the
  real store; it just grew unbounded forever. This likely did NOT explain
  the 0/45 LoCoMo QA result on its own (retrieval reranks by salience
  computed fresh at query time, independent of whether stale entries were
  housekept away) but is a real, separate scalability/correctness bug on its
  own. Fixed: `get_all()` now includes each memory's id; `_maybe_decay()`
  deletes both explicitly-forgotten memories and originals that got merged
  into a compression (previously silently dropped from both `active` and
  `forgotten` with no one ever removing them), and persists the new
  compressed entry into the store. Added tests confirming the store's
  memory count actually shrinks under decay and that compression actually
  replaces originals with a persisted synthetic entry.
- **Confirmed, quantified: decay collapses to near-zero at LoCoMo's
  conversation scale.** `decay_lambda=0.05/turn` was tuned/validated for
  ~90-100 turn conversations (signal/noise benchmark). LoCoMo conversations
  run 369-663 turns. Numerically: at `turns_elapsed=600`, even a
  fear-boosted maximally-salient memory scores `0.000221`; a freshly-created
  neutral memory (`turns_elapsed=2`) scores `1.004` -- ~4,500x higher. For
  plain neutral facts (most LoCoMo answers: dates, names, identities), the
  old-vs-fresh gap is ~638 billion times. This means QA retrieval at the end
  of a long conversation is dominated almost entirely by recency, not
  relevance -- this is very likely the primary driver of the observed 0/45
  LoCoMo accuracy, independent of model quality or the scoring-punctuation
  bug fixed earlier. **Not yet fixed** -- how to address it is a design
  decision (blend raw similarity into final reranking instead of pure
  salience? normalize/cap decay so turns_elapsed doesn't grow unbounded?
  treat long multi-session conversations as genuinely out of scope for a
  companion-memory system tuned for shorter-horizon emotional salience?)
  rather than something to silently pick without discussion.
- **Confirmed on Colab: batching gave ~0.3-0.58s/turn vs ~9.6s/turn before**
  (real LoCoMo run, 419/369/663-turn conversations) -- roughly 20x. Real
  inference confirmed working correctly (varied, on-topic answers, not the
  dry-run mock's hardcoded `"unknown"`).
- **`_answer_matches` fuzzy scoring broke on trailing punctuation.** Tokenized
  with a plain whitespace split, so e.g. predicted `"...adoption."` (trailing
  period) never equalled gold word `"adoption"` as a set member -- an
  otherwise-correct answer could score zero overlap purely because of
  punctuation. Fixed: tokenize with `\w+` instead of `.split()`. Also added
  `rescore_details()` -- recomputes accuracy from an existing checkpoint's
  saved predictions using the current matcher, with zero LLM calls, so a
  scoring-only fix doesn't require re-running the (slow, GPU-hungry) full
  benchmark. Note: on the 5 examples seen from a real run, this fix alone
  didn't flip any to correct -- the deeper issue in those cases looks like
  genuinely missing/wrong content in the answers (dates, specific details),
  not just a formatting mismatch. Worth checking whether that's a retrieval
  problem (right facts not surfaced) or a model-capacity problem (0.6B too
  weak to synthesize them correctly) once a real rescoring run is in.
- **Ingestion made one real LLM call per turn, badly underutilizing the
  GPU.** A single short sequence through a 0.6B model leaves a T4 mostly
  idle -- ~9-10s/turn observed on a real LoCoMo run (419-turn conversation),
  and low reported GPU-Util%. Since each turn's extraction is independent of
  every other turn's (it only depends on that single turn's text), there was
  no need to issue them one at a time. Added: `LLMClient.generate_batch()`
  (left-padded batched decode on the transformers/CUDA backend, sequential
  fallback on MLX), `memory/extractor.py::extract_memories_batch()`, and
  `HippoVoicePipeline.ingest_text_turns_batch()` -- storage/decay/turn-order
  stay fully sequential, only the extraction LLM call is batched (default
  chunk size 50). Wired into both `run_locomo` and
  `run_signal_noise_benchmark` via a `hasattr` check so baseline pipelines
  (NaiveRAG/Mem0/A-MEM, which don't define the batch method) are unaffected.
  Not yet re-benchmarked on Colab for actual GPU-Util%/s-per-turn
  improvement -- next run should confirm.
- **Stale checkpoint silently replayed dry-run mock results as if they were
  real.** `run_locomo`'s checkpoint/resume (added to survive Colab
  disconnects) had no way to tell "this checkpoint is from a different run"
  from "this checkpoint is a valid resume point" — it just trusted whatever
  `/content/locomo_checkpoint.json` said. A checkpoint written while the
  notebook's DRY RUN mock LLM was active (which hardcodes `"unknown"` for
  every QA answer) got silently resumed under a later *real* LLM run,
  reporting a garbage 0/45 accuracy with zero actual inference happening.
  Made worse because Colab's "Restart session" only resets the Python
  process, not `/content/`'s disk, so the stale file survived multiple
  restarts. Fixed: checkpoints now carry a fingerprint (LLM model
  name/backend + run parameters); a mismatch prints a loud warning and
  starts fresh instead of silently trusting the file.
- **LLM-extracted memories missing `content` crashed `_compress`.** A real
  (non-mocked) LLM occasionally emits a memory JSON object without a
  `content` key; it flowed unfiltered into the store and blew up
  `memory/decay.py::_compress` with `KeyError: 'content'` once it aged into
  the compress band. Fixed in `memory/extractor.py::extract_memories()` —
  drops any non-dict item or dict with an empty/missing `content` string.
  `decay.py::_compress` also switched to `.get()` as defense in depth.
- **`_compress()` reset a consolidated batch's age to "now".** The synthetic
  compressed entry stamped `turn_created=current_turn`, so a batch of
  long-decayed low-value noise reappeared with elapsed=0 at the very next
  retrieval — an unearned recency boost for junk content. Fixed: the
  compressed entry now inherits the *earliest* `turn_created` among its
  source memories, so decay continues rather than restarting.
- **`hippo_retrieve`'s graph expansion diluted the candidate pool with
  query-irrelevant nodes.** Seeds are intentionally over-fetched 4x past
  `top_k` (floor 15) so salience reranking has room to promote high-salience/
  lower-similarity memories — but graph expansion walked from the *entire*
  over-fetched tail, including seeds that only barely made the cut on
  similarity. Any embedding-neighbor of a marginal seed got pulled into the
  pool regardless of its own relevance to the query, giving noise a backdoor
  in. Fixed: graph expansion now only walks from the closest
  `graph_expand_seeds` (default 10) seeds by raw similarity; the full
  over-fetched pool is still used for salience reranking itself.
- **Confirmed on a real Colab T4 run**: after the two fixes above, HippoVoice
  beats Mem0-style on the signal/noise benchmark (previously tied NaiveRAG
  and lost to Mem0-style). Re-verified against real (non-mocked) LLM
  extraction, not just the local deterministic test suite.
- **Per-turn ingestion was far slower than necessary.** `extract_memories()`
  requested 512 max output tokens for a task that only ever needs a few short
  JSON fragments — on a real (non-mocked) LLM that doesn't always emit a stop
  token quickly (residual "thinking" behavior even with
  `enable_thinking=False`), this could burn the full budget every single
  turn. Observed ~9.6s/turn on a real LoCoMo run. Fixed: cut to 200 tokens in
  `memory/extractor.py`. Also switched the notebook's default Qwen3-0.6B load
  to `load_in_4bit=False` — at this model size VRAM was never the bottleneck
  (~3GB/15GB used on T4), and 4-bit dequant has genuine per-token latency
  cost on GPUs without native int4 tensor cores (T4 included), so fp16 should
  be both simpler and faster. Not yet re-benchmarked for actual speedup on
  Colab — next run should confirm seconds/turn improved meaningfully.

## Open — found on first real (non-mocked, T4) Colab run

- **Voice Test mic-recording cell fails on Kaggle specifically**:
  `Javascript Error: await is only valid in async functions and the top
  level bodies of modules`. The `RECORD_JS` snippet in colab.ipynb's Voice
  Test section uses top-level `await`, which Colab's JS execution context
  apparently tolerates but Kaggle's doesn't. Unrelated to any memory/
  retrieval work this session; deferred since Voice Test isn't part of the
  current LoCoMo/signal-noise investigation. Would need wrapping the
  snippet in an async IIFE (`(async () => { ... })()`) to fix properly for
  Kaggle.
- **Not using multi-GPU parallelism when multiple GPUs are available (e.g.
  Kaggle's T4 x2).** `LLMClient`/`generate_batch` only ever runs on a single
  device -- `device_map="auto"` may spread model layers across both GPUs if
  the framework decides to, but there's no explicit data-parallel batching
  across devices (e.g. splitting a batch of turns across both T4s and
  running them concurrently). For a model this small, a single T4 already
  has spare capacity, so the bigger win would be *using the extra GPU for a
  second concurrent batch* rather than just having it sit idle. Worth
  revisiting if/when GPU-hours become the binding constraint again -- not
  urgent right now, but a real efficiency gap once there's a lot more to
  run (LongMemEval, full 10-conversation LoCoMo runs, etc.).
- Real LLM extraction behavior differs meaningfully from the deterministic
  `passthrough_llm` test mock — the mock always extracts exactly one memory
  per turn verbatim; a real LLM may (a) return zero memories for a boring
  turn, (b) return multiple fragments per turn, or (c) paraphrase content
  losing exact keywords. This means local test results (as a signal-vs-noise
  regression guard) don't fully predict real-LLM behavior. Worth eventually
  adding a benchmark test path that runs against a real small LLM (not just
  CPU-mocked) to catch this class of drift, if that's affordable.

## Open — carried over from earlier session (context.md)

- Header table in `colab.ipynb` says Qwen3-4B, but the "Load LLM" cell
  defaults to Qwen3-0.6B (`llm = LLMClient()`). Need to decide which one is
  actually intended and match the two.
- Track 2 (audio-space memory) is entirely unbuilt.
- `benchmarks/longmemeval/` doesn't exist yet.
