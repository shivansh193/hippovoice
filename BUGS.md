# Known bugs / improvements — running list

Tracked here instead of fixed immediately so a first full real (non-mocked)
Colab run can surface as many issues as possible before we spend time tuning.
Add to this list; don't fix silently in passing.

## Fixed

- **Track 2 (`HippoAudioPipeline` + `GeminiLiveAudioModel`): two real bugs
  found via a live, non-mocked multi-turn run, both now fixed and
  re-verified for real.** Building STM (`[[pipeline_audio2audio.py]]`) and
  a real Gemini Live API adapter was the easy part; getting genuine memory
  *conditioning* (not just capture) working end-to-end surfaced two bugs a
  mocked/local test suite couldn't have caught.

  1. **pyttsx3 SAPI5 COM deadlock on Windows from engine reuse — caused a
     real multi-hour hang in production, not just in tests.** This exact
     bug was found once already earlier this session while writing STM
     tests and worked around only in test code at the time; the real
     pipeline code (`_build_context_audio` caching one `self._tts` engine
     instance and reusing it across calls) was never fixed, and a live
     multi-turn demo run hung for hours confirming it's a genuine
     production issue, not a test-only artifact. Fixed: `_build_context_audio`
     now creates a fresh `load_tts()` engine per call instead of reusing a
     cached one (`self._tts` stays purely a test-injection seam). Applied
     the same fix to the demo script's own `synth()` helper (fresh
     `pyttsx3.init()` per call). Verified via a full `tests/test_pipeline_audio2audio.py`
     run (9/9 passing, ~161s) and a real demo re-run with no hang.

  2. **Gemini Live API's automatic voice-activity detection silently
     truncated multi-segment audio, so injected memory context never
     actually reached the model.** `HippoAudioPipeline` concatenates a
     synthesized context clip (STM+LTM summary) with the real user audio
     before sending it to the model. With automatic VAD (the API default),
     the natural pause between those two segments got treated as
     end-of-turn — the model started responding after only the context
     clip, before the real trailing question was ever processed. Confirmed
     directly via the model's own `input_transcription`: it showed only the
     context, never the actual question, even though STM/LTM state and
     audio concatenation were all correct internally — this looked like a
     generation/relevance failure until the raw input transcript was
     logged. Fixed in `[[gemini_live_model.py]]` by disabling automatic
     activity detection (`realtime_input_config.automatic_activity_detection.disabled=True`)
     and manually bounding the whole context+question blob with
     `activity_start`/`activity_end` (confirmed against Google's own docs
     at https://ai.google.dev/gemini-api/docs/live-guide before
     implementing, not guessed). Re-verified for real after the fix: a fact
     stated in turn 1 ("My dog's name is Max"), asked about again in turn 3
     ("What's my dog's name again?"), correctly produced "Your dog's name
     is Max." — genuine memory conditioning, not just capture. Added
     `test_multi_segment_audio_is_not_truncated_by_vad` (marked
     `@pytest.mark.live_api`) as a permanent regression guard.

- **`run_locomo()` could never actually evaluate the baselines
  (Mem0Baseline/AMemBaseline/NaiveRAG) on the real, F1-scored LoCoMo QA
  benchmark -- only on the separate synthetic signal/noise one.** It
  hardcoded `conv_pipeline = HippoVoicePipeline(...)` per conversation
  regardless of what was passed in. Comparing our avg F1 directly against
  Mem0/A-MEM's own published LoCoMo numbers isn't fair either way -- their
  papers typically use a different (GPT-4-class) backbone LLM and an
  LLM-as-judge scoring methodology, not LoCoMo's own strict token-F1 we
  replicate -- so the only way to isolate the memory architecture as the
  single variable is to run all four systems through the identical LLM,
  data, and scoring ourselves.

  Added `pipeline_factory` (a `(llm_client) -> pipeline` callable) and
  `system_name` params to `run_locomo()`; defaults preserve the exact
  existing HippoVoice behavior. `system_name` is required alongside a
  custom `pipeline_factory` and vice versa, and both are validated with a
  `ValueError` before any data loading or LLM calls happen. Added
  `system_name` to the checkpoint fingerprint too -- otherwise a HippoVoice
  run and a Mem0-style run sharing a `checkpoint_path` could silently
  resume from each other's results.

  `scripts/run_full_locomo.py` now takes `--system
  hippovoice|mem0|amem|naive`, each writing its own checkpoint/results
  file. `NaiveRAG` needed a small adapter in the script (not a change to
  the baseline itself) -- it has no `.llm` attribute since it doesn't use
  one for memory management, but `run_locomo`'s QA-answering step always
  calls `conv_pipeline.llm.generate(...)` regardless of system, so the
  factory attaches the shared client after construction.

  Validated locally with a fully mocked `run_locomo` call (fake
  conversation + fake pipeline, no dataset or real LLM needed) confirming
  the factory is actually used instead of `HippoVoicePipeline`, and that
  the fingerprint records the right `system_name`. **Not yet run for real**
  -- next step is queuing all four systems through the full 10-conversation
  benchmark.

- **Confirmed on a real, genuinely-fresh run with both the decay/relevance
  fix (`cf0d3d1`) and date-preservation fix (`73491b2`) together: LoCoMo
  avg F1 jumped from 5.4% to 15.1% (bins {near_zero: 41→31, partial:
  4→14}), and signal/noise improved to 10% noise (from 20%), now beating
  every baseline cleanly.** This is the first real movement on the ~5%
  floor all session. Verified genuinely fresh this time (`commit
  [73491b2]` confirmed in Step 2 output before running).

- **Jon/John name-confusion bug (open since early this session): fixed.**
  Verified first that this is NOT cross-conversation leakage -- each
  LoCoMo conversation gets a fresh `HippoVoicePipeline` with a uniquified
  in-memory collection (`memory/store.py`'s `EphemeralClient` already
  suffixes collection names with a uuid specifically to prevent this, per
  an existing comment confirming it was tested) -- and `run_locomo`'s QA
  loop confirmed the same `conv_pipeline`/`conv["qa"]` pairing throughout,
  no object-reuse bug. So this really is what it was originally diagnosed
  as: multiple similarly-spelled people (Jon/John/Jean) within the *same*
  conversation, close enough in embedding space that pure cosine similarity
  can't tell them apart.

  Fixed by adding an exact, case-sensitive, whole-word proper-noun match as
  a scoring signal independent of embeddings (`memory/retriever.py`:
  `NAME_MATCH_BONUS = 0.3`, `_extract_proper_nouns`, `_content_matches_names`,
  `_name_match_ids`). Two things were necessary, not just a scoring bonus:
  (1) the bonus itself, applied during reranking, and (2) a full-store scan
  (`_name_match_ids`) unioned into the candidate pool *before* reranking --
  without it, a correctly-named memory crowded out of the raw
  embedding-similarity seed pool by many similarly-worded wrong-name
  memories would never even reach reranking for the bonus to promote.
  Confirmed both are load-bearing with a local test that specifically
  crowds the seed pool with 20 wrong-name distractors. Proper-noun
  extraction excludes common question/determiner words (`Which`, `What`,
  `The`, ...) since nearly every LoCoMo question starts with one and would
  otherwise produce spurious "matches". Applied to both the episodic path
  (`hippo_retrieve`) and the semantic pool (`HippoVoicePipeline.retrieve`),
  since durable facts can suffer the same confusion. Validated with real
  (CPU) sentence-transformer embeddings locally, no GPU needed -- three new
  tests in `test_retriever.py` and one in `test_pipeline.py`.
  **Not yet validated on a real LoCoMo run.**

- **Identity/relationship-status wording mismatch (e.g. "an lgbtq+
  individual who..." vs gold "transgender woman"): NOT a memory/retrieval
  bug -- the model's answers are substantively correct, they just don't use
  the exact canonical word LoCoMo's strict token-overlap scorer needs.**
  Nudged the QA-answering system prompt (in `run_locomo`) toward specific,
  direct-label answers instead of descriptive paraphrases. This is a lower-
  confidence, exploratory change -- unlike everything else on this list, it
  isn't validated against the real model (the extraction-prompt work showed
  how unreliable guessing at LLM behavior can be without a real check), so
  treat the result as informative rather than a proven fix either way.

- **Confirmed on a real (genuinely fresh, restart-verified) run with the
  decay/relevance fix (`cf0d3d1`) in place: it works as designed -- "Jean
  and John" retrieval now surfaces a Rome mention that wasn't there before
  -- but LoCoMo avg F1 still barely moved (5.4%, bins {near_zero: 41,
  partial: 4, high: 0}). Root cause of the remaining floor: extraction was
  silently stripping the date/time prefix off every turn, regardless of
  decay or retrieval quality.** `_flatten_conversation` prefixes each turn
  with `[time on date]` specifically so date questions are answerable, and
  `debug_extraction_for_turns` confirmed the model's input genuinely
  includes it (`[1:56 pm on 8 May, 2023] Caroline: ...`) -- but the
  extracted content was consistently just `'Caroline was happy and
  thankful for all the support'`, no date at all. Since most of every run's
  near-zero bin is date questions, this explains why retrieval improvements
  alone couldn't move the aggregate score: even when the right episodic
  memory survived and got retrieved, the specific date needed to answer the
  question had already been discarded during extraction.

  Fixed by adding an explicit date-preservation instruction and two dated
  few-shot examples to `EXTRACTION_PROMPT`. Took three iterations to get
  right, each validated with a cheap 5-case probe against the real model
  before touching any code:
  - v1 (instruction + one dated-event example: "went on a trip to Rome"):
    fixed the objective-event case (`"Jon lost his job... on 19 January,
    2023"`) but left feeling/opinion turns undated -- the model generalized
    the date-preservation rule too narrowly to the one shape it saw.
  - v2 (added a second, feeling/opinion dated example): fixed both dated
    cases, but broke a previously-working plain non-dated signal turn
    (returned `[]` instead of extracting normally) -- with every example
    now showing a date prefix, the model over-generalized to "only extract
    from turns that have one."
  - v3 (added back an explicit non-dated example, three examples total):
    all 5 test cases correct simultaneously -- both dated shapes preserve
    their date, junk still returns `[]`, plain non-dated turns extract
    normally with no phantom date. All three example shapes are load-
    bearing; this was confirmed by testing each omission separately rather
    than assumed.

  **Not yet validated on a real LoCoMo run** -- next step is exactly that,
  specifically re-checking whether the date-question near-zero failures
  (the majority of the bin) actually resolve now that both decay/relevance
  (`cf0d3d1`) and date preservation are in place together.

- **Confirmed on a real run with Qwen3-4B (extraction now fully working,
  see entry below): the LoCoMo score didn't move at all (5.0% avg F1,
  bins near-identical to every prior run this session, including the ones
  with badly broken extraction). Root cause: episodic decay_lambda was
  never actually the extraction problem's fault -- it was tuned for
  ~90-100 turn conversations and physically deletes almost every episodic
  memory well before a 369-663 turn LoCoMo conversation ends, independent
  of how good extraction or generation are.** Quantified: even a strongly
  emotion-boosted memory (fear, intensity 0.9) crosses FORGET_THRESHOLD
  (0.08) by ~turn 173; a plain neutral memory crosses it by ~turn 50. Every
  single run this session shows the same pattern in the near-zero bin --
  overwhelmingly date/temporal questions where the model correctly says
  "the context does not provide information," which isn't a generation
  failure, it's the episodic memory literally not existing anymore by the
  time the question is asked. This was flagged as an open, undecided design
  question early in this session ("treat long multi-session conversations
  as genuinely out of scope... rather than something to silently pick
  without discussion") and never revisited until now.

  Fixed by making both decay_lambda and episodic relevance_weight
  explicitly overridable rather than hardcoded module constants everywhere:
  - `memory/decay.py::apply_forgetting_cycle` and
    `memory/retriever.py::hippo_retrieve` both now accept a `decay_lambda`
    parameter (default unchanged, so every existing caller -- signal/noise,
    tests, Voice Test -- behaves identically unless it opts in).
  - `HippoVoicePipeline.__init__` accepts `decay_lambda`/`relevance_weight`
    (default `None` = module defaults, same reasoning).
  - `run_locomo()` defaults these to `decay_lambda=0.001,
    relevance_weight=0.85` -- LoCoMo-scale values, not the pipeline's
    short-conversation defaults, since every LoCoMo conversation is long by
    construction. `decay_lambda=0.001` keeps a plain neutral memory above
    COMPRESS_THRESHOLD even at 663 turns elapsed (versus ~1e-14 under the
    old default); `relevance_weight=0.85` (up from 0.65) leans harder on
    relevance now that availability barely discriminates between memories
    any more. Both values added to the checkpoint fingerprint so a stale
    checkpoint from a different decay/relevance setting can't be silently
    reused.
  - `colab.ipynb`'s LoCoMo cell surfaces `DECAY_LAMBDA`/`RELEVANCE_WEIGHT`
    as visible, tunable constants (matching `NUM_CONVERSATIONS` etc.) even
    though they now match `run_locomo()`'s own defaults, so they're easy to
    see and adjust without digging into source.

  Added local (zero-GPU) regression tests confirming the parameter actually
  threads through end-to-end: `test_decay.py::
  test_smaller_decay_lambda_keeps_long_conversation_memories_active`,
  `test_retriever.py::test_decay_lambda_override_keeps_old_memories_available`,
  `test_pipeline.py::
  test_decay_lambda_override_prevents_premature_forgetting_at_scale`.
  **Not yet validated on a real LoCoMo run** -- next step is exactly that.

- **Root cause of the entire extraction-prompt saga (three rounds, see the
  two entries below): 0.6B model capacity, not prompt wording. Switched the
  default model to Qwen3-4B and simplified the prompt back down.** After
  round 3 (`ee04302`) still collapsed to an identical `[]` for every input
  regardless of content (confirmed via raw pre-parse output -- not a
  parsing bug, a genuine deliberate model decision), tested whether
  `LLMClient.generate()` was accidentally stateful (accumulating
  conversation history across the many calls in a long session) -- ruled
  out by reading `llm/client.py`: each call builds a fresh `chat` list from
  scratch, no persisted state. Tested a much simpler prompt (one rule, one
  skip example, one keep example) on the same 0.6B model -- still collapsed
  to identical `[]` output for all 5 test turns, including an unambiguous
  preference statement. Ruled out prompt complexity as the cause too.

  Loaded Qwen3-4B (4-bit, ~3GB VRAM) as a separate model in the same
  session and ran the exact same 5 test turns through the exact same
  simplified prompt: it got every single one right on the first try --
  correctly skipped a mundane weather turn and a bare "Thanks, Mel!",
  correctly extracted a car accident, a friend moving away (with the
  attached emotional state as a separate fragment), and a hiking
  preference. No extra scaffolding needed. This is decisive: three rounds
  of increasingly careful 0.6B prompt engineering each fixed one failure
  mode while introducing another (over-extract junk -> under-extract
  everything -> under-extract specifically emotional turns -> under-extract
  everything again with a simpler prompt) because the actual constraint was
  never the wording -- a 0.6B model with greedy decoding appears to resolve
  this kind of nuanced conditional judgment as an unstable, prompt-surface-
  sensitive coin flip that swings the SAME direction for every input in a
  session rather than genuinely differentiating per-turn content.

  Fixed: adopted the simple prompt (validated against Qwen3-4B, documented
  in `memory/extractor.py` as such) and switched the default model
  everywhere -- `llm/client.py`'s fallback default (both transformers and
  mlx backends) and `colab.ipynb`'s Load LLM cell now default to
  `Qwen/Qwen3-4B` with `load_in_4bit=True`, matching what the notebook's
  own header table had said all along (a mismatch noted as an open item
  earlier in this file, now resolved by this decision). Real cost: slower
  per-turn extraction and more VRAM than 0.6B (still comfortably within a
  T4's 15GB). **Not yet validated with a full benchmark re-run** -- next
  step is a full signal/noise + LoCoMo run on Qwen3-4B to confirm this
  actually holds at scale, not just on 5 hand-picked probe sentences.

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

## Fixed (continued)

- **First full (10-conversation, all-QA) Kaggle commit run (2026-07-11)
  produced a real result — `avg F1 = 24.1% over 1540 questions` (bins:
  near_zero 966, partial 379, high 195) — but three separate bugs meant the
  checkpoint/results JSON never actually reached the Output tab, and a
  later cell crashed the whole commit:**

  1. **`colab.ipynb`'s Voice Test cell crashed the entire "Save & Run All
     (Commit)" job.** There's no microphone in a headless batch commit, so
     the JS recorder cell never actually fires and `/content/user_input.wav`
     is never created — but the next cell unconditionally called
     `transcribe_with_embedding` on that path, `ffmpeg` failed with `No such
     file or directory`, and papermill re-raised it as
     `PapermillExecutionError`, marking the whole commit failed. Distinct
     from the earlier-logged Kaggle JS `await` crash in the same section
     (see below) — this is a separate, Python-side failure mode. Fixed by
     guarding cell-26 (`if not os.path.exists(AUDIO_PATH): skip`) and
     cell-28 (skip TTS if `response_text` was never set) so a cell that's
     inherently interactive-only can't take down a multi-hour unattended
     run.
  2. **`CHECKPOINT_PATH` (LoCoMo cell) and the Save Results cell's output
     path were both hardcoded to `/content/...`.** `/content` is Colab-only
     scratch space — on Kaggle it's a plain, non-persisted directory, so
     everything written there survives only until the container tears down
     and never shows up in the Output tab. Confirmed directly: the Output
     tab after this run showed only the git-cloned repo, no checkpoint or
     results file. (The 24.1% figure wasn't lost only because it was also
     printed to the log, which was captured separately — a JSON artifact
     from that run does not exist.) Fixed both paths to auto-detect
     `/kaggle/working` if present else `/content`, mirroring the `REPO_DIR`
     pattern already used in the Step 2 clone cell.
  3. **The Save Results cell (cell-30) read `locomo_result['accuracy']` /
     `['correct']`, which haven't existed since the F1-scoring rewrite
     earlier this session** — `run_locomo()` returns `avg_f1`/`total`/
     `bins`/`details`. Would have raised `KeyError` even with the path fix
     above, on Colab or Kaggle. Fixed to use the real keys. Also guarded
     the Colab-only `google.colab.files.download` call, which raises
     `ModuleNotFoundError` on Kaggle (not something the earlier crash even
     let execution reach, but would have failed on the very next Kaggle
     run once #1 was fixed).

  None of this affects the 24.1% number itself, which is real and already
  fully captured (aggregate + full per-question breakdown) from the run's
  log output.

## Fixed (continued)

- **The earlier pyttsx3 SAPI5 fix ("fresh `load_tts()` engine per call")
  was incomplete — it wasn't a reuse bug, it was a per-process bug.**
  Surfaced while diagnosing why a local HippoAudio LoCoMo run
  (`benchmarks/locomo/evaluate_audio.py`) sat at ~2.5 min/question and
  occasionally hung outright with no exception. Killed the hung process
  and inspected it at the OS level: near-zero CPU growth over 20+ minutes,
  both TCP sockets to Google in `CLOSE_WAIT` — looked at first like a
  dropped Gemini Live connection (see the `GeminiLiveAudioModel` timeout
  fix above), but adding flushed per-step timing to the diagnostic showed
  the hang was actually in `_build_context_audio`, a purely local step
  with no network involved. Isolated it completely with a standalone
  script — zero pipeline code, just `load_tts()` called twice in a row —
  and the *second* call hung indefinitely. Two fresh, separate engine
  objects, no reuse of either one, and it still deadlocked: proof that
  `tts/model.py`'s old `load_tts()` (which called `pyttsx3.init()`
  directly) was itself the trigger every call site was hitting on its
  second-or-later call per process, not just `tts/synthesize.py`'s
  `synthesize()` as the earlier fix assumed. This also means the earlier
  "confirmed fixed" verification only worked because that test run never
  happened to call `load_tts()` a second time in the same process.

  Real fix: `tts/model.py`'s `load_tts()` no longer touches `pyttsx3` at
  all — it returns a lightweight `TTSHandle` carrying rate/volume only.
  The actual `pyttsx3.init()` → configure → `save_to_file`/`say` →
  `runAndWait()` sequence now runs inside a fresh OS subprocess
  (`tts/_synthesize_worker.py`), spawned once per `synthesize()`/`speak()`
  call by `tts/synthesize.py`. A new process always gets a new COM
  apartment, so it no longer matters how many times TTS has already run
  in the parent process. Also added a `SYNTHESIZE_TIMEOUT_SECONDS=30`
  guard around the subprocess call, same discipline as the
  `GeminiLiveAudioModel` fix, in case a *different* SAPI5 issue ever hangs
  the worker itself.

  Verified with a direct regression test (`tests/test_tts.py::
  test_synthesize_can_be_called_repeatedly_without_hanging`) that calls
  the real subprocess-isolated path three times in one process — the
  exact shape that hung before this fix — plus the full existing suite
  (`tests/test_pipeline_audio2audio.py`, `tests/test_evaluate_audio.py`,
  `tests/test_pipeline.py`), all passing unchanged since every existing
  test mocks `tts.model.load_tts`/`tts.synthesize.synthesize` at the
  function boundary this fix preserves. This also retroactively explains
  why the one real local HippoAudio LoCoMo run that did complete (60
  turns, 15 QA, avg F1 28.2%) averaged ~130s/question — almost certainly
  mostly SAPI5 COM contention occasionally resolving itself rather than
  real per-question cost, meaning that number's wall-clock time, not its
  F1 score, was the misleading part.

## Fixed (continued)

- **First real, completed head-to-head LoCoMo comparison: Mem0-style vs
  HippoVoice, same harness, same LLM, same scorer.** Mem0-style avg F1 =
  23.42% over 1540 questions (bins: near_zero 973, partial 380, high
  187), run via `colab.ipynb`'s `SYSTEM='mem0'` path on Kaggle, completed
  2026-08-31. HippoVoice's own 24.1% (2026-07-11, bins 966/379/195) edges
  it out, but it's close, and the error-distribution shape is nearly
  identical between the two. Every other baseline number in this project
  before now came from the smaller signal/noise synthetic benchmark, not
  this one -- this is the first time the actual LoCoMo avg-F1 metric has
  had a real comparison point at all. A-MEM-style and NaiveRAG are next.

  This specific run also survived a real interruption: it was cancelled
  by Kaggle mid-run (a session-limit hit, not a crash or anything either
  of us triggered) 7 of 10 conversations in, with a real partial result
  (avg F1 ~23.9% over 1035 questions) already sitting in its checkpoint.
  Rather than losing that and restarting cold, the checkpoint was pulled
  down, its `fingerprint.commit` field patched to the rerun's current
  HEAD (the only field that would have mismatched -- nothing in the
  intervening commits touches this baseline's actual behavior, only
  unrelated WeightEdit/audio fixes), uploaded as a Kaggle Dataset, and
  wired into a small resume-copy step added to the LoCoMo cell (gated on
  `SYSTEM == 'mem0'` and on the checkpoint path not already existing, so
  it can never silently clobber a genuinely fresh run). `run_locomo`'s
  own existing checkpoint-resume logic did the rest. Final checkpoint
  confirms all 10/10 conversations, 1540/1540 questions -- a complete,
  real result either way.

- **A-MEM-style's full LoCoMo run completed cleanly (no cancellation,
  no resume needed this time)** via the same `colab.ipynb` `SYSTEM='amem'`
  path, pushed as version 4 of the `hippovoicefinalbenchmark` kernel,
  2026-08-31. avg F1 = 22.02% over 1540 questions (bins: near_zero 1055,
  partial 287, high 198) -- the most near-zero-heavy of the three
  completed systems so far. HippoVoice (24.1%) now leads both real
  baseline comparisons on the actual LoCoMo metric; the gap to Mem0-style
  (23.4%) stays the closer of the two. NaiveRAG is the last baseline
  still queued.

- **`WeightEditBaseline` (ROME on GPT-2 XL) went from "never worked" to
  a fully validated, correct edit -- three real, distinct bugs found and
  fixed in sequence, each only reachable once the previous one was
  cleared.** All three confirmed on live Kaggle T4 runs, not guessed:

  1. **CUDA OOM at the edit step, 14MB short of the card.** Root cause,
     confirmed by reading EasyEdit's own `compute_v.py`: ROME batches
     ~15-20 reworded versions of the same sentence into one
     forward+backward pass through all 48 layers (so the edit
     generalizes across phrasings) -- that batched activation memory,
     not the ~6GB of plain weights, exhausted the card. Fixed with
     `gradient_checkpointing_enable()` on the loaded model.
  2. **Checkpointing barely helped (free memory moved from 6.8MB to
     10.8MB, still short).** HuggingFace gates checkpointing with
     `if self.gradient_checkpointing and self.training`, and EasyEdit
     leaves the model in `eval()` mode -- checkpointing was enabled but
     never engaging. Fixed by wrapping the edit call in
     `model.train()`/`model.eval()`.
  3. **The OOM was gone, but "AFTER edit" came back byte-identical to
     "BEFORE edit" -- a silent correctness bug, worse than a crash.**
     Two compounding causes, both confirmed by reading EasyEdit's source
     directly rather than guessing: GPT-2 XL's real dropout (~10%) got
     turned back on by `train()` mode, corrupting ROME's correction
     vector (fixed by zeroing every `nn.Dropout` module's `p`, making
     dropout a true no-op in either mode); and separately,
     `editor.edit()`'s default `sequential_edit=False` calls
     `restore_after_edit()` after every edit, which for any algorithm
     not in EasyEdit's small special-cased lists (ROME included) manually
     copies the original parameters back over the model -- applying the
     edit and then immediately undoing it, every single call. Fixed with
     `sequential_edit=True`.

  Also ruled out along the way: Kaggle's P100 accelerator is not a
  memory fix for this -- its preinstalled PyTorch build has no CUDA
  kernels for the P100's (Pascal, sm_60) architecture at all, unrelated
  to how much VRAM it has.

  Final validated sanity check: `BEFORE edit: 'Paris, France. The tower
  is the'` -> `AFTER edit: 'Rome.\n\nThe Vatican is located'` -> `AFTER
  reset: 'Paris, France. The tower is the'`. Correct edit, correct
  revert. The one remaining blocker on a full WeightEdit LoCoMo run is
  unrelated to any of this: `GEMINI_API_KEY` attached via Kaggle Secrets
  and interactively verified working hasn't carried over to a CLI-pushed
  "Save & Run All" commit across three separate attempts -- looks like a
  real platform quirk, not something fixable from the notebook side.

## Fixed (continued)

- **New self-hosted HippoAudio backend added: `Qwen25OmniAudioModel`
  (Qwen2.5-Omni-3B), roughly 4x faster than `GeminiLiveAudioModel`,
  validated on a real GPU before writing any adapter code.** Started
  from a real, honest question: Gemini Live's confirmed ~35-45s/question
  latency (see gemini_live_model.py's docstring) is real API latency,
  not something fixable on our side -- is there a faster backend?

  Two other candidates got investigated and ruled out for the same real
  reason, confirmed from their actual source/docs, not guessed: **Moshi**
  and **Amazon Nova Sonic** are both full-duplex/continuous conversational
  models with no clean per-turn completion signal. Nova Sonic's own
  official reference sample never even checks for one -- it just plays
  audio through speakers until a human presses Enter, and its session
  model has an 8-minute connection cap that only makes sense for a live
  continuous call, not discrete Q&A turns. Building a heuristic (guess a
  turn is "done" from a gap in output) would carry a real correctness
  risk this project has been careful to avoid all along: a wrong guess
  wouldn't crash, it would silently truncate or run past the real answer.

  Qwen2.5-Omni's `model.generate()` turned out to be a genuine
  synchronous call (confirmed from its own HF model card): send one
  input, get back one complete `(text_ids, audio)` result -- no
  streaming ambiguity, mapping directly onto `AudioToAudioModel.respond()`.

  Before writing any adapter code, validated this for real on a
  throwaway AWS `g4dn.xlarge` (provisioned via the AWS CLI, torn down
  completely afterward -- no lingering instance, security group, or IAM
  role/profile left behind, confirmed via `describe-instances`/
  `describe-volumes`): loaded in ~120s (one-time), answered a real
  spoken question correctly ("What is the capital of France?" ->
  "Paris"), used 12.6GB peak VRAM (fits a T4's ~15GB with real headroom),
  and completed the actual turn in **9.7s** -- roughly 4x faster than
  Gemini's confirmed 35-45s. Chose the 3B variant over 7B specifically
  for VRAM safety: 7B's plain bf16 weights alone (~14-15GB) leave
  essentially no room on a T4 with ~14.5GB usable (the same card that
  OOM'd on GPT-2 XL editing elsewhere in this project), while 3B's ~6GB
  leaves real headroom.

  One real parsing bug caught during that same live test, fixed before
  it ever reached local code: `model.generate()`'s `text_ids` includes
  the ENTIRE rendered chat template (system/user/assistant), not just
  the new response -- naively decoding the full thing returns
  `"system\n...\nuser\n\nassistant\nParis"` instead of just `"Paris"`.
  Fixed by slicing off the prompt's own token length
  (`text_ids[:, inputs.input_ids.shape[1]:]`) before decoding, not
  string-matching on `"assistant\n"`, which wouldn't reliably survive
  arbitrary system/user content.

  Also confirmed a real, stated constraint from Qwen's own warning
  during that test: audio output quality is only guaranteed with its
  default system prompt -- overriding it (the same pattern
  `GeminiLiveAudioModel.system_instruction` uses for concise-answer
  tuning) is supported here too, but flagged as a real risk to audio
  quality, not adopted by default the way it was for Gemini.

  Needs a specific install (the Qwen2.5-Omni preview transformers
  branch, plus `qwen-omni-utils`/`audioread`/`librosa`/`soundfile`), not
  bundled into `requirements.txt` since it conflicts with the project's
  normal `transformers>=4.40.0` pin -- same reasoning as EasyEdit's
  separate install. Not exercised by the local test suite for the same
  reason (`tests/test_qwen_omni_audio_model.py` mocks the model/
  processor to cover the adapter's own logic instead). Full local suite:
  203 passed, 2 deselected.

- **First complete, real HippoAudio benchmark result on Qwen2.5-Omni:
  40/40 questions, avg F1 25.81%, after a chain of four real bugs found
  only by actually running the full pipeline on a GPU.** The sanity
  check above (12.6GB VRAM, one tiny question) didn't surface any of
  these -- each needed the real LoCoMo pipeline (TTS-synthesized
  questions, real conversation-length context, dozens of consecutive
  calls) to show up:

  1. **pyttsx3/espeak silent length-limit failure** (`tts/synthesize.py`):
     text over ~100-150 characters made the worker subprocess exit 0
     with no output file and no error -- a genuine silent failure, not a
     crash. Initially misdiagnosed as a timing race and "fixed" with a
     retry-with-backoff in `pipeline_audio2audio.py`'s
     `_concatenate_audio` (wrong -- the exact same error recurred next
     run). Properly root-caused via isolated binary-search testing on
     the real Linux machine (ruled out shell-escaping) and fixed for
     real via sentence-boundary chunking + retry-by-halving. A third,
     distinct failure mode (corrupt file content, not missing) turned up
     on the very next run and needed a separate audio-readability check,
     not just a file-existence check.

  2. **GPU memory climbing across calls in the same process**: peak VRAM
     grew from the sanity check's ~12.6GB (one call) to a real
     `torch.OutOfMemoryError` by question 3 on the first attempted full
     run. `generate()`'s intermediate tensors (the DiT-based Token2Wav
     vocoder's own working memory, per the traceback) weren't being
     released between calls. Fixed with `torch.cuda.empty_cache()` +
     `gc.collect()` after every `respond()` call -- took the same run
     from failing at question 3 to question 15.

  3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` regression**:
     tried next, on the strength of PyTorch's own OOM error message
     suggesting it (and this project's own WeightEdit OOM having used
     the same knob). A live re-run with it set OOM'd on the very FIRST
     `generate()` call instead of question 15 -- the allocator's own log
     showed `expandable_segments: memory mapping failed` twice before
     the fatal error. A genuine regression for this model's allocation
     pattern, not an improvement -- reverted, with a regression-guard
     test (`test_load_does_not_set_expandable_segments`) added so it
     doesn't quietly come back.

  4. **Still OOM'd on the first real question even after reverting #3**:
     with only the confirmed-good fixes (cache-clearing, plus a concise
     `system_instruction` addressing a separate verbosity problem that
     was burying otherwise-correct answers in filler and hedging) in
     place, a fresh run still hit `torch.OutOfMemoryError` trying to
     allocate 2.00 GiB, inside `token2wav(...)`'s vocoder, on question 1
     -- proving the leak-across-calls fix and the `expandable_segments`
     revert were both real but incomplete; a single T4's 14.56GB was
     already nearly exhausted by one real forward pass, independent of
     any cross-call leak. Qwen's own HF model card documents
     `return_audio=False` as saving "about ~2GB of GPU memory" by
     skipping exactly that vocoder step -- the 2.00 GiB figure lined up
     exactly. Added as an opt-in constructor param (default `True`, so a
     live conversational deployment keeps real audio out); since this
     benchmark only scores the text transcript and always discards the
     response audio path anyway, there's nothing lost setting it `False`
     here. A minimal single-question sanity check on a fresh instance
     confirmed VRAM held flat (11.98GB/12.09GB allocated/reserved,
     identical before and after two consecutive calls) before spending a
     full run on it.

  With all four fixes in place, a clean 40-question run (2 conversations
  x 20 QA, `gemini-3.5-flash-lite` for extraction) completed in 1545s
  (~26min) with **zero errors**: avg F1 **25.81%** (21 near-zero, 15
  partial, 4 high-scoring), comparable to this project's own text-only
  Mem0-style baseline (23.4%, see README.md) despite going through a
  full TTS -> Qwen2.5-Omni -> transcript round trip rather than reading
  text directly. This replaces the earlier stale/partial 14/20 result
  (see prior entry) which reflected verbosity/hedging rather than a
  complete or fully-fixed run.

- **Real audio-in/audio-out + memory recall confirmed live on Qwen2.5-Omni
  -- the same "fact recalled turns later" win this project already had on
  GeminiLiveAudioModel, now on the self-hosted backend, with a GPU tier
  that actually fits it.** The 40-question benchmark result above used
  `return_audio=False` (text-out only, since that's all the benchmark
  scores) -- real audio OUTPUT for a live conversational deployment was
  a separate, still-open question until validated here.

  A single T4 (`g4dn.xlarge`) was confirmed, via a real Gemini-free
  sanity script (seeded LoCoMo turns directly into STM/LTM to isolate
  GPU behavior from Gemini's own reliability, see below), to
  genuinely NOT have enough headroom for `return_audio=True` at
  realistic full-conversation context length -- not a tuning problem:
  cutting the context budget (stm_window 5->3, top_k 5->3) fixed the
  audio ENCODER's OOM but simply exposed a second, separate ~2GB fixed
  overhead in the token2wav VOCODER that doesn't shrink with less
  context. Model weights + text generation alone already use ~12.7GB of
  the T4's 14.56GB usable, leaving too little room for both stages
  together regardless of context size.

  Moving to a `g5.xlarge` (A10G, 24GB) fixed this with zero code
  changes and zero context-budget cuts: the exact settings that OOM'd
  on the T4 (stm_window=5, top_k=5, full 60-turn context) ran cleanly,
  VRAM barely moving (11.98GB->12.17GB, more than half the card still
  free). Confirmed with a real live 3-turn conversation via
  `process_turn()` (not the benchmark's read-only `answer_question()`):
  turn 1 "My favorite color is blue" -> turn 2 unrelated filler -> turn
  3 "What is my favorite color?" got the real spoken reply "Your
  favorite color is blue." -- real TTS input, real Gemini extraction,
  real vector retrieval, real Qwen2.5-Omni audio output, zero OOM,
  VRAM flat at ~12.2GB throughout.

  Separately, a real Gemini outage was hit and independently confirmed
  (direct curl calls bypassing this project's own code entirely showed
  503/504 errors, and 1 of 3 raw calls timed out completely) while
  building the diagnostic scripts above -- not a quota/credits issue
  (that's a distinctly different 429 RESOURCE_EXHAUSTED error this
  codebase already handles separately), just real transient instability
  on Google's side that has since cleared. Worked around for the
  diagnostic scripts specifically by bypassing `ingest_text_turn()`'s
  Gemini-dependent `extract_memories()` call and writing raw turns
  directly into `HippoMemory` via `_add_memory()` -- valid for isolating
  GPU/memory behavior specifically, not a replacement for real
  extraction (the live 3-turn demo above uses real Gemini extraction,
  since only 3 calls were needed there and real extraction quality is
  the point of that demo).

  One real, unrelated bug caught along the way: `HippoAudioPipeline`'s
  `_maybe_decay()` reads `self.llm`, a lazy PROPERTY that instantiates a
  completely different local model (`llm.client.LLMClient`, 4-bit via
  `bitsandbytes`) when `self._llm` is `None` -- not just "stays None" as
  the raw attribute name would suggest. Surfaced as a `bitsandbytes`
  `PackageNotFoundError` on an AMI that doesn't have it installed, when
  a diagnostic script naively called `_maybe_decay()` after passing
  `llm_client=None`. Not a real-world issue (real callers always pass a
  real `llm_client`), but a real surprise worth knowing about this
  pipeline's `None`-handling if extending it further.

- **Track 1's avg F1 improved from 24.1% to 27.74% via a real, validated
  retrieval hyperparameter sweep -- `top_k`, not decay/relevance tuning,
  turned out to be the actual driver.** Ran a coordinate-descent sweep
  (`decay_lambda` × `relevance_weight` × `top_k`) on Kaggle rather than a
  full grid, for a real, specific reason: `decay_lambda` changes what
  physically gets DELETED during ingestion (the forgetting cycle runs every
  10 turns and removes memories below `FORGET_THRESHOLD`), so each
  `decay_lambda` value genuinely needs its own fresh ingestion -- but
  `relevance_weight` and `top_k` only affect the read-time reranking
  formula (`hippo_retrieve` reads `self.relevance_weight`/`self.decay_lambda`
  off the pipeline at call time, and `top_k` is passed directly to
  `retrieve()`), so they can be swept for free on top of one already-
  ingested pipeline. Confirmed this matters: mutating `decay_lambda`
  *after* ingestion would silently reuse whatever the forgetting cycle
  already deleted under the ORIGINAL rate, giving a wrong answer for "what
  if we'd decayed slower from the start" -- so only `decay_lambda` got the
  expensive fresh-ingestion treatment; `relevance_weight`/`top_k` were
  swept cheaply on top of the winning ingestion.

  Cheap-subset results (1 conversation, 25 QA pairs, holding the other two
  variables at their current defaults while sweeping each):

  | Variable swept | Values tried | Winner | Subset F1 |
  |---|---|---|---|
  | `decay_lambda` | 0.0005 / 0.001 / 0.002 / 0.003 | 0.0005 (tied exactly with 0.001) | 0.256 |
  | `relevance_weight` | 0.65 / 0.75 / 0.85 / 0.95 | 0.85 (unchanged from current default) | 0.256 |
  | `top_k` | 3 / 5 / 8 / 10 | **10** | **0.287** |

  `top_k` was the only variable that actually moved the number -- and by a
  real margin (0.256 → 0.287 on the subset, roughly +3 points). `decay_lambda`
  0.0005 and 0.001 scored IDENTICALLY on the subset (0.255984126984127 both,
  to 12 decimal places) -- not a real difference, just whichever was tested
  first in a strict `>` comparison. Kept 0.0005 anyway since that's the
  exact value that went into the full-set validation below, not because
  it's confirmed better than 0.001.

  Validated the winning combination (`decay_lambda=0.0005,
  relevance_weight=0.85, top_k=10`) against the real, full 1540-question
  LoCoMo set (not just the cheap subset) -- this is the number that
  actually matters:

  | | Original (`top_k=5`) | New (`top_k=10`) |
  |---|---|---|
  | avg F1 | 24.1% | **27.74%** |
  | near-zero | 966 (62.7%) | 877 (56.9%) |
  | partial | 379 (24.6%) | 431 (28.0%) |
  | high | 195 (12.7%) | 232 (15.1%) |

  The whole distribution shifted favorably (fewer near-zero, more partial
  *and* more high), not just the mean -- real evidence this is a genuine
  improvement, not an artifact of a few lucky questions. Full-set run took
  ~5.5 hours on a T4, notably slower than earlier full runs -- plausibly
  `top_k=10`'s larger per-question retrieved context, though this wasn't
  root-caused further since the run completed successfully.

  Deliberately did NOT bump `run_locomo()`'s own harness-wide `top_k`
  default from 5 to 10: Mem0-style (23.4%) and A-MEM-style (22.0%) were
  both run at `top_k=5` and have never been re-swept at 10, so changing the
  shared default would silently turn the README's comparison table into an
  apples-to-oranges comparison. `top_k` is now exposed as a real, named
  parameter on `run_locomo()` (previously hardcoded to `5` inline with no
  way to override it at all) -- `scripts/run_full_locomo.py` and
  `colab.ipynb`'s LoCoMo cell both opt into `top_k=10` explicitly for the
  `hippovoice` system only, same pattern as the existing
  `decay_lambda`/`relevance_weight` overrides.

- **Real, currently-open Kaggle platform bug hit and worked around while
  running the sweep above: this account got repeatedly assigned a Tesla
  P100, and Kaggle's current GPU Docker image ships a PyTorch build with
  zero compiled CUDA kernels for that architecture at all.** Four
  consecutive kernel pushes all landed on the same P100 (compute capability
  6.0); PyTorch's own startup warning states plainly that the installed
  build only supports `sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`. First
  suspected as bitsandbytes-specific (crashed with `Error named symbol not
  found ... ops.cu` + a hard segfault during 4-bit weight quantization) --
  added a runtime GPU-compute-capability check to fall back to a clean
  fp16 load path (`LLMClient(load_in_4bit=False)`, which already existed
  and never imports bitsandbytes) below `sm_70`. That fix got further but
  hit a harder, more fundamental wall on the very next real generate()
  call, in plain fp16 with no quantization at all:
  `torch.AcceleratorError: CUDA error: no kernel image is available for
  execution on the device` -- confirming this isn't narrowly a
  bitsandbytes problem, the installed PyTorch build genuinely cannot run
  ANY real CUDA operation on this GPU.

  Confirmed via real web research (not guessed) that this is a known,
  currently unresolved, actively-tracked Kaggle platform regression, not
  anything specific to this account: [Kaggle/docker-python#1546](https://github.com/Kaggle/docker-python/issues/1546)
  ("Pytorch CUDA P100 GPU Incompatibility"), plus several duplicate
  reports. The real, working fix: `kaggle kernels push` has a documented
  `--accelerator` flag accepting exact IDs like `NvidiaTeslaT4` (found via
  web research after an earlier guess with an arbitrary string was
  silently accepted/ignored by the CLI, wasting one push) -- forcing
  `NvidiaTeslaT4` explicitly resolved it immediately, with zero further
  GPU-lottery risk.

  Investigated a real fix for the upstream Kaggle bug itself (not required
  for this project, done as a good-faith side contribution): their GPU
  image is `FROM` Google's Colab base image and just reinstalls whatever
  torch version that base image already has, rather than choosing a CUDA
  index itself -- so the actual sm_60-dropping wheel index is inherited,
  not chosen in `Kaggle/docker-python`. Confirmed against PyTorch's own
  `.ci/manywheel/build_env_setup.py` that the `cu126` wheel index still
  ships `sm_60` kernels (unlike the newer `cu128` line, which trades it for
  Blackwell/`sm_100+` support). Opened
  [Kaggle/docker-python#1561](https://github.com/Kaggle/docker-python/pull/1561):
  a small, GPU-template-scoped `Dockerfile.tmpl` change that reinstalls
  the *same* torch version already present, but from `cu126` instead.
  Explicitly flagged in the PR as unverified against Kaggle's real build
  pipeline/GPU fleet (no way to test that from outside), and flagged an
  already-open same-day PR removing Kaggle's P100 CI agent entirely --
  which may mean P100 is being deprecated rather than fixed, information
  the maintainers need to actually decide whether this PR is even wanted.

- **Root-caused three distinct, separately-attributable failure patterns
  behind the confirmed 27.74% avg F1 result -- not one generic "retrieval
  is bad" problem.** Downloaded the full per-question `details` (all 1540
  entries, including the actual retrieved `context` string per question --
  logged specifically so this kind of analysis doesn't need a fresh run)
  from the official top_k=10 confirmation run and analyzed it directly:
  F1 broken down by category, and for near-zero questions, how much of the
  gold answer's own tokens were actually present in what got retrieved.

  Finding: only 39% of near-zero failures were genuine retrieval misses
  (gold answer never in context at all). The other 61% had the gold
  answer partially (49%) or even fully (10%) sitting right there in
  context, and the model still scored near-zero -- meaning generation-side
  and extraction-side problems account for more of the failure budget than
  a pure "retrieval isn't finding the right memory" story would suggest.
  All three looked like cheap, well-scoped fixes going in; on real testing
  all three were reverted -- two made things measurably worse or provided
  no measurable benefit (see the two entries below), one broke an existing
  test outright (see the third entry below). Category 3's hedging problem
  and category 2's relative-date gap are both still real, confirmed, and
  unfixed -- this analysis is left here as a documented starting point for
  whoever picks it up next, not as closed work.

  Category breakdown (out of 1540 questions): category 1 (multi-hop, n=282)
  22.2% avg F1; category 2 (temporal, n=321) 26.4%; category 3 (inferential
  "would X likely...", n=96) **16.6% -- worst category by a wide margin**;
  category 4 (largest bucket, n=841, mostly single-fact lookups) 31.4% avg
  but the most total near-zero questions (470) by raw count.

- **Tried and reverted: category 3's hedging fix made F1 worse, not
  better, on real validation.** 19% of category 3's near-zero answers were
  the model outright refusing to infer -- "the context does not explicitly
  state... therefore cannot be determined" -- scoring 0.044 avg F1, versus
  0.194 for answers that actually attempted the inference the question
  asked for (LoCoMo's own gold answers for this category are judgment
  calls like "likely no" or "yes, since she collects...", not verbatim
  quotes). Category 3 answers were also the longest of any category (13.7
  words avg vs 6.9 for category 2) despite the prompt saying "be concise".
  Hypothesized root cause: the QA system prompt said "Answer the question
  using ONLY the provided context" -- read by the model as "if it isn't
  stated verbatim, refuse." Pulled the inline prompt out to a named
  `QA_SYSTEM_PROMPT` constant in `benchmarks/locomo/evaluate.py` with
  explicit permission to infer for judgment-style questions specifically,
  while keeping plain factual questions grounded in what's stated.

  Validated on a cheap, targeted Kaggle run (T4, `Qwen/Qwen3-4B`) against
  the real 96 category-3 questions, reusing their already-logged retrieved
  context so only the QA step itself was retested: the new prompt scored
  **worse**, 0.118 avg F1 vs. the original's 0.166, and hedging went **up**
  (23% vs. 19%) despite the entire point being to reduce it. Spot-checking
  the actual generations showed the new prompt did make the model attempt
  more inferences instead of refusing outright -- but the attempted
  inferences were often longer and drifted further from the gold phrasing
  (e.g. gold `"likely no"` vs. new prompt's `"no, since caroline mentioned
  that counseling and support groups improved her life and that her
  support system..."`), costing more stemmed-token F1 than the hedging
  it fixed. Reverted `QA_SYSTEM_PROMPT` to the original wording; category
  3's real problem (worst-scoring category by a wide margin) is still
  unfixed. One caveat on this validation: the "old" numbers are the
  original production run's logged predictions (not regenerated in this
  same session), so some of the gap could in principle be sampling
  variance rather than purely the prompt -- but the hedge-rate move in the
  wrong direction, on the prompt's own stated goal, is hard to explain as
  noise alone.

- **Tried and reverted: relative-time-word resolution in extraction did
  not resolve a single real case on validation, and regressed one.** Real
  example found in the actual logged context: "Jon went to Paris
  yesterday" got extracted and stored with "yesterday" left in as-is, so
  the benchmark's own predicted answer to "When was Jon in Paris?" was
  literally `"yesterday."` -- not a hallucination, a faithful readout of
  what got stored. Hypothesized root cause: `memory/extractor.py`'s
  `EXTRACTION_PROMPT` already teaches attaching a turn's own date/time
  prefix to a fact (fixed in an earlier session), but never teaches
  resolving a RELATIVE time word inside the turn's own text against that
  prefix. Added an explicit instruction plus one worked example (prefix
  "29 January, 2023" + "went to Paris yesterday" -> resolved to "28
  January, 2023").

  Validated on the same Kaggle run against 6 real relative-time turns
  pulled live from the actual LoCoMo dataset (containing "yesterday",
  "last week", or "next month" alongside a date prefix), regenerating
  both the old and new extraction prompts back-to-back in the same
  session for a clean comparison (no old-run/new-run confound here, unlike
  the fix above). Result: the new prompt resolved **zero** of the five
  cases the old prompt didn't already handle -- most turns describing
  ongoing or future plans ("next month") were correctly left unresolved by
  both (there is no fixed date to resolve to), and most past-tense "last
  week" turns were left with "last week" still literally in the output by
  *both* prompts. Worse, on one turn the new prompt actively regressed:
  instead of summarizing ("Melanie and Caroline have been friends for 5
  years", what the old prompt produced), it dumped the entire raw turn
  text verbatim as the extracted "content", relative-time word and all.
  Reverted `EXTRACTION_PROMPT` to the original wording. Category 2's
  relative-date gap is real and confirmed, but resolving prose-relative
  dates via prompt instruction alone was apparently too hard an ask for
  Qwen3-4B -- a real fix would likely need either a bigger model for
  extraction specifically, or a deterministic post-processing pass (regex
  + date arithmetic) rather than relying on the LLM to do the subtraction.

- **Tried and reverted: near-duplicate episodic memory supersession at
  storage time -- broke a real, existing test on the first attempt.**
  Third category-2 contributor found in the analysis: the same recurring
  event gets restated across a long conversation with DIFFERENT extracted
  dates each time -- real example: "Jon lost his job on 20 January" / "9
  April" / "9 July" all stored as three separate episodic memories, none
  matching the real gold date (19 January) -- so retrieval surfaces
  several competing near-duplicates together with no signal for which
  date the question actually wants.

  Tried: at `_add_memory` time, deleting an existing episodic memory
  whenever new content fell within a high cosine-similarity threshold of
  it (`NEAR_DUPLICATE_DISTANCE_THRESHOLD`), mirroring Zep-style's own
  deterministic edge-invalidation-on-contradiction (see
  `baselines/zep_baseline.py`) applied to this pipeline's free-text
  episodic store instead of a structured subject/predicate/object graph.

  Reverted immediately on the first local test run:
  `test_decay_lambda_override_prevents_premature_forgetting_at_scale`
  ingests 400 short, structurally-templated-but-genuinely-distinct events
  ("the weather was mild on day 0" / "day 1" / "day 2" / ...), and the
  dedup logic collapsed them down to 54 -- treating each day's entry as a
  "restatement" of the previous one. This confirmed the real problem, not
  a contrived one: differing by exactly one token (a date, or here a day
  number) is precisely what makes two sentences embed close together via
  all-MiniLM-L6-v2, regardless of whether they're a genuine restatement or
  two entirely independent events sharing a sentence template -- the same
  sentence shape as the actual target problem (Jon's job-loss restatements
  also differ by exactly one token: the date). Pure content-embedding
  similarity cannot reliably tell these apart. A real fix here would need
  something more structured than raw text similarity -- e.g. extracting an
  explicit subject + event-type separate from the date, the way Zep-
  style's own subject/predicate/object extraction already does -- left as
  a documented open problem rather than shipped as a fix that provably
  deletes real, distinct memories. Reverted with a regression-guard test
  (`test_structurally_similar_but_distinct_episodic_memories_all_survive`)
  so it can't quietly come back without someone re-confirming it doesn't
  reproduce this exact failure.

- **Fixed: category 4's answer-verbosity problem, validated with a real
  F1 gain -- scoped to category 4 only, not shipped globally.** After the
  category 2/3 investigation above, root-caused category 4 next: it's the
  largest LoCoMo category by far (841/1540 -- 55% of the whole benchmark)
  and, on the confirmed 27.74% run, had the worst answer-length ratio of
  any category -- predictions averaged **2.38x** longer than gold (12.6
  vs. 4.8 words), worse than category 1 (1.68x), category 2 (2.06x), or
  category 3 (2.09x) -- even though `QA_SYSTEM_PROMPT` already asks
  abstractly for conciseness ("be concise -- one sentence or less").
  Near-zero breakdown: 47.3% genuine retrieval misses (gold never in
  context -- the single biggest share of any category analyzed so far,
  and a real, harder, separate problem left undone -- see below), 45.9%
  partial-context, only 6.8% gold-fully-in-context-but-still-wrong.
  Retrieval was never empty (0% of near-zero questions had zero retrieved
  memories) -- the failures are about which memories got retrieved and how
  they got answered, not a totally broken retrieval path.

  Root cause on the generation side: describing conciseness abstractly
  wasn't enough -- the model still defaulted to full explanatory sentences
  ("caroline is excited about the adoption process because she views it
  as a way of giving back...") instead of matching gold's terse phrase
  style ("creating a family for kids who need one"). Added
  `CATEGORY_4_QA_SYSTEM_PROMPT`, which keeps the same context-grounding
  instruction but adds three few-shot examples *demonstrating* the target
  terse shape (a short phrase or single word, no "because..." reasoning) --
  the same principle already validated for `EXTRACTION_PROMPT` (a 4B model
  follows a shown example far more reliably than an abstract rule).

  Validated properly this time: both the original prompt and the new one
  were regenerated fresh, back-to-back, in the same Kaggle session against
  an identical 260-question sample stratified across category 4's real F1
  distribution (150 near-zero / 40 low / 30 mid / 20 high / 20 already-
  correct) -- no old-run/new-run confound, unlike the category-3
  validation. Result: **0.181 -> 0.209 avg F1 (+15.5% relative)** on the
  sample, with predicted length dropping from 12.6 to 3.5 words, confirming
  the terseness instruction actually took effect. Per-bucket breakdown:

  | bucket (orig. F1 range) | n   | OLD avg F1 | NEW avg F1 | delta   |
  |--------------------------|-----|-----------|-----------|---------|
  | near-zero (<0.05)        | 150 | 0.000     | 0.036     | +0.036  |
  | low (0.05-0.25)          | 40  | 0.147     | 0.203     | +0.056  |
  | mid (0.25-0.50)          | 30  | 0.331     | 0.330     | -0.001  |
  | high (0.50-0.75)         | 20  | 0.595     | 0.709     | +0.114  |
  | top (0.75-1.00)          | 20  | 0.962     | 0.840     | -0.122  |

  Real, honest caveat from the sample validation: the "top" bucket
  regressed -- over-terseness sometimes drops a qualifying word gold
  actually needed, on questions the original prompt was already answering
  well. Weighting the sample's deltas by category 4's true bucket sizes in
  the full 841-question population projected a net category-4 gain of
  roughly +0.010 avg F1 (+0.006 overall) -- a real but modest estimate.

  **Confirmed on a full, official 1540-question run** (same production
  entry point, `scripts/run_full_locomo.py --system hippovoice`, that
  confirmed `top_k=10`): the real effect was notably larger than the
  sample projected -- category 4 moved **0.3139 -> 0.3456 avg F1
  (+3.16pp)**, overall **27.74% -> 29.47% (+1.73pp)**. Score-distribution
  bins moved favorably too (near-zero/partial/high: 877/431/232 ->
  852/410/278 -- fewer near-zero, fewer partial, more high-scoring
  answers, not just a mean shift). Categories 1, 2, 3, and 5 reproduced
  their exact prior F1 values bit-for-bit across the two independent full
  runs (0.2216 / 0.2640 / 0.1660 / 0.0000, unchanged to 4 decimal places)
  -- a clean, deterministic confirmation that `CATEGORY_4_QA_SYSTEM_PROMPT`
  is fully isolated to category 4 with zero cross-category side effects,
  not a coincidence given the underlying generation is greedy/deterministic
  and every other category's prompt and inputs were untouched.

  Deliberately scoped `CATEGORY_4_QA_SYSTEM_PROMPT` to `category == 4`
  only in the QA loop rather than replacing the shared `QA_SYSTEM_PROMPT`
  globally, for two reasons: (1) the top-bucket regression shows this
  specific prompt isn't a strict improvement even within category 4 --
  applying it to categories 1/2/3/5 without validating there first would
  repeat exactly the mistake the category-3 QA-prompt revert already
  taught (a prompt validated on one category's question shapes doesn't
  automatically transfer); (2) category 3 in particular is known to be
  unusually sensitive to QA-prompt wording changes. The 47.3% genuine-
  retrieval-miss share for category 4 is left as a documented, real, and
  harder open problem -- a generation-prompt fix can't recover an answer
  that was never retrieved in the first place; that would need retrieval-
  side work (embedding quality, hybrid search, chunk granularity), not
  attempted this round given the size of that undertaking relative to the
  time budget.

- **Fixed: category 4's genuine retrieval misses, via BM25 keyword-match
  seeding.** The 47.3% genuine-retrieval-miss item directly above was
  picked up as real retrieval-side work rather than left open. Added a
  full-store BM25 scan (`_bm25_seed_ids` in `memory/retriever.py`) as a
  second seeding path into `hippo_retrieve()`, structurally identical to
  the existing `_name_match_ids` mechanism already solving the same class
  of problem for proper nouns (the "Jon"/"John" disambiguation fix) --
  generalized from exact name matches to any keyword/number/date overlap
  via standard Okapi BM25. `BM25_MATCH_BONUS` deliberately reuses
  `NAME_MATCH_BONUS`'s exact magnitude (0.3) rather than introducing a
  second new free parameter alongside a new mechanism. Shared by both
  `HippoVoicePipeline` and `HippoAudioPipeline`'s episodic retrieval (both
  call the same `hippo_retrieve()`), so both tracks get this fix at once.
  4 new tests, full suite clean. **Not yet confirmed with a real F1
  measurement** -- unit-tested and logically sound (mirrors an
  already-validated mechanism), but per this project's own standing
  discipline (see the category-3/2 QA-prompt reverts above), a real
  before/after LoCoMo run is still needed before trusting the size of the
  effect, only the mechanism itself.

- **Fixed: Zep-style edge invalidation missed contradictions phrased with
  different predicates.** Caught by `zep_sanity_check` exactly as that
  check is designed to -- before spending GPU time on a full run, not
  after. Real Qwen3-4B extraction of "Caroline lives in Seattle" then
  "Caroline moved to Portland" left both facts live (the sanity check
  expects 1), because the deterministic invalidation rule required an
  EXACT string match on predicate, and the LLM phrased the same
  real-world update with two different verbs. The existing unit test
  never caught this because its own mock happened to reuse the identical
  predicate string for both turns. Fixed by comparing predicates on
  embedding similarity instead of string equality -- checked the actual
  threshold against real measured values rather than assuming
  `ENTITY_RESOLUTION_THRESHOLD` (0.75) transfers to short verb phrases (it
  doesn't: "lives in" vs "moved to" scores 0.604, "likes" vs "dislikes"
  --should NOT invalidate-- scores 0.557, only a 0.033 gap from "likes"
  vs "prefers" --should invalidate-- at 0.590). Set
  `PREDICATE_SIMILARITY_THRESHOLD = 0.6`: clears the two confirmed real
  cases this fixes with real margin above the highest confirmed
  false-positive risk, at the honest, documented cost of still missing
  some genuine updates phrased very differently -- a known limitation,
  not a regression (those cases behave exactly as they did before this
  fix). 2 new tests, full suite clean.

## Open

- **Category 4's genuine retrieval misses -- BM25 seeding fix landed
  (mechanism above), full-run confirmation still pending.** The mechanism
  is real and unit-tested, but the actual F1 effect on the full 1540-
  question set hasn't been measured yet -- next step is a full
  `run_full_locomo.py --system hippovoice` confirmation run, the same way
  `top_k=10` and the category-4 QA-prompt fix were each confirmed for
  real before being trusted.
- **NaiveRAG confirmed at 33.9% avg F1 (top_k=10) -- higher than
  HippoVoice's 29.47%, for an explainable reason, not a refutation.** It
  never forgets anything (no decay, no salience), so on a small, static,
  10-conversation benchmark that never grows large enough to punish an
  unbounded store, pure recall wins -- exactly the regime this benchmark
  tests. The noise-contamination table (10% for HippoVoice vs. 30% for
  naive/Mem0-style retrieval) is the sharper comparison for what managed
  memory actually buys: comparable-or-better recall at a third of the
  noise, not "wins every metric." Mem0-style and A-MEM-style also skip
  real forgetting, making them the more relevant comparison point than a
  strawman with zero memory management at all.

- **Zep-style and Mem0-style (top_k=10 rerun) are both real, long-running
  jobs that hit a genuine Kaggle platform constraint: the free-tier
  weekly GPU quota (30 hours) got exhausted mid-run, cancelling BOTH
  concurrently-running kernels at the same instant and then refusing new
  pushes outright ("Maximum weekly GPU quota of 30.00 hours reached").**
  Real measured per-turn costs explain why these two specifically are the
  expensive ones: Mem0-style ~9-10s/turn (LLM-based ADD/UPDATE/DELETE/NOOP
  decision every turn), Zep-style ~20s/turn (entity+fact extraction plus
  entity-resolution embedding calls every turn) -- against ~5,500 total
  turns across the full 10-conversation set, Zep-style alone extrapolates
  to ~25-30 hours, comfortably past a single Kaggle session's ~12h cap
  even before the weekly quota is considered.

  Real resume mechanism now wired in, not just "restart and hope": each
  cancelled run's checkpoint (`locomo_checkpoint_full_<system>.json`,
  written by `run_locomo()`'s own existing checkpoint/resume logic) does
  survive into that kernel's Output tab despite the cancellation --
  confirmed by downloading it directly (Mem0 at 1,035/1,540 questions,
  Zep at 584/1,540). The reason resume wasn't happening automatically
  across separate kernel pushes is that `/kaggle/working` doesn't persist
  between distinct kernel version runs -- each fresh push starts in a
  clean container with a fresh git clone, checkpoint included. Fixed the
  same way the original mem0 top_k=5 run's own real Kaggle cancellation
  was handled: uploaded each checkpoint as its own private Kaggle Dataset
  (`hippovoice-mem0-checkpoint-resume`, `hippovoice-zep-checkpoint-resume`),
  wired as a `dataset_sources` entry in each kernel's metadata, and added
  a copy-if-not-already-present step to each kernel script (copies from
  `/kaggle/input/<dataset>/...` to the exact path `run_locomo()` expects,
  only when that path doesn't already exist, so a genuinely fresh run is
  never silently short-circuited). Verified the fingerprint will actually
  match on resume: both checkpoints' recorded commit (`717fbf9`) equals
  the current repo HEAD, and no decay_lambda/relevance_weight/top_k/model
  values have changed since either checkpoint was written.

  **Blocked until the weekly GPU quota resets** -- pushing either kernel
  right now fails immediately with the quota error above, before even
  reaching the resume logic. Nothing is lost in the meantime: both
  checkpoints are safely preserved as Kaggle Datasets independent of
  quota state, ready to resume the moment a push succeeds again.
- Weight-editing (ROME/MEMIT) still needs wiring into
  `scripts/run_full_locomo.py`'s `SYSTEMS` dict -- `kaggle_weightedit_only.
  ipynb` has a real, separate setup for it (different base model, GPT-2
  XL, not Qwen3-4B) that hasn't been checked yet for whether it plugs into
  the shared harness as-is or needs its own runner.
- Categories 1 (multi-hop, 22.2% avg F1) and 2's remaining relative-date
  gap haven't had the same kind of dedicated root-cause pass category 4
  just got -- category 1 in particular is unexamined beyond its aggregate
  F1 number.
- Track 2 (audio) has never had the decay_lambda/relevance_weight/top_k
  sweep Track 1 got, and its own 25.81% benchmark only covered 2 of
  LoCoMo's 10 conversations -- scaling to the full set would make it
  directly comparable to Track 1's baselines.

## Open — carried over from earlier session (context.md)

- **Resolved, stale note removed:** this used to say `colab.ipynb`'s header
  table claimed Qwen3-4B while the "Load LLM" cell's bare `LLMClient()`
  defaulted to Qwen3-0.6B. Checked directly against the current code
  (`llm/client.py`): `LLMClient.__init__`'s own default is now
  `"Qwen/Qwen3-4B"` (and `"mlx-community/Qwen3-4B-4bit"` on the MLX path) --
  the two already match, nothing to fix.
- **Resolved, stale note removed:** this used to say Track 2 (audio-space
  memory) was entirely unbuilt. It's since been built end to end --
  `pipeline_audio2audio.py`, a confirmed live audio-in/audio-out recall
  demo on a real GPU, a 40-question text-out LoCoMo benchmark (25.81% avg
  F1), and an interactive Gradio demo (`demo/track2_app.py`) verified live
  against the real Qwen2.5-Omni backend. See README's Status section and
  the chapters above for the real trail.
- `benchmarks/longmemeval/` still doesn't exist -- genuinely still open, a
  second long-term-memory benchmark dataset beyond LoCoMo that nothing in
  this project has started on yet.
