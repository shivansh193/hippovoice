# Known bugs / improvements — running list

Tracked here instead of fixed immediately so a first full real (non-mocked)
Colab run can surface as many issues as possible before we spend time tuning.
Add to this list; don't fix silently in passing.

## Fixed

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
