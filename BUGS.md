# Known bugs / improvements — running list

Tracked here instead of fixed immediately so a first full real (non-mocked)
Colab run can surface as many issues as possible before we spend time tuning.
Add to this list; don't fix silently in passing.

## Fixed

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
