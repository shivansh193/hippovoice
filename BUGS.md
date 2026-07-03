# Known bugs / improvements — running list

Tracked here instead of fixed immediately so a first full real (non-mocked)
Colab run can surface as many issues as possible before we spend time tuning.
Add to this list; don't fix silently in passing.

## Fixed

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

## Open — found on first real (non-mocked, T4) Colab run

- **HippoVoice ties NaiveRAG and loses to Mem0-style on the signal/noise
  benchmark with a real LLM** (HippoVoice 30% noise vs Mem0-style 20%, vs a
  passing/deterministic-mock local test suite where HippoVoice beats both).
  The two fixes above target the most likely root causes (compress recency
  reset, graph-expansion contamination) but haven't yet been re-verified
  against a real Colab run — only against the local CPU test suite with the
  deterministic `passthrough_llm` mock. **Next step: rerun the signal/noise
  cell on Colab and confirm HippoVoice actually beats Mem0-style now.** If it
  still loses, print the actual `retrieved` list (content + current_salience)
  to see exactly which noise items are winning and why — recency-vs-decay
  dynamics at this conversation length (66 turns) are the leading suspect,
  documented in `benchmarks/signal_noise/run.py`'s module docstring as a
  known limitation of pure exponential decay past ~90-100 turns; it may be
  showing up earlier than expected.
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
