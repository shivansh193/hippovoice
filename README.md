# HippoVoice

A biologically-inspired memory system for voice AI companions — Ebbinghaus-style
decay plus HippoRAG-style graph retrieval, built to give a conversational agent
memory that behaves more like a person's than a vector database's: durable facts
stick around indefinitely, episodic details fade unless they're emotionally
salient or get recalled again, and retrieval favors what's actually relevant
over whatever was said most recently.

Benchmarked on real data, not a synthetic toy: [LoCoMo](https://github.com/snap-research/locomo)
(10 long-term conversations, 369-689 turns each, real ground-truth QA pairs) —
the same dataset Mem0, A-MEM, and MemoryBank report on.

## Results

**LoCoMo (real benchmark, published F1 scoring methodology — stemmed token F1,
category-branched):**

| System | avg F1 | Questions |
|---|---|---|
| **HippoVoice** | **24.1%** | 1540 (10 conversations, all QA pairs) |

Confirmed on a full run, 2026-07-11. Baseline comparisons (Mem0-style,
A-MEM-style, NaiveRAG) run through the identical harness/LLM/scoring are
in progress — see [BUGS.md](BUGS.md) for exact provenance and what's
confirmed vs. still running.

**Signal/Noise benchmark (a separate, synthetic ~90-100 turn benchmark
measuring what fraction of retrieved context is irrelevant noise):**

| System | Noise rate |
|---|---|
| **HippoVoice** | **10%** (down from an earlier 20% after decay/relevance fixes) |
| Mem0-style | 30% |
| NaiveRAG | 30% |
| A-MEM-style | 10% |

(Mem0-style/NaiveRAG/A-MEM-style numbers are from the same measured run as
HippoVoice's 20% figure — see `BUGS.md` for the exact commit. HippoVoice's
noise rate improved further to 10% in a later fix; baselines weren't
re-measured at that exact point, so treat 30%/30%/10% as the last confirmed
baseline snapshot, not necessarily concurrent with HippoVoice's most recent
number.)

Full history of what was tried, what broke, and how each result was
verified lives in [BUGS.md](BUGS.md) — kept as a running log rather than
cleaned up, since the false starts (and how they were caught) are as much
the point as the final numbers.

## Architecture

Two structurally distinct memory stores, not one undifferentiated vector
index — this split is the core architectural bet of the project:

- **Semantic memory** (`fact` / `preference` / `person` types): durable
  knowledge that doesn't decay. "User's dog is named Max" doesn't become
  less true or less retrievable as time passes.
- **Episodic memory** (`event` type): specific occurrences, subject to
  Ebbinghaus-style forgetting — salience decays over turns unless boosted
  by emotional intensity or refreshed by being recalled again.

**Why split at all**: an early version routed everything through one
decaying store, and durable facts got physically deleted by the same
forgetting cycle designed for fading episodic detail — a person doesn't
forget their friend's name at the same rate they forget the specific
sentence used to state it. Splitting by extracted type fixed this without
weakening decay for what should genuinely fade.

**Retrieval** (episodic): HippoRAG-style graph-walk expansion over
candidate memories, reranked by a blended score —
`relevance_weight * relevance + (1 - relevance_weight) * availability`,
where availability is the memory's current Ebbinghaus salience,
log-normalized against a forget threshold. Two simpler designs (raw
unnormalized blending, reciprocal rank fusion) were tried and empirically
rejected first — see `BUGS.md` for why. Semantic candidates are scored by
relevance alone (no availability term), since a durable fact's relevance
to a query doesn't depend on how long ago it was learned; blending in a
constant availability term for these turned out to give every semantic-
store hit a flat, unconditional bonus and blew up noise rate to 90% in
testing before being caught and reverted.

**Name-match disambiguation**: pure embedding similarity can't reliably
tell two similarly-named people apart in a long conversation (confirmed
directly: "Jon" and "John" pulled in each other's content). An exact,
case-sensitive proper-noun match is scored as an independent signal
alongside embedding relevance, with a full-store scan unioned into the
candidate pool before reranking — the scan step was necessary in addition
to the scoring bonus, since a correctly-named memory crowded out of the
initial similarity-based seed pool never reaches reranking to be promoted.

**Decay tuning is scale-dependent, not a fixed constant**: `decay_lambda`
tuned for ~90-100 turn conversations forgets almost every episodic memory
well before a 369-689 turn LoCoMo conversation ends, independent of
extraction or retrieval quality. Both `decay_lambda` and `relevance_weight`
are explicit, overridable parameters (not hardcoded), with LoCoMo-scale
defaults distinct from the pipeline's own short-conversation defaults.

**Extraction**: an LLM call per turn converts raw conversational text into
typed memory fragments (fact/preference/person/event), with a prompt
iterated multiple rounds against real model output — not assumed correct
from wording alone — after early versions oscillated between
over-extracting filler and under-extracting genuine facts. See `BUGS.md`
for the full iteration history.

## Repo layout

```
pipeline.py                    Track 1: text/cascaded pipeline (HippoVoicePipeline)
pipeline_audio2audio.py        Track 2: audio-to-audio pipeline (HippoAudioPipeline)
gemini_live_model.py           Real audio-to-audio backend (Gemini Live API)
memory/
  store.py                     Chroma-backed vector store + graph
  scorer.py                    Salience/decay math
  decay.py                     Forgetting/compression cycle
  retriever.py                 HippoRAG-style graph-walk retrieval + reranking
  extractor.py                 LLM-based turn -> typed memory extraction
baselines/                     Mem0-style, A-MEM-style, NaiveRAG, ROME/MEMIT weight-editing
benchmarks/locomo/             Real LoCoMo dataset loading + F1 scoring + eval harness
colab.ipynb                    Track 1 GPU runner (LoCoMo, signal/noise, baselines)
colab_track2.ipynb             Track 2 exploratory pass (memory capture, pre-conditioning)
kaggle_full_benchmark.ipynb    Weight-editing + Track 2 audio LoCoMo benchmarks
```

## Status

- **Track 1 (text)**: validated on real LoCoMo data — 24.1% avg F1. Baseline
  comparisons and further tuning ongoing.
- **Track 2 (audio-to-audio + memory)**: real memory *conditioning*
  confirmed on a live multi-turn run (a fact stated in turn 1 correctly
  recalled in turn 3, via `HippoAudioPipeline` + Gemini's Live API) — a
  full LoCoMo-scale benchmark is running now.
- **Weight-editing baseline (ROME/MEMIT on GPT-2 XL)**: built, being
  benchmarked now as an alternative to retrieval-based memory — see
  `baselines/weight_edit_baseline.py`'s module docstring for why this is
  expected to degrade at LoCoMo's scale, and why that's a legitimate
  result rather than a failed benchmark.

Everything above is backed by a real run cited in `BUGS.md`, not aspirational.
