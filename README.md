# HippoVoice

A memory system for voice AI companions, built around two ideas borrowed
from how human memory actually works: Ebbinghaus-style decay (things fade
unless they matter or get recalled again) and HippoRAG-style graph
retrieval. The goal is memory that behaves less like a vector database
and more like a person's: a fact someone mentioned once sticks around,
what they had for breakfast three weeks ago doesn't, and what actually
gets pulled up when you ask a question is whatever's relevant, not
whatever was said most recently.

Benchmarked on real data. [LoCoMo](https://github.com/snap-research/locomo)
has 10 long-term conversations, 369-689 turns each, with real ground-truth
QA pairs. It's the same dataset Mem0, A-MEM, and MemoryBank report on.

## Results

**LoCoMo**, scored with the published methodology (stemmed token F1,
category-branched, not a rough approximation of it):

| System | avg F1 | Questions |
|---|---|---|
| HippoVoice | 24.1% | 1540 (10 conversations, all QA pairs) |
| Mem0-style | 23.4% | 1540 (10 conversations, all QA pairs) |
| A-MEM-style | 22.0% | 1540 (10 conversations, all QA pairs) |

HippoVoice's run finished 2026-07-11; Mem0-style's finished 2026-08-31;
A-MEM-style finished 2026-08-31 as well. Same harness, same LLM, same
scorer across all three — HippoVoice currently leads both baselines on
the actual LoCoMo metric, though the gap to Mem0-style is close (0.7
points) and the error-distribution shape is similar across all three
(near-zero/partial/high bins: 966/379/195 HippoVoice, 973/380/187
Mem0-style, 1055/287/198 A-MEM-style). NaiveRAG is the last one still
queued — check [BUGS.md](BUGS.md) for what's actually confirmed versus
what's pending.

There's also a smaller, synthetic benchmark (~90-100 turn conversations)
for noise contamination: what fraction of retrieved context is actually
irrelevant.

| System | Noise rate |
|---|---|
| HippoVoice | 10% (was 20% before a decay/relevance fix) |
| Mem0-style | 30% |
| NaiveRAG | 30% |
| A-MEM-style | 10% |

Worth being precise about this one: the Mem0/NaiveRAG/A-MEM numbers were
measured against HippoVoice's earlier 20% result, not its current 10%.
Baselines weren't rerun after the fix that got HippoVoice to 10%, so
don't read that row as five numbers from one simultaneous run.

BUGS.md has the full history: what got tried, what broke, how each
number was actually verified rather than assumed. It reads more like a
lab notebook than a changelog, on purpose — the dead ends are as
informative as the wins, and I didn't want to clean them out just to
make the project look tidier than the work actually was.

## Architecture

The store is split in two, and that split is the main architectural
decision in the codebase.

**Semantic memory** holds facts, preferences, and people. It doesn't
decay. "User's dog is named Max" isn't less true a month from now.

**Episodic memory** holds specific events. It decays the way Ebbinghaus
described: salience drops over turns unless something boosts it, either
emotional intensity or getting recalled again later.

An earlier version ran everything through one decaying store, and the
forgetting cycle meant to fade out stale episodic detail was quietly
deleting durable facts along with it. That's backwards. A person doesn't
forget a friend's name at the same rate they forget the exact sentence
that friend used to introduce themselves. Splitting the store by
extracted type fixed it without softening decay for the stuff that
should genuinely fade.

Episodic retrieval walks the graph HippoRAG-style, then reranks
candidates on a blended score:

```
relevance_weight * relevance + (1 - relevance_weight) * availability
```

Availability is the memory's current salience, log-normalized against a
forget threshold. I tried two simpler approaches before landing here:
raw unnormalized blending (availability spans something like 13 orders
of magnitude, so it swamps relevance completely) and reciprocal rank
fusion (works until the candidate pool is small, then rank 0 vs. rank 1
barely means anything). Both are written up in BUGS.md with the numbers
that killed them.

Semantic candidates get scored on relevance alone, no availability term.
A durable fact's relevance to a query has nothing to do with when it was
learned. I found this out the hard way: blending in a constant
availability term for semantic hits gave every one of them a flat bonus
regardless of relevance, and noise rate jumped to 90% in testing before
I caught it and reverted.

Name matching needed its own fix. Embedding similarity alone can't
reliably separate two similarly-named people in a long conversation.
Confirmed this directly: content about "Jon" was getting pulled in by
questions about "John," a different person entirely. The fix scores an
exact, case-sensitive proper-noun match as its own signal, and it needed
a full-store scan feeding into the candidate pool before reranking even
happens. The scoring bonus alone wasn't enough. A correctly-named memory
that never made the initial similarity-based seed pool never gets to
reranking to be promoted by that bonus.

Decay rate isn't one fixed number either. A `decay_lambda` tuned for a
90-100 turn conversation forgets nearly every episodic memory before a
369-689 turn LoCoMo conversation is even halfway done, regardless of how
good extraction or retrieval are. Both `decay_lambda` and
`relevance_weight` are exposed as overridable parameters with separate
defaults for LoCoMo scale versus the pipeline's normal short-conversation
defaults.

Extraction runs one LLM call per turn, converting raw conversation text
into typed fragments (fact, preference, person, event). The prompt took
several real iterations to get right, swinging between over-extracting
filler and under-extracting actual facts before settling. BUGS.md has
that whole saga if you want to see how much trial and error went into
what looks like one paragraph of instructions.

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

Track 1 (text) is validated on real LoCoMo data at 24.1% avg F1. Baseline
comparisons and further tuning are ongoing.

Track 2 (audio-to-audio + memory) has a real, confirmed win: on a live
multi-turn run through `HippoAudioPipeline` and Gemini's Live API, a fact
stated in turn 1 got correctly recalled in turn 3. Gemini Live's own API
latency (~35-45s/question) made a larger benchmark impractical there, so
a second, self-hosted backend (`Qwen25OmniAudioModel`, Qwen2.5-Omni-3B)
was validated instead: after fixing four real bugs surfaced only by
running the full pipeline on a GPU (see `BUGS.md`), a complete 40-question
LoCoMo run finished with **25.81% avg F1** -- comparable to Track 1's own
text-only Mem0-style baseline (23.4%) despite going through a full
TTS -> Qwen2.5-Omni -> transcript round trip rather than reading text
directly.

The weight-editing baseline (ROME/MEMIT on GPT-2 XL) is built and being
benchmarked as an alternative to retrieval-based memory. Worth reading
`baselines/weight_edit_baseline.py`'s docstring before assuming it'll
compete: it's expected to degrade at LoCoMo's scale, and that's a real
result worth having, not a failure of the benchmark.

Every number above traces back to an actual run logged in `BUGS.md`.
Nothing here is projected or aspirational.
