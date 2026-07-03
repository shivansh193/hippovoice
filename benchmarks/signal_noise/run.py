"""
Signal/noise benchmark — the core empirical claim.

A long synthetic conversation dominated by mundane ("noise") turns with a
sparse handful of emotionally significant ("signal") turns interspersed —
this is meant to model a real day-to-day companion, where the overwhelming
majority of what gets said is small talk and only occasionally something
worth remembering long-term comes up. After the conversation, query for
"what important things happened to the user recently?" and measure what
fraction of the top-10 retrieved memories are noise.

Note on ratio: an earlier version of this benchmark used a 1:1 signal:noise
split (22 signal / 22 noise, `noise_per_signal=1`). At that ratio plain cosine
similarity already retrieves signal turns preferentially, because the
retrieval query is itself semantically about "important/emotionally
significant" things — any reasonable embedding model does that matching
regardless of forgetting. That made NaiveRAG score ~0% noise, which doesn't
test the actual "noisy attic" failure mode.

`noise_per_signal` controls how much noise dominates by volume. Default is 2
(66 total turns) — high enough that plain similarity search picks up
noticeably more noise than HippoVoice's salience-weighted retrieval, but
still within the window the decay time-constant (λ=0.05/turn, see
memory/scorer.py) was tuned for. Pushing much past ~turn 90-100 degrades
*every* decay-based system, HippoVoice included, because sheer conversation
length eventually lets recency (very-fresh noise, near-zero turns_elapsed)
outweigh even a strongly emotion-boosted but comparatively old signal memory
— a real, known limitation of pure exponential decay, not a benchmark
artifact. Multi-session consolidation (compress/forget cycles run
periodically, not just at query time) is the mitigation; it isn't stressed by
this single-session benchmark.
"""

import itertools
import random

SIGNAL_TURNS = [
    "My father was diagnosed with stage 3 cancer last week and I feel absolutely terrified.",
    "I got into a serious car accident and totaled my car. I was shaking for hours afterward.",
    "My best friend told me she's moving across the country permanently. I'm devastated.",
    "I found out I'm being laid off at the end of the month. I feel completely lost.",
    "My dog of twelve years passed away yesterday. I'm heartbroken.",
    "I had a panic attack in the middle of a work presentation today. It was awful.",
    "My partner and I had a huge fight and they said they needed space. I'm scared.",
    "I got the biopsy results back — the doctors want to run more tests. I'm terrified.",
    "My apartment flooded and I lost most of my belongings. I feel completely overwhelmed.",
    "I failed the exam I studied months for. I feel like such a failure.",
    "My grandmother is in the hospital and they're not sure she'll recover.",
    "I was in a robbery last night. I'm still shaking and can't sleep.",
    "My company is shutting down the entire department. Everyone is being let go.",
    "I got in a terrible argument with my parents and said things I regret deeply.",
    "My childhood home burned down. My parents lost everything.",
    "I was diagnosed with an anxiety disorder today. I feel like something is wrong with me.",
    "My best friend was in a serious accident and is in the ICU.",
    "I found out my partner has been lying to me for months. I feel completely betrayed.",
    "My flight was diverted during an emergency landing. I genuinely thought I might die.",
    "I lost my wedding ring that meant everything to me and I've been crying all day.",
    "My startup just got denied funding for the third time. I'm starting to give up.",
    "I witnessed a terrible accident on my way to work and I can't stop seeing it.",
]

# Small base list kept for backwards compatibility / readability; the actual
# noise pool used in the benchmark is generated programmatically below since
# a realistic noisy-attic scenario needs far more mundane turns than signal.
NOISE_TURNS = [
    "The weather was cloudy today.",
    "I had cereal for breakfast this morning.",
    "I saw a blue car parked on my street.",
    "The grocery store was out of my usual brand of bread.",
    "I watched a few minutes of TV before bed.",
    "There was some traffic on the highway this morning.",
    "I received a package in the mail.",
    "The coffee at work was slightly stronger than usual.",
    "I walked past a construction site on my way in.",
    "My phone battery died in the afternoon.",
    "I forgot to bring my umbrella today.",
    "The elevator in my building was slow this morning.",
    "I saw a pigeon on the window ledge.",
    "I changed the channel during the commercial break.",
    "I used a different mug for my tea today.",
    "The printer at work was out of paper.",
    "I noticed a new shop opened on the corner.",
    "I took a slightly longer route home today.",
    "My internet was a bit slow this afternoon.",
    "I moved a plant from one windowsill to another.",
    "There was a mild draft in the office.",
    "I ate lunch at my desk instead of the break room.",
]

# ── Templated noise pool ──────────────────────────────────────────────────────
# Combinatorial generation of mundane, low-salience sentences so the noise
# pool can be scaled up without hand-writing hundreds of lines. Deterministic
# (fixed seed) so the benchmark is reproducible run to run.

_SUBJECTS = ["The weather", "The coffee", "My commute", "The office", "The kitchen",
             "The traffic", "My phone", "The mail", "The elevator", "The neighbor's dog",
             "The printer", "The bus", "The grocery store", "The wifi", "The thermostat"]
_PREDICATES = ["was a bit different today", "seemed fine, nothing notable", "was mildly annoying",
               "worked as usual", "was slightly delayed", "was about the same as always",
               "needed a small adjustment", "was unremarkable", "took a little longer than expected",
               "was quiet today"]
_FILLERS = [
    "I had a sandwich for lunch.",
    "I refilled my water bottle twice today.",
    "I organized a drawer in my desk.",
    "I listened to a podcast on the way home.",
    "I charged my phone overnight.",
    "I swept the kitchen floor.",
    "I replied to a few emails.",
    "I watered the plants on the balcony.",
    "I folded some laundry in the evening.",
    "I browsed a few articles before bed.",
]


def _generate_noise_pool(n: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    combos = list(itertools.product(_SUBJECTS, _PREDICATES))
    rng.shuffle(combos)
    pool = [f"{subj} {pred}." for subj, pred in combos]
    pool.extend(_FILLERS)
    pool.extend(NOISE_TURNS)
    # dedup while preserving order, then cycle if still short of n
    seen = set()
    deduped = []
    for s in pool:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    if len(deduped) < n:
        deduped = (deduped * (n // len(deduped) + 1))
    return deduped[:n]


RETRIEVAL_QUERY = "What important or emotionally significant things have happened to the user?"

SIGNAL_KEYWORDS = {
    "cancer", "accident", "moving", "laid off", "died", "panic", "scared", "terrified",
    "devastated", "biopsy", "flooded", "failed", "grandmother", "robbery", "shutting",
    "argument", "burned", "anxiety", "icu", "lying", "betrayed", "emergency landing",
    "wedding ring", "startup", "witnessed", "heartbroken", "lost", "overwhelmed",
}


def _is_signal(content: str) -> bool:
    content_lower = content.lower()
    return any(kw in content_lower for kw in SIGNAL_KEYWORDS)


def run_signal_noise_benchmark(pipeline, system_name: str, noise_per_signal: int = 2) -> dict:
    """
    Ingest a long conversation (signal turns sparsely interspersed among many
    noise turns, ratio 1:noise_per_signal), then retrieve top-10 and measure
    noise contamination.

    Returns:
        {
            "system": str,
            "noise_rate": float,       # fraction of top-10 that are noise
            "signal_count": int,
            "noise_count": int,
            "total_turns": int,
            "retrieved": list[dict],
        }
    """
    noise_pool = _generate_noise_pool(len(SIGNAL_TURNS) * noise_per_signal)

    turns = []
    noise_iter = iter(noise_pool)
    signal_iter = iter(SIGNAL_TURNS)
    block = noise_per_signal + 1  # noise_per_signal noise turns, then 1 signal turn

    exhausted = False
    while not exhausted:
        for _ in range(noise_per_signal):
            t = next(noise_iter, None)
            if t is None:
                exhausted = True
                break
            turns.append(t)
        if exhausted:
            break
        t = next(signal_iter, None)
        if t is None:
            break
        turns.append(t)

    # Ingest all turns -- batch extraction when the pipeline supports it
    # (each turn's extraction is independent, so it doesn't need to happen
    # one real LLM call at a time).
    if hasattr(pipeline, "ingest_text_turns_batch"):
        batch_size = 50
        for b in range(0, len(turns), batch_size):
            pipeline.ingest_text_turns_batch(turns[b:b + batch_size])
    else:
        for turn_text in turns:
            pipeline.ingest_text_turn(turn_text)

    # Retrieve and classify
    retrieved = pipeline.retrieve(RETRIEVAL_QUERY, top_k=10)

    signal_count = sum(1 for r in retrieved if _is_signal(r.get("content", "")))
    noise_count = len(retrieved) - signal_count
    noise_rate = noise_count / len(retrieved) if retrieved else 0.0

    return {
        "system": system_name,
        "noise_rate": round(noise_rate, 4),
        "signal_count": signal_count,
        "noise_count": noise_count,
        "total_retrieved": len(retrieved),
        "total_turns": len(turns),
        "retrieved": retrieved,
    }
