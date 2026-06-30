"""
Signal/noise benchmark — the core empirical claim.

45-turn synthetic conversation:
  odd turns  → signal (fear/sadness, intensity 0.7–0.95)
  even turns → noise  (neutral, intensity 0.01)

After 45 turns, query for "what important things happened to the user recently?"
and measure what fraction of the top-10 retrieved memories are noise.
"""

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


def run_signal_noise_benchmark(pipeline, system_name: str) -> dict:
    """
    Ingest 45 turns (alternating noise/signal starting with noise at turn 0),
    then retrieve top-10 and measure noise contamination.

    Returns:
        {
            "system": str,
            "noise_rate": float,       # fraction of top-10 that are noise
            "signal_count": int,
            "noise_count": int,
            "retrieved": list[dict],
        }
    """
    # Build the 45-turn script: noise on even indices (0,2,4…), signal on odd (1,3,5…)
    turns = []
    noise_iter = iter(NOISE_TURNS)
    signal_iter = iter(SIGNAL_TURNS)

    for i in range(44):
        if i % 2 == 0:
            t = next(noise_iter, None)
        else:
            t = next(signal_iter, None)
        if t:
            turns.append(t)

    # Final turn is always a signal turn (turn 44 = even → noise pattern ends with signal)
    turns.append(next(signal_iter, SIGNAL_TURNS[-1]))

    # Ingest all turns
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
        "retrieved": retrieved,
    }
