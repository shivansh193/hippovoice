# HippoVoice Track 1 — Implementation Plan

## Project Structure

```
hippovoice/
├── stt/                    # Canary-Qwen STT
├── memory/                 # Hippocampal memory layer
│   ├── extractor.py        # Memory extraction + emotion tagging
│   ├── store.py            # ChromaDB + NetworkX graph
│   ├── scorer.py           # Salience scoring
│   ├── decay.py            # Forgetting cycle
│   └── retriever.py        # HippoRAG graph walk
├── llm/                    # Generation
├── tts/                    # Fish S2 Pro TTS
├── pipeline.py             # End-to-end orchestration
├── benchmarks/             # LoCoMo, LongMemEval, signal/noise
│   ├── locomo/
│   ├── longmemeval/
│   └── signal_noise/
├── baselines/              # Naive RAG, Mem0, A-MEM wrappers
└── tests/                  # Unit + integration tests
```

---

## Phase 0 — Environment Setup

### Task 0.1 — Python environment and dependencies

**Do:** Create `requirements.txt` and verify all packages install cleanly on Colab A100.

```
torch>=2.1.0
transformers>=4.40.0
nemo_toolkit[asr]>=1.23.0
chromadb>=0.4.0
networkx>=3.2
fastapi>=0.110.0
numpy>=1.26.0
scipy>=1.12.0
sentence-transformers>=2.7.0
```

**Expect:** `pip install -r requirements.txt` completes without conflicts.

**Verify:**
```python
import torch, transformers, chromadb, networkx
assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).total_memory > 35 * 1e9  # A100 > 35GB
print("GPU:", torch.cuda.get_device_name(0))
```
Pass: prints A100, no import errors.

---

### Task 0.2 — Project skeleton with CI-ready test runner

**Do:** Create `tests/` folder with `pytest.ini` and a smoke test.

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

```python
# tests/test_smoke.py
def test_imports():
    import torch, chromadb, networkx
    assert True
```

**Expect:** `pytest` runs and passes.

**Verify:** `pytest -v` → `1 passed`.

---

## Phase 1 — STT Module (Canary-Qwen)

### Task 1.1 — Load Canary-Qwen model

**Do:** Write `stt/model.py` that loads `nvidia/canary-qwen-2.5b` from HuggingFace and returns a transcriber object.

```python
# stt/model.py
from nemo.collections.asr.models import EncDecMultiTaskModel

def load_canary():
    model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-qwen-2.5b")
    model.eval()
    return model
```

**Expect:** Model loads in ~2-3 minutes, no CUDA OOM on A100 (2.5B params ≈ 5GB VRAM in fp16).

**Verify:**
```python
# tests/test_stt_load.py
def test_canary_loads():
    from stt.model import load_canary
    model = load_canary()
    assert model is not None
    param = next(model.parameters())
    assert param.is_cuda
```
Pass: model loads onto GPU without error.

---

### Task 1.2 — Transcribe a known audio file

**Do:** Write `stt/transcribe.py` with a `transcribe(audio_path: str) -> str` function. Use a 5-second synthetic WAV with known content as the test fixture.

```python
# stt/transcribe.py
def transcribe(model, audio_path: str) -> str:
    output = model.transcribe([audio_path], batch_size=1)
    return output[0] if output else ""
```

**Expect:** Given a WAV file saying "the cat sat on the mat", output is that string ± minor punctuation.

**Verify:**
```python
# tests/test_stt_transcribe.py
def test_transcribe_known_phrase(canary_model):
    audio_path = "tests/fixtures/cat_sat.wav"
    from stt.transcribe import transcribe
    result = transcribe(canary_model, audio_path)
    assert "cat" in result.lower()
    assert "mat" in result.lower()
```
Pass: key words present in transcription.

---

### Task 1.3 — Batch transcription + WER measurement

**Do:** Write `stt/benchmark.py` that runs on LibriSpeech test-clean (100 samples) and computes WER.

```python
# stt/benchmark.py
from jiwer import wer

def compute_wer(hypotheses: list[str], references: list[str]) -> float:
    return wer(references, hypotheses)
```

**Expect:** WER ≤ 6% on LibriSpeech test-clean (paper reports 5.63%).

**Verify:**
```python
# tests/test_stt_wer.py
def test_wer_within_bounds(canary_model, librispeech_100_samples):
    from stt.transcribe import transcribe
    from stt.benchmark import compute_wer
    hyps = [transcribe(canary_model, p) for p, _ in librispeech_100_samples]
    refs = [ref for _, ref in librispeech_100_samples]
    assert compute_wer(hyps, refs) < 0.07
```
Pass: WER < 7%.

---

### Task 1.4 — Extract prosody signals from encoder intermediate layer

**Do:** Modify `transcribe()` to also return the encoder's last hidden state mean-pooled per utterance (1280-dim vector). This is the acoustic embedding used in emotion tagging.

```python
# stt/transcribe.py
def transcribe_with_embedding(model, audio_path: str) -> tuple[str, np.ndarray]:
    transcript = transcribe(model, audio_path)
    embedding = extract_encoder_embedding(model, audio_path)  # shape: (1280,)
    return transcript, embedding
```

**Expect:** Returns (str, ndarray) where ndarray has shape (1280,) and is not all zeros.

**Verify:**
```python
# tests/test_stt_embedding.py
def test_embedding_shape_and_nonzero(canary_model):
    from stt.transcribe import transcribe_with_embedding
    text, emb = transcribe_with_embedding(canary_model, "tests/fixtures/cat_sat.wav")
    assert emb.shape == (1280,)
    assert np.any(emb != 0)
    assert not np.any(np.isnan(emb))
```
Pass: shape correct, non-zero, no NaNs.

---

### Task 1.5 — STT latency benchmark

**Do:** Measure wall-clock time for transcribing a 10-second audio clip.

**Expect:** < 2 seconds on A100 (real-time factor < 0.2).

**Verify:**
```python
# tests/test_stt_latency.py
import time

def test_transcription_latency(canary_model):
    from stt.transcribe import transcribe
    audio_10s = "tests/fixtures/ten_second_clip.wav"
    start = time.perf_counter()
    transcribe(canary_model, audio_10s)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"STT took {elapsed:.2f}s, target < 2.0s"
```
Pass: transcription completes in < 2s.

---

## Phase 2 — Memory Extraction + Emotion Tagging

### Task 2.1 — Memory extraction: identify memorable facts from a turn

**Do:** Write `memory/extractor.py` with `extract_memories(turn_text: str, llm_client) -> list[dict]`. Each memory is a dict with `content`, `entity`, `type` (fact/preference/event/person).

```python
EXTRACTION_PROMPT = """
Extract distinct, self-contained memory fragments from this conversation turn.
Each fragment should be a single fact, preference, or event worth remembering.
Return as JSON list: [{"content": "...", "entity": "...", "type": "fact|preference|event|person"}]

Turn: {turn}
"""
```

**Expect:** Given "I have a golden retriever named Max who loves swimming", returns at least two memories: one about having a dog, one about the dog's name.

**Verify:**
```python
# tests/test_extractor.py
def test_extracts_dog_facts(mock_llm):
    from memory.extractor import extract_memories
    turn = "I have a golden retriever named Max who loves swimming"
    memories = extract_memories(turn, mock_llm)
    contents = [m["content"].lower() for m in memories]
    assert any("dog" in c or "golden retriever" in c for c in contents)
    assert any("max" in c for c in contents)
    assert all(m["type"] in ["fact", "preference", "event", "person"] for m in memories)
```
Pass: dog and name both extracted, all types valid.

---

### Task 2.2 — Emotion tagging from text (baseline)

**Do:** Write `memory/extractor.py::tag_emotion_text(text: str) -> dict` using a small classifier. Returns `{"label": str, "intensity": float}`.

Valid labels: `neutral`, `joy`, `sadness`, `fear`, `anger`, `surprise`, `disgust`.
Intensity: 0.0–1.0.

**Expect:** "I'm so excited about my promotion!" → `{label: "joy", intensity: > 0.6}`. "The weather is okay." → `{label: "neutral", intensity: < 0.3}`.

**Verify:**
```python
# tests/test_emotion_text.py
def test_joy_detection():
    from memory.extractor import tag_emotion_text
    result = tag_emotion_text("I'm so excited about my promotion!")
    assert result["label"] == "joy"
    assert result["intensity"] > 0.6

def test_neutral_detection():
    from memory.extractor import tag_emotion_text
    result = tag_emotion_text("The weather is okay.")
    assert result["label"] == "neutral"
    assert result["intensity"] < 0.3
```
Pass: both assertions hold.

---

### Task 2.3 — Prosody-aware emotion tagging (voice-specific upgrade)

**Do:** Write `memory/extractor.py::tag_emotion_audio(text: str, audio_embedding: np.ndarray) -> dict`. Fuses text sentiment with audio embedding variance as a proxy for emotional speech energy.

```python
def tag_emotion_audio(text: str, audio_embedding: np.ndarray) -> dict:
    text_emotion = tag_emotion_text(text)
    audio_intensity_signal = float(np.std(audio_embedding))
    normalized_signal = min(audio_intensity_signal / 5.0, 1.0)

    if text_emotion["label"] == "neutral" and normalized_signal > 0.4:
        text_emotion["intensity"] = min(text_emotion["intensity"] + 0.2, 1.0)
        text_emotion["prosody_boosted"] = True

    return text_emotion
```

**Expect:** Identical text but high-variance audio embedding produces higher intensity than flat embedding.

**Verify:**
```python
# tests/test_emotion_audio.py
def test_prosody_boosts_neutral():
    from memory.extractor import tag_emotion_audio
    flat_emb = np.ones(1280) * 0.5
    loud_emb = np.random.randn(1280) * 2.0

    result_flat = tag_emotion_audio("I see", flat_emb)
    result_loud = tag_emotion_audio("I see", loud_emb)

    assert result_loud["intensity"] >= result_flat["intensity"]

def test_prosody_boost_flag_set():
    from memory.extractor import tag_emotion_audio
    loud_emb = np.random.randn(1280) * 3.0
    result = tag_emotion_audio("That's fine", loud_emb)
    if result.get("prosody_boosted"):
        assert result["intensity"] > 0.2
```
Pass: prosody signal influences intensity.

---

### Task 2.4 — Full extraction pipeline: text + audio → memory list with emotions

**Do:** Write `memory/extractor.py::extract_turn(turn_text, audio_embedding, llm_client) -> list[dict]`. Each memory gets an `emotion` field.

```python
def extract_turn(turn_text, audio_embedding, llm_client):
    memories = extract_memories(turn_text, llm_client)
    emotion = tag_emotion_audio(turn_text, audio_embedding)
    for m in memories:
        m["emotion"] = emotion
    return memories
```

**Expect:** Fearful statement produces memories all tagged with fear/sadness at intensity > 0.5.

**Verify:**
```python
# tests/test_extract_turn.py
def test_fearful_turn_emotion(mock_llm, sad_audio_embedding):
    from memory.extractor import extract_turn
    memories = extract_turn(
        "My dog Max got hit by a car today",
        sad_audio_embedding,
        mock_llm
    )
    assert len(memories) > 0
    for m in memories:
        assert m["emotion"]["label"] in ["fear", "sadness", "anger"]
        assert m["emotion"]["intensity"] > 0.4
```
Pass: all extracted memories carry appropriate emotion tags.

---

## Phase 3 — Salience Scoring

### Task 3.1 — Base salience formula

**Do:** Write `memory/scorer.py::compute_salience(base_weight, emotion, recall_count, turns_elapsed, decay_lambda=0.05) -> float`.

Formula: `salience = base_weight × emotion_multiplier × intensity_factor × recall_boost × e^(−λ × turns_elapsed)`

```python
EMOTION_MULTIPLIERS = {
    "neutral":  1.0,
    "joy":      1.4,
    "sadness":  1.6,
    "fear":     1.8,
    "anger":    1.5,
    "surprise": 1.3,
    "disgust":  1.4,
}

def compute_salience(base_weight, emotion, recall_count, turns_elapsed, decay_lambda=0.05):
    em = EMOTION_MULTIPLIERS.get(emotion["label"], 1.0)
    intensity_factor = 1 + emotion["intensity"]
    recall_boost = 1 + (0.3 * recall_count)
    decay = math.exp(-decay_lambda * turns_elapsed)
    return base_weight * em * intensity_factor * recall_boost * decay
```

**Expect:**
- Fear + intensity 0.95 + 0 turns → salience > 3.0
- Neutral + intensity 0.01 + 45 turns → salience < 0.25

**Verify:**
```python
# tests/test_scorer.py
def test_fear_high_intensity_no_decay():
    from memory.scorer import compute_salience
    score = compute_salience(1.0, {"label": "fear", "intensity": 0.95}, 0, 0)
    assert score > 3.0

def test_neutral_decays_below_threshold():
    from memory.scorer import compute_salience
    score = compute_salience(1.0, {"label": "neutral", "intensity": 0.01}, 0, 45)
    assert score < 0.25

def test_recall_boosts_salience():
    from memory.scorer import compute_salience
    low = compute_salience(1.0, {"label": "joy", "intensity": 0.5}, 0, 10)
    high = compute_salience(1.0, {"label": "joy", "intensity": 0.5}, 5, 10)
    assert high > low

def test_decay_is_exponential():
    from memory.scorer import compute_salience
    s10 = compute_salience(1.0, {"label": "neutral", "intensity": 0.5}, 0, 10)
    s20 = compute_salience(1.0, {"label": "neutral", "intensity": 0.5}, 0, 20)
    s30 = compute_salience(1.0, {"label": "neutral", "intensity": 0.5}, 0, 30)
    ratio_1 = s10 / s20
    ratio_2 = s20 / s30
    assert abs(ratio_1 - ratio_2) < 0.1
```
Pass: all four assertions hold.

---

### Task 3.2 — Forgetting cycle: compress and forget thresholds

**Do:** Write `memory/decay.py::apply_forgetting_cycle(memories, current_turn) -> tuple[list, list]`. Returns `(active_memories, forgotten_memories)`.

Rules:
- Salience < 0.25 → compress (merge into summary, reduce to single entry)
- Salience < 0.08 → forget (remove entirely)

**Expect:** After 45 turns, neutral-affect memories are forgotten. Fear-affect memories remain active.

**Verify:**
```python
# tests/test_decay.py
def test_neutral_memories_forgotten_after_45_turns():
    from memory.decay import apply_forgetting_cycle
    neutral_memories = [
        {"content": f"fact {i}", "base_weight": 1.0,
         "emotion": {"label": "neutral", "intensity": 0.01},
         "recall_count": 0, "turn_created": 0}
        for i in range(5)
    ]
    active, forgotten = apply_forgetting_cycle(neutral_memories, current_turn=45)
    assert len(forgotten) == 5
    assert len(active) == 0

def test_fear_memories_survive_45_turns():
    from memory.decay import apply_forgetting_cycle
    fear_memories = [
        {"content": "dog died", "base_weight": 1.0,
         "emotion": {"label": "fear", "intensity": 0.9},
         "recall_count": 0, "turn_created": 0}
    ]
    active, forgotten = apply_forgetting_cycle(fear_memories, current_turn=45)
    assert len(active) > 0
    assert len(forgotten) == 0

def test_compress_threshold_reduces_count():
    from memory.decay import apply_forgetting_cycle
    low_sal_memories = [
        {"content": f"minor thing {i}", "base_weight": 1.0,
         "emotion": {"label": "neutral", "intensity": 0.3},
         "recall_count": 0, "turn_created": 0}
        for i in range(5)
    ]
    active, forgotten = apply_forgetting_cycle(low_sal_memories, current_turn=20)
    assert len(active) == 1  # compressed into one summary
    assert len(forgotten) == 0
```
Pass: all three threshold behaviors correct.

---

## Phase 4 — Memory Store (ChromaDB + NetworkX)

### Task 4.1 — ChromaDB vector store setup

**Do:** Write `memory/store.py::MemoryStore` class with `add(memory, memory_id)` and `search(query, top_k) -> list[dict]`.

**Expect:** Add 10 memories, search for a related query, most semantically similar ranks first.

**Verify:**
```python
# tests/test_store.py
def test_semantic_search_ranking():
    from memory.store import MemoryStore
    store = MemoryStore("test_collection")
    store.add({"content": "user has a golden retriever named Max"}, "mem_0")
    store.add({"content": "user likes hiking on weekends"}, "mem_1")
    store.add({"content": "user's sister lives in Seattle"}, "mem_2")

    results = store.search("what kind of pet does the user have?", top_k=3)
    assert "Max" in results[0]["content"] or "retriever" in results[0]["content"].lower()

def test_add_and_retrieve_count():
    from memory.store import MemoryStore
    store = MemoryStore("test_count")
    for i in range(10):
        store.add({"content": f"fact number {i}"}, f"id_{i}")
    results = store.search("fact", top_k=5)
    assert len(results) == 5
```
Pass: semantic search returns most relevant memory first.

---

### Task 4.2 — NetworkX association graph

**Do:** Write `memory/store.py::AssociationGraph`. Auto-connects nodes with cosine similarity > 0.55. `get_neighbors(memory_id) -> list[str]`.

**Expect:** Dog memories auto-connect. Hiking memory does not connect to dog memories.

**Verify:**
```python
# tests/test_graph.py
def test_similar_memories_auto_connect():
    from memory.store import AssociationGraph
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    graph = AssociationGraph()

    graph.add_node("mem_dog1", embedder.encode("user has a golden retriever"))
    graph.add_node("mem_dog2", embedder.encode("user's dog is named Max"))
    graph.add_node("mem_hike", embedder.encode("user likes hiking in mountains"))

    assert "mem_dog2" in graph.get_neighbors("mem_dog1")
    assert "mem_hike" not in graph.get_neighbors("mem_dog1")

def test_graph_serialization():
    from memory.store import AssociationGraph
    import pickle
    graph = AssociationGraph()
    graph.add_node("a", np.ones(384))
    serialized = pickle.dumps(graph)
    restored = pickle.loads(serialized)
    assert "a" in restored.graph.nodes
```
Pass: dog memories link, hiking memory does not.

---

### Task 4.3 — Persistence: save and reload memory state

**Do:** Write `HippoMemory.save(path)` and `HippoMemory.load(path)`. Persist ChromaDB to disk and pickle the NetworkX graph.

**Expect:** Save, reload into fresh instance, search returns same results.

**Verify:**
```python
# tests/test_persistence.py
def test_save_and_reload(tmp_path):
    from memory.store import HippoMemory
    store = HippoMemory()
    store.add({"content": "user's dog is named Max",
               "emotion": {"label": "joy", "intensity": 0.7},
               "base_weight": 1.0, "recall_count": 0, "turn_created": 0}, "m1")
    store.save(str(tmp_path / "state"))

    restored = HippoMemory()
    restored.load(str(tmp_path / "state"))
    results = restored.search("dog name", top_k=1)
    assert "Max" in results[0]["content"]
```
Pass: search works identically after reload.

---

## Phase 5 — HippoRAG Retrieval

### Task 5.1 — Seed retrieval (vector search step)

**Do:** Write `memory/retriever.py::retrieve_seeds(query, store, top_k=3) -> list[str]`. Returns memory IDs.

**Verify:**
```python
def test_seed_retrieval_returns_relevant_ids(populated_store):
    from memory.retriever import retrieve_seeds
    seeds = retrieve_seeds("tell me about the user's pets", populated_store, top_k=3)
    seed_contents = [populated_store.get_by_id(s)["content"] for s in seeds]
    assert any("dog" in c.lower() or "max" in c.lower() for c in seed_contents)
```

---

### Task 5.2 — Graph walk (1-hop neighborhood expansion)

**Do:** Write `memory/retriever.py::expand_via_graph(seed_ids, graph, max_hops=1) -> list[str]`.

**Verify:**
```python
def test_graph_walk_expands_connected_nodes(connected_graph):
    from memory.retriever import expand_via_graph
    expanded = expand_via_graph(["mem_dog1"], connected_graph)
    assert "mem_dog1" in expanded
    assert "mem_dog2" in expanded
    assert "mem_hike" not in expanded

def test_graph_walk_no_neighbors_returns_seed_only(isolated_graph):
    from memory.retriever import expand_via_graph
    expanded = expand_via_graph(["isolated_mem"], isolated_graph)
    assert expanded == ["isolated_mem"]
```

---

### Task 5.3 — Full HippoRAG retrieve: seed + walk + rank by salience

**Do:** Write `memory/retriever.py::hippo_retrieve(query, store, graph, current_turn, top_k=5) -> list[dict]`. Seed → walk → rerank by salience → increment recall count.

**Verify:**
```python
def test_salience_reranking_over_similarity(populated_store, graph):
    from memory.retriever import hippo_retrieve
    populated_store.add({
        "content": "user mentioned pets briefly",
        "emotion": {"label": "neutral", "intensity": 0.01},
        "base_weight": 1.0, "recall_count": 0, "turn_created": 0
    }, "low_sal_pets")
    populated_store.add({
        "content": "user's dog Max died last month",
        "emotion": {"label": "sadness", "intensity": 0.95},
        "base_weight": 1.0, "recall_count": 0, "turn_created": 0
    }, "high_sal_dog")

    results = hippo_retrieve("pets", populated_store, graph, current_turn=5)
    assert results[0]["current_salience"] > 1.5

def test_retrieve_increments_recall_count(populated_store, graph):
    from memory.retriever import hippo_retrieve
    initial = populated_store.get_by_id("mem_dog1")["recall_count"]
    hippo_retrieve("dog", populated_store, graph, current_turn=5)
    after = populated_store.get_by_id("mem_dog1")["recall_count"]
    assert after > initial
```

---

### Task 5.4 — Retrieval latency benchmark

**Expect:** < 500ms with 500 memories in store.

**Verify:**
```python
def test_retrieval_latency_500_memories():
    from memory.store import HippoMemory
    from memory.retriever import hippo_retrieve
    import time
    store = HippoMemory()
    graph = AssociationGraph()
    for i in range(500):
        store.add({"content": f"memory {i} about various topics",
                   "emotion": {"label": "neutral", "intensity": 0.5},
                   "base_weight": 1.0, "recall_count": 0, "turn_created": i}, f"m{i}")
    start = time.perf_counter()
    hippo_retrieve("topic 250", store, graph, current_turn=600)
    assert time.perf_counter() - start < 0.5
```

---

## Phase 6 — LLM Generation

### Task 6.1 — LLM client wrapper

**Do:** Write `llm/client.py::LLMClient` wrapping Qwen3-8B. Exposes `generate(system, messages, max_tokens) -> str`.

**Verify:**
```python
def test_basic_generation(llm_client):
    response = llm_client.generate(
        system="You are helpful. Answer briefly.",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=50
    )
    assert "4" in response

def test_generation_respects_max_tokens(llm_client):
    response = llm_client.generate(
        system="You are helpful.",
        messages=[{"role": "user", "content": "Write a very long essay."}],
        max_tokens=20
    )
    token_count = len(llm_client.tokenizer.encode(response))
    assert token_count <= 25
```

---

### Task 6.2 — Context injection: retrieved memories → system prompt

**Do:** Write `llm/context.py::build_system_prompt(retrieved_memories, base_prompt) -> str`. Orders memories by salience descending. Bounded to 4096 tokens.

**Verify:**
```python
def test_memories_in_prompt():
    from llm.context import build_system_prompt
    memories = [
        {"content": "user's dog Max died", "current_salience": 2.5},
        {"content": "user likes hiking", "current_salience": 0.8},
    ]
    prompt = build_system_prompt(memories, "You are a compassionate companion.")
    assert "Max" in prompt
    assert "hiking" in prompt
    assert prompt.index("Max") < prompt.index("hiking")

def test_empty_memories_returns_base_prompt():
    from llm.context import build_system_prompt
    base = "You are a companion."
    assert build_system_prompt([], base) == base
```

---

## Phase 7 — TTS Module (Fish S2 Pro)

### Task 7.1 — Load Fish S2 Pro model

**Expect:** Model loads in < 60 seconds, no OOM (~9GB VRAM in fp16).

**Verify:**
```python
def test_fish_loads():
    from tts.model import load_fish_tts
    model = load_fish_tts()
    assert model is not None
    assert next(model.parameters()).is_cuda
```

---

### Task 7.2 — Synthesize speech from text

**Expect:** Given "Hello, how are you today?", produces a WAV > 0.5 seconds long.

**Verify:**
```python
def test_synthesize_produces_audio(fish_model, tmp_path):
    from tts.synthesize import synthesize
    import soundfile as sf
    out_path = str(tmp_path / "output.wav")
    synthesize(fish_model, "Hello, how are you today?", out_path)
    data, sr = sf.read(out_path)
    assert len(data) / sr > 0.5
    assert not np.all(data == 0)

def test_synthesize_latency(fish_model, tmp_path):
    import time
    from tts.synthesize import synthesize
    out_path = str(tmp_path / "lat_test.wav")
    start = time.perf_counter()
    synthesize(fish_model, "This is a latency test.", out_path)
    assert time.perf_counter() - start < 1.0
```

---

## Phase 8 — End-to-End Pipeline

### Task 8.1 — Pipeline orchestration class

**Do:** Write `pipeline.py::HippoVoicePipeline`. `process_turn(audio_path, output_path) -> str` runs the full loop: STT → extract → store → retrieve → generate → TTS → decay every 10 turns.

**Verify:**
```python
def test_single_turn_end_to_end(tmp_path):
    from pipeline import HippoVoicePipeline
    import soundfile as sf
    pipe = HippoVoicePipeline()
    out = pipe.process_turn("tests/fixtures/cat_sat.wav", str(tmp_path / "r.wav"))
    data, sr = sf.read(out)
    assert len(data) / sr > 0.3

def test_memory_persists_across_turns(tmp_path):
    from pipeline import HippoVoicePipeline
    pipe = HippoVoicePipeline()
    pipe.process_turn("tests/fixtures/my_dog_max.wav", str(tmp_path / "r1.wav"))
    pipe.process_turn("tests/fixtures/tell_me_about_pet.wav", str(tmp_path / "r2.wav"))
    results = pipe.memory.search("dog", top_k=1)
    assert len(results) > 0
```

---

### Task 8.2 — Multi-session persistence

**Verify:**
```python
def test_cross_session_memory(tmp_path):
    from pipeline import HippoVoicePipeline
    pipe1 = HippoVoicePipeline()
    pipe1.process_turn("tests/fixtures/my_dog_max.wav", str(tmp_path / "r1.wav"))
    pipe1.save(str(tmp_path / "state"))

    pipe2 = HippoVoicePipeline()
    pipe2.load(str(tmp_path / "state"))
    results = pipe2.memory.search("dog", top_k=1)
    assert len(results) > 0
    assert pipe2.current_turn == pipe1.current_turn
```

---

## Phase 9 — Benchmarks

### Task 9.1 — Signal/noise benchmark (core research claim)

**Do:** 45-turn synthetic conversation: odd turns = signal (fear/sadness, intensity 0.7–0.95), even turns = noise (neutral, intensity 0.01). After 45 turns, measure noise contamination in top-10 retrieved memories.

**Expect:** HippoVoice noise rate < 10%. Naive RAG noise rate > 40%.

**Verify:**
```python
def test_hippovoice_noise_rate():
    from pipeline import HippoVoicePipeline
    from benchmarks.signal_noise.run import run_signal_noise_benchmark
    pipe = HippoVoicePipeline(text_only=True)
    result = run_signal_noise_benchmark(pipe, "HippoVoice")
    assert result["noise_rate"] < 0.10

def test_naive_rag_baseline_is_noisy():
    from baselines.naive_rag import NaiveRAG
    from benchmarks.signal_noise.run import run_signal_noise_benchmark
    result = run_signal_noise_benchmark(NaiveRAG(), "NaiveRAG")
    assert result["noise_rate"] > 0.40  # validates benchmark is discriminating
```

---

### Task 9.2 — LoCoMo benchmark

**Do:** Run all 30 LoCoMo conversations through the pipeline (text mode), answer QA questions using retrieved context, compute accuracy.

**Expect:** > 65% accuracy (above Mem0 baseline).

**Verify:**
```python
def test_locomo_accuracy_above_baseline():
    from benchmarks.locomo.evaluate import run_locomo
    result = run_locomo(num_conversations=30)
    assert result["accuracy"] > 0.65
    print(f"LoCoMo accuracy: {result['accuracy']:.1%}")
```

---

### Task 9.3 — Token cost benchmark

**Expect:** Average tokens per query context ≤ 1000.

**Verify:**
```python
def test_token_cost_per_query(populated_pipeline):
    from benchmarks.token_cost.measure import measure_token_cost
    avg_tokens = measure_token_cost(populated_pipeline, num_queries=50)
    assert avg_tokens < 1000
    print(f"Avg tokens per query: {avg_tokens}")
```

---

## Test Coverage Summary

| Phase | Tasks | Tests | Key Assertion |
|-------|-------|-------|---------------|
| 0. Setup | 2 | 2 | GPU available, pytest runs |
| 1. STT | 5 | 7 | WER < 7%, embedding shape (1280,), latency < 2s |
| 2. Extraction | 4 | 6 | Facts extracted, emotions tagged, prosody boosts neutral |
| 3. Salience | 2 | 7 | Fear > 3.0, neutral < 0.25 after 45 turns, exponential decay |
| 4. Store | 3 | 5 | Semantic search ranks correctly, graph auto-connects, persistence works |
| 5. Retrieval | 4 | 7 | Graph walk expands correctly, salience overrides similarity, < 500ms |
| 6. LLM | 2 | 4 | Generates correct answers, memories injected in salience order |
| 7. TTS | 2 | 3 | Audio produced, > 0.5s duration, < 1s latency |
| 8. Pipeline | 2 | 4 | End-to-end runs, memory persists across sessions |
| 9. Benchmarks | 3 | 5 | Noise < 10%, LoCoMo > 65%, tokens < 1000 |

**Total: 29 tasks, 50 tests.**

Every paper claim has a corresponding test. Run `pytest -v` at any point to see current status.

---

## Compute Requirements

- **Track 1:** Colab A100 (40GB VRAM)
  - Canary-Qwen 2.5B: ~5GB
  - Qwen3-8B LLM: ~16GB
  - Fish S2 Pro (4B + 400M): ~9GB
  - Total: ~30GB — fits within A100 with careful loading
- **Estimated cost:** $50–150 total on Colab pay-per-use A100

## Build Order

Start → Phase 0 → Phase 2 (scorer first, no GPU needed) → Phase 3 → Phase 4 → Phase 5 → Phase 1 (GPU) → Phase 6 (GPU) → Phase 7 (GPU) → Phase 8 → Phase 9

Build and test each phase independently before moving to the next.
