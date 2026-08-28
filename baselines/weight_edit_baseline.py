"""
WeightEditBaseline -- targeted weight editing (ROME/MEMIT-style) as an
alternative to RAG-style memory, benchmarked through the same run_locomo
harness as the other baselines (Mem0-style, A-MEM-style, NaiveRAG), for a
genuine comparison rather than an isolated demo.

Motivation (from a real conversation about this project): if a model's
weights can be directly edited to encode a fact, does that work better or
worse than storing the fact externally and retrieving it? Nobody does
weight editing for personalized conversational memory in production, for
good reason -- see below -- but "nobody does it" isn't the same as "we
checked and it doesn't work here," so this benchmark actually checks.

Structurally different from every other baseline in this project. Mem0-
style/A-MEM-style/NaiveRAG all share one fixed LLM and only swap the
external memory store between conversations. This one mutates the model's
own weights as ingestion proceeds -- the edited model *is* the memory, so
there's no separate retrieval step feeding context into an unedited
model's prompt at generation time. That means each conversation needs its
own independently-editable model state, reset back to base weights before
the next conversation starts, for the same reason the RAG baselines start
each conversation with an empty store: otherwise conversation 2 would
inherit conversation 1's edits, which isn't a fair per-conversation
comparison.

Known limitation going in, not discovered after the fact: ROME/MEMIT were
designed and validated for single, discrete factual edits, not for
accumulating hundreds of loosely-structured personal facts over an ongoing
relationship. Published work on sequential editing shows edits
interfering with each other and degrading unrelated knowledge well before
edit counts reach what a single real conversation needs (LoCoMo
conversations run 369-689 turns). Expected outcome is that this baseline
shows real degradation at scale where the RAG-style systems don't --
that's a legitimate result to demonstrate, not a failure of the benchmark.

Two genuinely separate concerns, deliberately split into two classes so
one of them stays testable without a GPU:
  1. Converting a free-text extracted memory into a ROME/MEMIT-compatible
     edit request (a cloze-style prompt + subject + target-new) -- this is
     just another prompted LLM call, structurally identical to
     memory/extractor.py's own extraction step, and is fully testable with
     a mock LLM, no real model needed.
  2. Actually applying that edit to a real model's weights and generating
     from the edited model -- this needs a real model loaded and the
     EasyEdit/ROME-MEMIT machinery, and genuinely can't be meaningfully
     mocked: a fake edit function validates that Python calls happen in
     the right order, not whether editing actually works. WeightEditor's
     real implementation (GPT-2 XL -- one of the two models the original
     ROME paper validated the technique on, chosen specifically because it
     fits a free Kaggle T4 unlike GPT-J-6B, the other) is built and
     validated on Kaggle, not here. MockWeightEditor stands in for it so
     everything else -- extraction reuse, edit-request conversion,
     per-conversation reset, harness wiring -- can be validated for free
     first, same discipline as pipeline_audio2audio.py's MockAudioToAudioModel.
"""

import json

from memory.extractor import extract_memories

EDIT_EXTRACTION_PROMPT = """\
Convert this piece of remembered information into a fill-in-the-blank fact
suitable for a knowledge-editing system. Respond with JSON:
{{"prompt": "<cloze-style sentence ending right before the answer>",
  "subject": "<the specific entity the fact is about>",
  "target_new": "<the short answer that completes the prompt>"}}

Example: content "Caroline's dog is named Max" ->
{{"prompt": "Caroline's dog is named", "subject": "Caroline's dog", "target_new": "Max"}}

If this isn't a simple, single-answer factual statement (an event, a
feeling, something with no one-word/short-phrase answer), respond with
{{"skip": true}} instead -- not every memory is editable this way, and
forcing one into this shape is worse than skipping it.

Memory: {memory}
"""


class WeightEditor:
    """
    Interface real weight-editing backends implement. Kept separate from
    WeightEditBaseline specifically so "does editing work" (needs a real
    model, can't be mocked) is isolated from "convert memories into edit
    requests, reset between conversations" (pure logic, mockable) -- same
    split pattern as AudioToAudioModel in pipeline_audio2audio.py.
    """

    def load(self):
        raise NotImplementedError

    def edit(self, prompt: str, subject: str, target_new: str) -> None:
        """Apply one ROME/MEMIT-style edit to the model's weights."""
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate from the *current* (possibly already-edited) weights."""
        raise NotImplementedError

    def reset(self) -> None:
        """
        Restore base (unedited) weights. Called at the start of every new
        WeightEditBaseline instance (i.e. every conversation, matching
        pipeline_factory's per-conversation construction in run_locomo) --
        the real implementation needs to actually reload/restore original
        weights here, not just clear a Python-side dict.
        """
        raise NotImplementedError


class MockWeightEditor(WeightEditor):
    """
    Dry-run stand-in -- no model weights, no GPU. Tracks applied edits in a
    plain dict and answers generate() by literal subject-string lookup, so
    tests can assert on exactly what got edited and confirm reset()
    actually clears state. Deliberately validates nothing about whether
    real weight editing works or degrades under many edits -- see module
    docstring for why that's out of scope for a mock.
    """

    def __init__(self):
        self.loaded = False
        self.edits: dict[str, str] = {}  # subject -> target_new, last edit wins
        self.edit_log: list[tuple[str, str, str]] = []

    def load(self):
        self.loaded = True

    def edit(self, prompt: str, subject: str, target_new: str) -> None:
        self.edits[subject] = target_new
        self.edit_log.append((prompt, subject, target_new))

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        for subject, target in self.edits.items():
            if subject.lower() in prompt.lower():
                return target
        return "[no edited fact matches this prompt]"

    def reset(self) -> None:
        self.edits.clear()
        self.edit_log.clear()


class _EditedModelLLM:
    """
    Thin adapter so run_locomo's QA step (conv_pipeline.llm.generate(
    system=, messages=, max_tokens=)) can call a WeightEditor with the same
    call shape it calls a real LLMClient with. system/messages collapse
    into one generate() call on the editor -- the editor has no separate
    system-prompt concept, its "context" is whatever's been edited into its
    weights, not text handed to it per-call, which is the entire point
    being tested.
    """

    def __init__(self, editor: WeightEditor):
        self.editor = editor
        self.model_name = "weight-edited-model"
        self._backend = "weight-edit"

    def generate(self, system: str, messages: list[dict], max_tokens: int = 256) -> str:
        prompt = messages[-1]["content"] if messages else ""
        return self.editor.generate(prompt, max_tokens=max_tokens)


class WeightEditBaseline:
    """
    Wire-compatible with the other baselines (ingest_text_turn, .llm with a
    .generate() method) so it drops into run_locomo via the same
    pipeline_factory mechanism as Mem0-style/A-MEM-style/NaiveRAG.

    llm_client here is used only for extraction (memory/extractor.py's
    extract_memories -- the same function every other system in this
    project uses, so extraction quality/rate is identical across systems
    and isn't a confound) and for converting an extracted memory's content
    into an edit request. It is deliberately NOT what QA answers come
    from. QA answers come from self.llm (_EditedModelLLM wrapping the
    actual edited model), because the whole point of this baseline is
    testing whether the model itself, via its own edited weights, can
    answer correctly with zero retrieved context at generation time.
    """

    def __init__(self, llm_client, editor: WeightEditor):
        self._extraction_llm = llm_client
        self.editor = editor
        self.editor.reset()  # every new conversation starts from base weights
        self.llm = _EditedModelLLM(editor)

    def ingest_text_turn(self, text: str):
        for memory in extract_memories(text, self._extraction_llm):
            edit_request = self._to_edit_request(memory.get("content", ""))
            if edit_request is not None:
                self.editor.edit(**edit_request)

    def _to_edit_request(self, content: str) -> dict | None:
        if not content:
            return None
        raw = self._extraction_llm.generate(
            system="You convert conversational memories into knowledge-edit requests.",
            messages=[{"role": "user", "content": EDIT_EXTRACTION_PROMPT.format(memory=content)}],
            max_tokens=150,
        )
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if parsed.get("skip"):
            return None
        if not all(k in parsed for k in ("prompt", "subject", "target_new")):
            return None
        return {"prompt": parsed["prompt"], "subject": parsed["subject"], "target_new": parsed["target_new"]}
