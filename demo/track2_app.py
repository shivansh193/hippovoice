"""
Interactive Track 2 (audio-to-audio memory) demo -- a Gradio app wrapping
HippoAudioPipeline (pipeline_audio2audio.py) so the memory capture +
conditioning loop can actually be driven live instead of only exercised
by benchmarks/locomo/evaluate_audio.py and the unit tests.

Two backends, selected via BACKEND:
  - "mock" (default): MockAudioToAudioModel, zero GPU, zero model download.
    Validates the whole UI/plumbing loop (STT -> retrieve -> context-audio
    build -> respond -> extract -> store) for free before spending any GPU
    time on it -- same "validate cheap before expensive" discipline as
    qwen_omni_audio_model.py's own throwaway-AWS-instance validation.
    The mock's own response text is an obviously-synthetic echo, not a
    real answer, so this mode is for checking the pipeline wiring and the
    UI, not for judging real audio-to-audio quality.
  - "qwen": the real Qwen25OmniAudioModel (Qwen2.5-Omni-3B), the backend
    already confirmed live on a throwaway AWS GPU instance (see that
    module's docstring). Needs a GPU with the extra install listed there
    (the preview transformers branch + qwen-omni-utils) -- this app does
    not attempt to install those itself, since doing so silently would
    conflict with this project's normal transformers>=4.40.0 pin for
    everyone NOT running this specific demo.

STT is whisper-tiny (stt/model.py's load_whisper, aliased load_canary) for
BOTH backends -- HippoAudioPipeline.process_turn takes user_text as an
explicit, separate parameter rather than deriving it from user_audio_path
internally (see that class's own docstring for why), and Qwen's own
transcript output is of the ASSISTANT's reply, not the user's utterance,
so it can't stand in for this. whisper-tiny runs fine on CPU, which keeps
the "mock" backend usable with zero GPU at all.

Core turn-handling logic (build_pipeline, seed_memory, handle_turn) is
kept independent of any Gradio import so it stays unit-testable without
a Gradio runtime -- see tests/test_track2_app.py.
"""

import os

BACKEND = os.environ.get("HIPPOVOICE_DEMO_BACKEND", "mock")


def build_pipeline(backend: str = BACKEND, llm_client=None, memory_path: str | None = None):
    """Construct a loaded HippoAudioPipeline for the given backend name."""
    from pipeline_audio2audio import HippoAudioPipeline, MockAudioToAudioModel

    if backend == "mock":
        audio_model = MockAudioToAudioModel()
    elif backend == "qwen":
        from qwen_omni_audio_model import Qwen25OmniAudioModel
        audio_model = Qwen25OmniAudioModel()
    else:
        raise ValueError(f"Unknown backend {backend!r} -- expected 'mock' or 'qwen'")

    audio_model.load()
    return HippoAudioPipeline(audio_model=audio_model, llm_client=llm_client, memory_path=memory_path)


def get_stt_model():
    """
    Lazy singleton so the app only pays whisper-tiny's load cost once per
    process, not once per turn -- mirrors how HippoAudioPipeline itself
    lazy-loads its own LLM via the `llm` property.
    """
    from stt.model import load_whisper
    return load_whisper("tiny")


def seed_memory(pipeline, text: str) -> str:
    """
    Ingest a line of background context without forcing a spoken turn --
    lets a demo start with "the user's dog is named Max" already in
    memory instead of needing several real conversational turns first to
    build up anything worth retrieving. Returns a short status string for
    the UI rather than None, since a bare ingest gives no visible
    confirmation otherwise.
    """
    if not text or not text.strip():
        return "(nothing to seed)"
    pipeline.ingest_text_turn(text.strip())
    return f"Seeded: {text.strip()!r} (turn {pipeline.current_turn})"


def handle_turn(pipeline, stt_model, user_audio_path: str) -> dict:
    """
    One full demo turn: transcribe what the user said, show what memory
    that retrieves (before generation, so the UI can display exactly what
    context is about to condition the response -- not a separate,
    possibly-different retrieval from the one process_turn does
    internally, since retrieve() is a pure read with no side effects and
    calling it twice with the same query returns the same result), then
    run the real audio-to-audio turn and let it store its own memory.

    Returns a dict rather than a tuple so the Gradio callback can map keys
    onto specific UI components by name instead of positional order.
    """
    from stt.transcribe import transcribe

    user_text = transcribe(stt_model, user_audio_path)
    retrieved = pipeline.retrieve(user_text, top_k=5)
    response_audio_path, transcript = pipeline.process_turn(user_audio_path, user_text)

    return {
        "user_text": user_text,
        "retrieved": retrieved,
        "response_audio_path": response_audio_path,
        "response_transcript": transcript,
    }


def format_retrieved(retrieved: list[dict]) -> str:
    """Human-readable retrieved-memory panel content -- content + score,
    newest-scored-first (retrieve() already returns them in that order)."""
    if not retrieved:
        return "(nothing retrieved yet -- memory is empty, or nothing matched)"
    lines = []
    for r in retrieved:
        score = r.get("_score", 0.0)
        content = r.get("content", "")
        lines.append(f"[{score:.3f}] {content}")
    return "\n".join(lines)


def _launch():
    """Gradio UI wiring -- kept out of module import time so importing this
    file for its testable functions (see tests/test_track2_app.py) doesn't
    require gradio to be installed at all."""
    import gradio as gr

    pipeline = build_pipeline()
    stt_model = get_stt_model()
    history: list[str] = []

    def on_seed(text):
        status = seed_memory(pipeline, text)
        return status, ""

    def on_turn(audio_path):
        if not audio_path:
            return "(record something first)", "", None, "", "\n".join(history)
        result = handle_turn(pipeline, stt_model, audio_path)
        history.append(f"You: {result['user_text']}")
        history.append(f"Companion: {result['response_transcript']}")
        return (
            result["user_text"],
            format_retrieved(result["retrieved"]),
            result["response_audio_path"],
            result["response_transcript"],
            "\n".join(history),
        )

    def on_reset():
        history.clear()
        return "(memory + history cleared)", ""

    with gr.Blocks(title="HippoVoice Track 2 Demo") as demo:
        gr.Markdown(
            f"# HippoVoice -- Track 2 (audio-to-audio memory) demo\n"
            f"Backend: **{BACKEND}** (set `HIPPOVOICE_DEMO_BACKEND=qwen` "
            f"before launch for the real Qwen2.5-Omni backend -- needs a "
            f"GPU + extra install, see `qwen_omni_audio_model.py`)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Seed background memory (optional)")
                seed_box = gr.Textbox(
                    label="Background fact to remember",
                    placeholder="e.g. My dog's name is Max and he's a golden retriever.",
                )
                seed_btn = gr.Button("Seed memory")
                seed_status = gr.Textbox(label="Status", interactive=False)
                reset_btn = gr.Button("Reset memory + conversation", variant="stop")

            with gr.Column(scale=2):
                gr.Markdown("### Speak a turn")
                mic = gr.Audio(sources=["microphone"], type="filepath", label="Your turn")
                turn_btn = gr.Button("Send", variant="primary")
                user_text_box = gr.Textbox(label="What you said (transcribed)", interactive=False)
                retrieved_box = gr.Textbox(
                    label="Memory retrieved for this turn (score, content)",
                    interactive=False, lines=6,
                )
                response_audio = gr.Audio(label="Companion's spoken reply", interactive=False)
                response_text_box = gr.Textbox(label="Companion's reply (transcript)", interactive=False)

        gr.Markdown("### Conversation log")
        log_box = gr.Textbox(label="", interactive=False, lines=10)

        seed_btn.click(on_seed, inputs=[seed_box], outputs=[seed_status, seed_box])
        turn_btn.click(
            on_turn,
            inputs=[mic],
            outputs=[user_text_box, retrieved_box, response_audio, response_text_box, log_box],
        )
        reset_btn.click(on_reset, outputs=[seed_status, log_box])

    # share=True gets a public *.gradio.live tunnel URL -- the practical
    # way to hand a live link to someone when this app is running on a
    # remote GPU instance (there's no other browser-reachable path to a
    # box that isn't already exposing its own port). Off by default so a
    # purely local run (the "mock" backend's normal use case) doesn't
    # silently open a public tunnel nobody asked for.
    share = os.environ.get("HIPPOVOICE_DEMO_SHARE", "0") == "1"
    demo.launch(share=share, server_name="0.0.0.0" if share else "127.0.0.1")


if __name__ == "__main__":
    _launch()
