"""
STT model loader — Whisper (lightweight, runs on CPU/MPS/CUDA).

Default: whisper-tiny (~150MB, fast on CPU)
Upgrade path: whisper-base, whisper-small, whisper-medium, whisper-large-v3
"""


def load_whisper(model_size: str = "tiny"):
    """
    Load an OpenAI Whisper model.

    pip install openai-whisper
    """
    import whisper
    model = whisper.load_model(model_size)
    model.eval()
    return model


# Alias used by voice pipeline cells
load_canary = load_whisper
