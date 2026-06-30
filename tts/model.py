"""
TTS model loader for Fish Audio S2 Pro.

Installation on Colab:
    git clone https://github.com/fishaudio/fish-speech
    cd fish-speech && pip install -e .

Or via HuggingFace:
    pip install fish-speech  (unofficial package — verify before use)

Model needs ~9GB VRAM (4B slow AR + 400M fast AR) in fp16 on A100.
"""


def load_fish_tts(model_path: str | None = None):
    """
    Load Fish S2 Pro. Tries HuggingFace hub if no local path given.
    Returns a model object with a .generate() method.
    """
    try:
        # Attempt fish-speech package import (installed from their repo)
        from fish_speech.models.text2semantic.llama import DualARTransformer
        from fish_speech.inference import TTSInferenceEngine

        engine = TTSInferenceEngine.from_pretrained(
            model_path or "fishaudio/fish-speech-1.5"
        )
        return engine
    except ImportError:
        raise RuntimeError(
            "Fish Speech not installed. On Colab:\n"
            "  git clone https://github.com/fishaudio/fish-speech\n"
            "  cd fish-speech && pip install -e .\n"
            "Then restart the runtime."
        )
