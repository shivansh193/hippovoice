"""
TTS model loader — pyttsx3 (offline, zero download, works on Mac/Linux/Windows).

pip install pyttsx3

Upgrade path: swap load_tts() for kokoro-onnx, Coqui TTS, or Fish Speech
when you need higher quality audio.
"""


def load_tts(rate: int = 175, volume: float = 1.0):
    """
    Load pyttsx3 engine. Returns a configured engine instance.

    rate:   words per minute (default 175)
    volume: 0.0–1.0
    """
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    return engine


# Alias used by voice pipeline cells
load_fish_tts = load_tts
