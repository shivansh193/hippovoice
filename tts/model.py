"""
TTS model loader — pyttsx3 (offline, zero download, works on Mac/Linux/Windows).

pip install pyttsx3

Upgrade path: swap load_tts() for kokoro-onnx, Coqui TTS, or Fish Speech
when you need higher quality audio.
"""


class TTSHandle:
    """
    Lightweight stand-in for a pyttsx3 engine -- holds configured
    rate/volume but never calls pyttsx3.init() itself.

    Confirmed as a real, reproducible deadlock, not just the previously
    documented "reusing one engine instance across calls" case: an
    isolated test showed even a SECOND freshly-constructed pyttsx3.init()
    call in the same process hangs forever on Windows (SAPI5's COM
    apartment state doesn't clean up between engine lifecycles). Since
    load_tts() is the thing every call site was calling repeatedly
    (once per turn, once per QA question, ...) expecting a fresh,
    independent engine each time, load_tts() itself was the actual
    trigger -- not just tts.synthesize.synthesize(). The real
    pyttsx3.init() now happens exactly once per OS process, inside the
    short-lived subprocess tts.synthesize.synthesize()/speak() spawns for
    each call -- this handle just carries the rate/volume settings there.

    Exposes getProperty()/setProperty() matching pyttsx3's own engine
    interface so existing callers that configure rate/volume before
    passing the handle to synthesize() don't need to change.
    """

    def __init__(self, rate: float = 175, volume: float = 1.0):
        self._properties = {"rate": rate, "volume": volume}

    def getProperty(self, name: str):
        return self._properties[name]

    def setProperty(self, name: str, value) -> None:
        self._properties[name] = value


def load_tts(rate: int = 175, volume: float = 1.0) -> TTSHandle:
    """
    Returns a lightweight handle carrying the requested rate/volume --
    NOT a live pyttsx3 engine (see TTSHandle's docstring for why: a real
    pyttsx3.init() here would be the second one in-process on every call
    after the first, which reliably deadlocks on Windows). The real
    engine gets constructed fresh inside a subprocess by
    tts.synthesize.synthesize()/speak(), which read rate/volume off
    whatever handle they're given.

    rate:   words per minute (default 175)
    volume: 0.0–1.0
    """
    return TTSHandle(rate=rate, volume=volume)


# Alias used by voice pipeline cells
load_fish_tts = load_tts
