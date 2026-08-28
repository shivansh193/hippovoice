"""
GeminiLiveAudioModel -- real AudioToAudioModel implementation wrapping
Google's Gemini Live API (bidiGenerateContent), for pipeline_audio2audio.py.

Chosen over self-hosting an open-weight audio-to-audio model (Moshi,
Qwen2.5-Omni, etc.) specifically because HippoAudioPipeline doesn't edit
this model's weights anywhere -- it only calls respond() -- so there's
nothing an open-weight model buys here that a proprietary API doesn't,
and the API needs no GPU, no self-hosting, and no AWS instance at all.

Confirmed working end-to-end via a real, live call before this file was
written (not reconstructed from docs alone, unlike the Mini-Omni notebook,
which had no GPU available to test against): input transcript correctly
captured, a genuine coherent spoken response came back, response audio
saved and playable. The exact call shape below (client.aio.live.connect,
send_realtime_input with a Blob, iterating session.receive() for
server_content.model_turn.parts[].inline_data.data and
input_transcription/output_transcription) is what that real test used,
not a best-effort guess.

Audio format is dictated by Google's own spec, not a guess: 16-bit PCM,
16000Hz, little-endian in; 16-bit PCM, 24000Hz, little-endian out. Input
gets resampled to 16kHz mono here regardless of what HippoAudioPipeline
hands it (which itself is already resampled to whatever the "real user
audio" rate is by _concatenate_audio -- this adapter doesn't assume that
happens to already be 16kHz).
"""

import asyncio

from pipeline_audio2audio import AudioToAudioModel

GEMINI_INPUT_SAMPLE_RATE = 16000
GEMINI_OUTPUT_SAMPLE_RATE = 24000


class GeminiLiveAudioModel(AudioToAudioModel):
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash-native-audio-latest"):
        # api_key=None lets the underlying SDK fall back to the
        # GEMINI_API_KEY / GOOGLE_API_KEY env var -- avoids ever needing the
        # key to exist as a literal string in code that might get committed.
        self._api_key = api_key
        self.model = model
        self._client = None

    def load(self):
        from google import genai
        self._client = genai.Client(api_key=self._api_key) if self._api_key else genai.Client()

    def respond(self, audio_path: str) -> tuple[str, str]:
        """
        Returns (response_audio_path, transcript) where transcript is the
        model's own spoken reply in text form (output_transcription) --
        what it said, not what it heard. Input transcript is still
        captured internally (self.last_input_transcript) for debugging,
        since HippoAudioPipeline already gets user_text passed explicitly
        and doesn't rely on this adapter for that.
        """
        if self._client is None:
            self.load()
        return asyncio.run(self._respond_async(audio_path))

    async def _respond_async(self, audio_path: str) -> tuple[str, str]:
        from google.genai import types

        pcm_bytes = _read_as_pcm16(audio_path, GEMINI_INPUT_SAMPLE_RATE)

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        audio_chunks = []
        input_transcript = ""
        output_transcript = ""

        async with self._client.aio.live.connect(model=self.model, config=config) as session:
            await session.send_realtime_input(
                audio=types.Blob(data=pcm_bytes, mime_type=f"audio/pcm;rate={GEMINI_INPUT_SAMPLE_RATE}")
            )
            await session.send_realtime_input(audio_stream_end=True)

            async for message in session.receive():
                sc = message.server_content
                if sc is None:
                    continue
                if sc.input_transcription and sc.input_transcription.text:
                    input_transcript += sc.input_transcription.text
                if sc.output_transcription and sc.output_transcription.text:
                    output_transcript += sc.output_transcription.text
                if sc.model_turn and sc.model_turn.parts:
                    for part in sc.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            audio_chunks.append(part.inline_data.data)
                if sc.turn_complete:
                    break

        self.last_input_transcript = input_transcript

        import tempfile
        response_path = tempfile.mktemp(suffix=".wav")
        _write_pcm16_wav(response_path, b"".join(audio_chunks), GEMINI_OUTPUT_SAMPLE_RATE)
        return response_path, output_transcript


def _read_as_pcm16(audio_path: str, target_sr: int) -> bytes:
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample

    data, sr = sf.read(audio_path, dtype="int16")
    if data.ndim > 1:  # stereo -> mono, Gemini Live expects mono input
        data = data.mean(axis=1).astype(np.int16)
    if sr != target_sr:
        data = resample(data.astype(np.float32), int(len(data) * target_sr / sr)).astype(np.int16)
    return data.tobytes()


def _write_pcm16_wav(path: str, pcm_bytes: bytes, sample_rate: int) -> None:
    import soundfile as sf
    import numpy as np

    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    sf.write(path, arr, sample_rate)
