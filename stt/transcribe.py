"""
Transcription and acoustic embedding extraction via Whisper.

Whisper's encoder output (1500-frame, 384/512/768-dim depending on model size)
is mean-pooled to produce a fixed-size acoustic embedding for prosody fusion.
"""

import numpy as np


def transcribe(model, audio_path: str) -> str:
    """Return transcript string for a single audio file."""
    result = model.transcribe(audio_path, fp16=False)
    return result.get("text", "").strip()


def extract_encoder_embedding(model, audio_path: str) -> np.ndarray:
    """
    Run Whisper's encoder on audio and return a mean-pooled embedding.

    Shape: (n_audio_ctx, n_mels) → mean over time → (n_mels,)
    Used for prosody-aware emotion fusion — variance of this vector correlates
    with speech energy/pitch variation.
    """
    import torch
    import whisper

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    with torch.no_grad():
        encoded = model.encoder(mel.unsqueeze(0))  # (1, n_ctx, n_state)

    embedding = encoded[0].mean(dim=0).cpu().numpy().astype(np.float32)
    return embedding


def transcribe_with_embedding(model, audio_path: str) -> tuple[str, np.ndarray]:
    """Convenience wrapper: returns (transcript, acoustic_embedding)."""
    text = transcribe(model, audio_path)
    emb = extract_encoder_embedding(model, audio_path)
    return text, emb
