"""
Transcription and acoustic embedding extraction via Canary-Qwen 2.5B.
"""

import numpy as np


def transcribe(model, audio_path: str) -> str:
    """Return transcript string for a single audio file."""
    output = model.transcribe([audio_path], batch_size=1)
    if not output:
        return ""
    # NeMo returns Hypothesis objects or plain strings depending on version
    result = output[0]
    return result.text if hasattr(result, "text") else str(result)


def extract_encoder_embedding(model, audio_path: str) -> np.ndarray:
    """
    Run the FastConformer encoder on audio and return a mean-pooled 1280-dim vector.

    We hook into the encoder's output before the decoder to get acoustic
    representations that carry prosodic information text transcription discards.
    """
    import torch

    # NeMo's preprocessor + encoder pipeline
    device = next(model.parameters()).device

    # Preprocess audio to mel features
    processed, lengths = model.preprocessor(
        input_signal=_load_audio_tensor(audio_path, device),
        length=_get_audio_length(audio_path, device),
    )

    with torch.no_grad():
        encoded, encoded_len = model.encoder(audio_signal=processed, length=lengths)

    # encoded shape: (batch=1, time, dim=1280) — mean pool over time
    embedding = encoded[0, :encoded_len[0], :].mean(dim=0).cpu().numpy()
    return embedding.astype(np.float32)


def transcribe_with_embedding(model, audio_path: str) -> tuple[str, np.ndarray]:
    """Convenience wrapper: returns (transcript, 1280-dim acoustic embedding)."""
    text = transcribe(model, audio_path)
    emb = extract_encoder_embedding(model, audio_path)
    return text, emb


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_audio_tensor(audio_path: str, device):
    import torch
    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono
    # NeMo expects (batch, samples)
    return torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)


def _get_audio_length(audio_path: str, device):
    import torch
    import soundfile as sf

    info = sf.info(audio_path)
    length = torch.tensor([info.frames], dtype=torch.long, device=device)
    return length
