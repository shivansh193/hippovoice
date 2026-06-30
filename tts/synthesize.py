"""Speech synthesis via Fish S2 Pro."""

import numpy as np
import soundfile as sf


def synthesize(model, text: str, output_path: str, sample_rate: int = 44100) -> str:
    """
    Synthesise text to speech and write a WAV file.

    Returns the output_path on success.
    """
    audio_array = model.generate(text)

    # Normalise if needed
    if isinstance(audio_array, np.ndarray):
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        max_val = np.abs(audio_array).max()
        if max_val > 1.0:
            audio_array /= max_val
    else:
        # Some Fish Speech versions return a torch tensor
        import torch
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.cpu().numpy().astype(np.float32)

    sf.write(output_path, audio_array, sample_rate)
    return output_path
