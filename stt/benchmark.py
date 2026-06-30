"""WER computation for STT benchmarking."""

from jiwer import wer as jiwer_wer


def compute_wer(hypotheses: list[str], references: list[str]) -> float:
    """Return Word Error Rate in [0, 1] range."""
    return float(jiwer_wer(references, hypotheses))
