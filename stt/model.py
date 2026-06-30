"""
STT model loader for Canary-Qwen 2.5B.

Requires NeMo: pip install nemo_toolkit[asr]
Run on Colab A100 — model needs ~5GB VRAM in fp16.
"""


def load_canary():
    """Load nvidia/canary-qwen-2.5b from HuggingFace via NeMo."""
    from nemo.collections.asr.models import EncDecMultiTaskModel

    model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-qwen-2.5b")
    model.eval()
    return model
