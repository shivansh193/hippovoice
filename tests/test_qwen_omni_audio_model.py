"""
Tests for qwen_omni_audio_model.py. The real model needs a GPU and the
Qwen2.5-Omni preview transformers branch, neither available in this local
test environment -- validated for real instead on a throwaway AWS GPU
instance (see BUGS.md and the module's own docstring for that run's
numbers). These tests cover the adapter's own logic (system prompt
selection, token-slicing to isolate just the generated response, audio
save path/sample rate) against a mocked model/processor, the same
approach test_gemini_live_model.py uses for GeminiLiveAudioModel.
"""
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf
import torch
import pytest

from qwen_omni_audio_model import Qwen25OmniAudioModel, QWEN_OUTPUT_SAMPLE_RATE


@pytest.fixture(autouse=True)
def fake_qwen_omni_utils(monkeypatch):
    """qwen_omni_utils isn't installed locally (needs the Qwen2.5-Omni
    preview transformers branch) -- respond() imports it lazily, so a
    fake module in sys.modules is enough to exercise that import path
    without the real dependency."""
    fake_module = types.ModuleType("qwen_omni_utils")
    fake_module.process_mm_info = MagicMock(return_value=(["fake_audio"], None, None))
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", fake_module)
    return fake_module


def _make_mock_model_and_processor(prompt_len=10, response_tokens=(1, 2, 3), response_text="Paris"):
    """Builds a mock model + processor pair matching the real shapes
    respond() reads from: inputs.input_ids.shape[1] for the prompt
    length, model.generate() returning (text_ids, audio)."""
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "rendered chat template"

    mock_inputs = MagicMock()
    mock_inputs.input_ids.shape = (1, prompt_len)
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs

    # Full text_ids includes the whole rendered prompt (prompt_len tokens)
    # followed by the actual generated response tokens -- respond() must
    # slice off exactly the first `prompt_len` before decoding. Real
    # torch tensors, not plain numpy, matching what model.generate()
    # actually returns (respond() calls .detach().cpu().numpy() on the
    # audio, which numpy arrays don't support).
    full_ids = torch.tensor([[0] * prompt_len + list(response_tokens)])
    mock_audio = torch.zeros(2400)  # 0.1s @ 24kHz

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.dtype = "float32"
    mock_model.generate.return_value = (full_ids, mock_audio)

    def fake_batch_decode(ids, skip_special_tokens=True):
        # Confirms respond() actually passed the SLICED ids, not the full
        # ones -- a real regression this test would catch: if respond()
        # forgot to slice, ids.shape[1] would be prompt_len + len(response_tokens),
        # not just len(response_tokens).
        assert ids.shape[1] == len(response_tokens), (
            f"expected sliced response-only ids (len {len(response_tokens)}), "
            f"got {ids.shape[1]} -- prompt wasn't stripped before decoding"
        )
        return [response_text]

    mock_processor.batch_decode.side_effect = fake_batch_decode
    return mock_model, mock_processor


def test_respond_slices_prompt_tokens_before_decoding(tmp_path):
    """Direct regression test for the real parsing bug this adapter
    avoids: a live test showed model.generate()'s text_ids includes the
    ENTIRE rendered chat template (system/user/assistant), not just the
    new response -- naively decoding the full thing would return
    "system\\n...\\nuser\\n\\nassistant\\nParis" instead of just "Paris"."""
    model = Qwen25OmniAudioModel()
    model._model, model._processor = _make_mock_model_and_processor(
        prompt_len=10, response_tokens=(1, 2, 3), response_text="Paris"
    )

    audio_path = str(tmp_path / "question.wav")
    sf.write(audio_path, np.zeros(16000, dtype=np.int16), 16000)

    response_path, transcript = model.respond(audio_path)

    assert transcript == "Paris"


def test_respond_saves_audio_at_correct_sample_rate(tmp_path):
    model = Qwen25OmniAudioModel()
    model._model, model._processor = _make_mock_model_and_processor()

    audio_path = str(tmp_path / "question.wav")
    sf.write(audio_path, np.zeros(16000, dtype=np.int16), 16000)

    response_path, _ = model.respond(audio_path)

    data, sr = sf.read(response_path)
    assert sr == QWEN_OUTPUT_SAMPLE_RATE == 24000
    assert len(data) > 0


def test_respond_uses_qwens_own_default_system_prompt_when_unset(tmp_path):
    """Confirmed as a real constraint from the live test, not a guess:
    Qwen's own warning states audio output quality is only guaranteed
    with its default system prompt. system_instruction=None (the
    default) must use that exact default, not an empty/generic one."""
    model = Qwen25OmniAudioModel()  # system_instruction defaults to None
    model._model, model._processor = _make_mock_model_and_processor()

    audio_path = str(tmp_path / "question.wav")
    sf.write(audio_path, np.zeros(16000, dtype=np.int16), 16000)

    model.respond(audio_path)

    conversation = model._processor.apply_chat_template.call_args[0][0]
    system_text = conversation[0]["content"][0]["text"]
    assert "Qwen" in system_text and "Alibaba" in system_text


def test_respond_uses_custom_system_instruction_when_set(tmp_path):
    """The override is real and reaches the conversation -- but see the
    class docstring: this carries a real, stated risk to audio output
    quality per Qwen's own warning, not something to reach for by
    default the way GeminiLiveAudioModel's system_instruction is."""
    model = Qwen25OmniAudioModel(system_instruction="Be extremely concise.")
    model._model, model._processor = _make_mock_model_and_processor()

    audio_path = str(tmp_path / "question.wav")
    sf.write(audio_path, np.zeros(16000, dtype=np.int16), 16000)

    model.respond(audio_path)

    conversation = model._processor.apply_chat_template.call_args[0][0]
    system_text = conversation[0]["content"][0]["text"]
    assert system_text == "Be extremely concise."


def test_respond_calls_load_if_not_already_loaded(tmp_path, monkeypatch):
    model = Qwen25OmniAudioModel()
    assert model._model is None

    def fake_load():
        model._model, model._processor = _make_mock_model_and_processor()

    monkeypatch.setattr(model, "load", fake_load)

    audio_path = str(tmp_path / "question.wav")
    sf.write(audio_path, np.zeros(16000, dtype=np.int16), 16000)

    model.respond(audio_path)

    assert model._model is not None


def test_respond_clears_cuda_cache_after_each_call(tmp_path, monkeypatch):
    """Direct regression test for a real, confirmed bug: peak VRAM
    climbed call over call on a live benchmark run (~12.6GB on the first
    call to a CUDA OOM by the third), because generate()'s intermediate
    tensors weren't being released between calls in the same process.
    Confirms respond() actually calls torch.cuda.empty_cache() (and
    gc.collect()) after every call, not just once at some other point."""
    import torch

    model = Qwen25OmniAudioModel()
    model._model, model._processor = _make_mock_model_and_processor()

    audio_path = str(tmp_path / "question.wav")
    sf.write(audio_path, np.zeros(16000, dtype=np.int16), 16000)

    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(1))

    model.respond(audio_path)
    model.respond(audio_path)

    assert len(empty_cache_calls) == 2  # once per respond() call, not just the first
