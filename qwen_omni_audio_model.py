"""
Qwen25OmniAudioModel -- real AudioToAudioModel implementation wrapping
Qwen2.5-Omni-3B (self-hosted via HuggingFace transformers), an
alternative backend to GeminiLiveAudioModel for pipeline_audio2audio.py.

Confirmed as a real, viable backend via a live GPU test (a throwaway AWS
g4dn.xlarge, provisioned via the AWS CLI, torn down immediately after --
no lingering instance, security group, or IAM role) before writing any
of this adapter code, same "validate cheap before expensive" discipline
as every other new backend in this project (see BUGS.md). That real run:
loaded in ~120s (one-time cost, not per-question), answered a real
spoken question correctly ("What is the capital of France?" -> "Paris"),
used 12.6GB peak VRAM (fits a single T4's ~15GB with real headroom), and
completed the actual generate() call in 9.7s -- roughly 4x faster than
GeminiLiveAudioModel's confirmed real 35-45s/question (see that module's
own docstring for where that number comes from).

Chosen over the 7B variant specifically for VRAM headroom: 7B's stated
BF16 figures on its HF card (31GB for a 15s *video* clip) are for
video+audio combined, heavier than this pipeline's audio-only short
turns, but its plain weights alone (~14-15GB) leave essentially no room
on a T4's ~14.5GB usable (confirmed via a real OOM elsewhere in this
project -- see baselines/easyedit_weight_editor.py). 3B's ~6GB of
weights leaves real headroom to actually find out if it fits, rather
than gambling the whole test on 7B's edge case.

Chosen over Moshi and Amazon Nova Sonic (both investigated first, real
research not a guess) because those are full-duplex/continuous
conversational models with no clean per-turn completion signal -- Nova
Sonic's own official reference sample never even checks for one, it just
plays audio through speakers until a human presses Enter, and its
session model has an 8-minute connection cap that only makes sense for a
live continuous call. Qwen2.5-Omni's model.generate() is a genuine
synchronous call (confirmed from its own HF model card, not guessed):
send one input, get back one complete (text_ids, audio) result. No
streaming ambiguity, maps directly onto this project's own
AudioToAudioModel.respond() contract -- unlike Nova Sonic/Moshi, no
silence-based heuristic segmentation is needed (and the correctness risk
that would carry, since a wrong heuristic could silently truncate or
run past the real answer without ever raising an error).

Needs a specific install, not a plain requirements.txt entry -- the
preview transformers branch conflicts with this project's normal
transformers>=4.40.0 pin, so bundling it there would break everyone
else's install. Confirmed from Qwen's own HF card, plus a real install
error hit during the live test (qwen-omni-utils doesn't pull in its own
audioread dependency despite needing it):
    pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
    pip install accelerate "qwen-omni-utils[decord]" audioread librosa soundfile

Needs a GPU -- not exercised by the local test suite at all (same
reasoning as EasyEditWeightEditor: only actually instantiating this
needs the real model/torch/CUDA installed; the local suite mocks the
model/processor to cover this file's own parsing/plumbing logic
instead). Validated for real on the throwaway GPU instance described
above, not guessed.

return_audio=False (constructor param, default True to preserve a live
conversational deployment's actual need for spoken audio out) is a
second, real fix discovered on a live full-benchmark re-run, distinct
from the per-call cache-clearing and the reverted expandable_segments
attempt: with the confirmed-good fixes (concise system_instruction,
cache-clearing) in place and no expandable_segments, the very FIRST
generate() call on a real LoCoMo question still OOM'd -- "Tried to
allocate 2.00 GiB. GPU 0 has a total capacity of 14.56 GiB ... this
process has 13.35 GiB memory in use" -- inside self.token2wav(...)'s
DiT-based vocoder (see the traceback in BUGS.md). That 2.00 GiB figure
lines up exactly with Qwen's own documented savings from
return_audio=False ("This option will save about ~2GB of GPU memory",
per its HF model card) -- real corroborating evidence, not a guess,
that the audio-decoding step itself (not a cross-call leak, already
mitigated separately) was what pushed a single T4's 14.56GB over the
edge on real conversation-length input. Since this benchmark only
scores the TEXT transcript and always discards the response audio path
(see benchmarks/locomo/evaluate_audio.py), there's no real audio output
to lose by setting this False for benchmark runs specifically -- a live
conversational deployment would still want it True.
"""

QWEN_OUTPUT_SAMPLE_RATE = 24000


class Qwen25OmniAudioModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-3B",
        speaker: str = "Chelsie",
        system_instruction: str | None = None,
        return_audio: bool = True,
    ):
        # system_instruction stays None by default deliberately: a real,
        # confirmed warning from the live test says audio output quality
        # is only guaranteed with Qwen's own default system prompt
        # ("System prompt modified, audio output may not work as
        # expected"). Overriding it (e.g. for a concise-answer benchmark
        # prompt, the same pattern used in gemini_live_model.py) is
        # possible here but carries that real, stated risk -- flagged,
        # not silently ignored, since it directly trades off against
        # LoCoMo's strict token-F1 scorer the same way Gemini's own
        # verbosity issue did.
        self.model_name = model_name
        self.speaker = speaker
        self.system_instruction = system_instruction
        # See module docstring: real, confirmed to save ~2GB of GPU
        # memory by skipping the token2wav vocoder step entirely. Only
        # set False for a use case (like a benchmark) that discards the
        # response audio anyway -- a live deployment needs real speech
        # out, so True stays the default.
        self.return_audio = return_audio
        self._model = None
        self._processor = None

    def load(self):
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True was tried here
        # after a live run hit CUDA OOM by question 15 (the OOM's own
        # error message suggested it, and this project's earlier
        # WeightEdit OOM used the same knob -- see
        # baselines/easyedit_weight_editor.py). Confirmed a REGRESSION on
        # a real re-run, not an improvement: with it set, the very FIRST
        # generate() call OOM'd, with the allocator's own log showing
        # "expandable_segments: memory mapping failed" twice before the
        # fatal error -- worse than the question-15 failure without it.
        # expandable_segments' virtual-memory-mapping strategy apparently
        # doesn't suit this specific model's allocation pattern. Left
        # deliberately unset; per-call cache-clearing in respond() below
        # is the mitigation that's actually confirmed to help (took a
        # real run from crashing at question 3 to question 15).
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self._processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)

    def respond(self, audio_path: str) -> tuple[str, str]:
        """
        One real audio-in/audio-out turn. Returns (response_audio_path,
        transcript) matching AudioToAudioModel's contract exactly --
        model.generate() being a genuine synchronous call (confirmed
        real, see module docstring) means this needs none of
        GeminiLiveAudioModel's asyncio/thread-pool machinery for calling
        an async API from a sync context; it's a plain blocking call.
        """
        if self._model is None:
            self.load()

        from qwen_omni_utils import process_mm_info

        default_system = (
            "You are Qwen, a virtual human developed by the Qwen Team, "
            "Alibaba Group, capable of perceiving auditory and visual "
            "inputs, as well as generating text and speech."
        )
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_instruction or default_system}],
            },
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio_path}],
            },
        ]

        text = self._processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self._processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        )
        inputs = inputs.to(self._model.device).to(self._model.dtype)

        # Real, confirmed API shape difference (see module docstring):
        # generate() returns a (text_ids, audio) tuple when return_audio
        # is True, but ONLY text_ids (no tuple at all) when False -- not
        # (text_ids, None). Branching on self.return_audio rather than
        # inspecting the return value's shape, since a tuple-vs-bare-
        # tensor check would be guessing at an internal detail that's
        # already known from Qwen's own documented contract.
        if self.return_audio:
            text_ids, audio = self._model.generate(
                **inputs, use_audio_in_video=False, speaker=self.speaker, return_audio=True,
            )
        else:
            text_ids = self._model.generate(
                **inputs, use_audio_in_video=False, speaker=self.speaker, return_audio=False,
            )
            audio = None

        # Slicing off the prompt's own token length before decoding, not
        # string-splitting on "assistant\n" -- confirmed real from the
        # live test that text_ids includes the whole rendered chat
        # template (system/user/assistant) followed by the actual
        # completion, e.g. "system\n...\nuser\n\nassistant\nParis". Token
        # slicing gets exactly the newly generated continuation
        # regardless of what words happen to appear in the system/user
        # portions, which naive string-matching on a role name wouldn't
        # reliably guarantee.
        prompt_len = inputs.input_ids.shape[1]
        generated_ids = text_ids[:, prompt_len:]
        transcript = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if self.return_audio:
            import tempfile
            import soundfile as sf

            response_path = tempfile.mktemp(suffix=".wav")
            sf.write(response_path, audio.reshape(-1).detach().cpu().numpy(), samplerate=QWEN_OUTPUT_SAMPLE_RATE)
        else:
            # No audio was generated -- nothing to save. Callers that
            # only need the transcript (every current caller in this
            # project: evaluate_audio.py discards this path entirely,
            # see module docstring) are fine with None here; a live
            # deployment wanting real audio out must leave return_audio
            # at its True default instead of handling None itself.
            response_path = None

        # Confirmed as a real, distinct bug on a live benchmark run, not
        # theoretical: peak VRAM climbed from ~12.6GB (the very first
        # call, see module docstring) to a CUDA OOM by the third call in
        # the same process -- generate()'s intermediate tensors (the
        # DiT-based Token2Wav vocoder's own working memory in particular,
        # per the OOM traceback) weren't being released between calls.
        # Dropping references and clearing PyTorch's caching allocator
        # here is the standard mitigation for exactly this shape of
        # problem; unlike a precision change, this has zero effect on the
        # actual output, only on how much GPU memory sits idle-but-
        # reserved between calls.
        del text_ids, generated_ids, audio, inputs
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return response_path, transcript
