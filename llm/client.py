from __future__ import annotations
import platform
import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_IS_MAC = platform.system() == "Darwin"


class LLMClient:
    """
    Causal LM wrapper with two backends:
      - CUDA (Colab/AWS): HuggingFace transformers + bitsandbytes 4-bit NF4
      - Apple Silicon (Mac): mlx-lm with pre-quantized MLX models

    Mac model name should be an MLX-quantized HF repo, e.g.:
      'mlx-community/Qwen3-4B-4bit'
    Colab/CUDA model name is the standard HF repo, e.g.:
      'Qwen/Qwen3-4B'
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        load_in_4bit: bool = True,
    ):
        self._backend = "mlx" if _IS_MAC else "transformers"

        if self._backend == "mlx":
            # self.model_name must reflect what's actually loaded, not the
            # raw (possibly None) constructor argument -- otherwise
            # llm.model_name prints "None" even when a real fallback model
            # was loaded, hiding which model a run actually used.
            self.model_name = model_name or "mlx-community/Qwen3-4B-4bit"
            self._init_mlx(self.model_name)
        else:
            self.model_name = model_name or "Qwen/Qwen3-4B"
            self._init_transformers(self.model_name, device, load_in_4bit)

    # ── MLX backend (Apple Silicon) ───────────────────────────────────────────

    def _init_mlx(self, model_name: str):
        from mlx_lm import load
        self.model, self.tokenizer = load(model_name)
        self.device = "mps"

    def _generate_mlx(self, system: str, messages: list[dict], max_tokens: int) -> str:
        from mlx_lm import generate
        chat = [{"role": "system", "content": system}] + messages
        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        out = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        return _THINK_RE.sub("", out).strip()

    # ── Transformers backend (CUDA / CPU) ─────────────────────────────────────

    def _init_transformers(self, model_name: str, device: str | None, load_in_4bit: bool):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if load_in_4bit and self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quant_config,
                device_map="auto", trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device != "cuda":
                self.model = self.model.to(self.device)
        self.model.eval()

    def _generate_transformers(self, system: str, messages: list[dict], max_tokens: int) -> str:
        import torch
        chat = [{"role": "system", "content": system}] + messages
        text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][input_len:]
        out = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _THINK_RE.sub("", out).strip()

    def _generate_transformers_batch(
        self, system: str, messages_list: list[list[dict]], max_tokens: int
    ) -> list[str]:
        import torch

        # Causal-LM batched generation requires left-padding so every row's
        # real content ends at the same position -- the model would
        # otherwise keep generating mid-sequence for shorter, right-padded
        # rows instead of continuing past their actual last token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        texts = []
        for messages in messages_list:
            chat = [{"role": "system", "content": system}] + messages
            texts.append(self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            ))

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        outputs = []
        for row in output_ids:
            new_tokens = row[input_len:]
            out = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outputs.append(_THINK_RE.sub("", out).strip())
        return outputs

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, system: str, messages: list[dict], max_tokens: int = 512) -> str:
        if self._backend == "mlx":
            return self._generate_mlx(system, messages, max_tokens)
        return self._generate_transformers(system, messages, max_tokens)

    def generate_batch(
        self, system: str, messages_list: list[list[dict]], max_tokens: int = 512
    ) -> list[str]:
        """
        Batched generation: one forward pass covering many independent
        prompts instead of one generate() call each. Only meaningfully
        parallel on the transformers/CUDA backend -- a single small model
        badly underutilizes GPU compute one sequence at a time. MLX falls
        back to a sequential loop (no batched decode path here).
        """
        if not messages_list:
            return []
        if self._backend == "mlx":
            return [self._generate_mlx(system, messages, max_tokens) for messages in messages_list]
        return self._generate_transformers_batch(system, messages_list, max_tokens)

    def unload(self):
        """Free memory — call before loading STT or TTS."""
        import gc
        del self.model
        gc.collect()
        if self._backend == "transformers":
            import torch
            torch.cuda.empty_cache()
