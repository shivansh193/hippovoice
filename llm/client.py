from __future__ import annotations
import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMClient:
    """
    Thin wrapper around a causal LM loaded via HuggingFace transformers.

    Default: Qwen/Qwen3-4B in 4-bit quantization (~3GB VRAM) — fits on T4 alongside other models.
    Set load_in_4bit=False and model_name='Qwen/Qwen3-8B' for A100.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        device: str | None = None,
        load_in_4bit: bool = True,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.model_name = model_name
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
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
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

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 512,
    ) -> str:
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
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _THINK_RE.sub("", text).strip()

    def unload(self):
        """Free VRAM — call before loading STT or TTS on T4."""
        import torch, gc
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
