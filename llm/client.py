from __future__ import annotations


class LLMClient:
    """
    Thin wrapper around a causal LM loaded via HuggingFace transformers.

    Default model: Qwen/Qwen3-8B (fits on A100 alongside STT + TTS with care).
    Falls back gracefully — swap model_name for Mistral-7B or any compatible model.

    On machines without CUDA the model runs on CPU in fp32 (slow but functional
    for unit tests with a small model like Qwen/Qwen3-0.6B).
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-8B", device: str | None = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
            chat, tokenize=False, add_generation_prompt=True
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
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
