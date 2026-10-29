"""LLM pequeno (Qwen3.5 etc.) via transformers, zero-shot ou com adapter LoRA."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from review_intel.core.device import best_device
from review_intel.core.predictors import Prediction, Predictor, as_dict
from review_intel.core.prompts import chat_messages
from review_intel.core.schema import parse_output


class HFGenerator(Predictor):
    def __init__(self, model_id: str, adapter: str | None = None, device: str | None = None,
                 max_new_tokens: int = 160, name: str | None = None):
        self.device = device or best_device()
        self.tok = AutoTokenizer.from_pretrained(adapter or model_id)
        self.tok.padding_side = "left"  # geração em lote precisa de padding à esquerda
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        short = model_id.split("/")[-1]
        self.name = name or (f"{short} + LoRA" if adapter else f"{short} zero-shot")

    def prompts(self, texts: list[str]) -> list[str]:
        return [self.tok.apply_chat_template(chat_messages(t), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for t in texts]

    @torch.inference_mode()
    def generate(self, texts: list[str]) -> list[str]:
        enc = self.tok(self.prompts(texts), return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def predict(self, texts):
        return [Prediction(as_dict(parse_output(g))) for g in self.generate(texts)]
