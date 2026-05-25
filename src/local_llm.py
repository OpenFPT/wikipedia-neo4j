"""Local SLM wrapper for Qwen2.5-7B-Instruct (4-bit NF4 quantization)."""

from __future__ import annotations

from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

_model = None
_tokenizer = None


def _load_model():
    """Load Qwen2.5-7B-Instruct with 4-bit NF4 quantization."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = settings.local_model_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading local model", extra={"model": model_id})

    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load LoRA adapter if configured
    if settings.lora_adapter_path:
        from peft import PeftModel

        logger.info("Loading LoRA adapter", extra={"path": settings.lora_adapter_path})
        _model = PeftModel.from_pretrained(_model, settings.lora_adapter_path)  # type: ignore[assignment]

    logger.info("Local model loaded", extra={"model": model_id})
    return _model, _tokenizer


def generate(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate text from a prompt using the local model."""
    import torch

    model, tokenizer = _load_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def chat(messages: list[dict[str, str]], max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate a chat response from a list of messages.

    Messages format: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    import torch

    model, tokenizer = _load_model()

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)
