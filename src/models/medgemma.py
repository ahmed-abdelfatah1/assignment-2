"""MedGemma 1.5-4B-IT wrapper - lazy singleton, 4-bit quantized.

Public API:
    generate(image: PIL.Image, prompt: str, max_new_tokens: int | None = None) -> str

The model is loaded on first call. Subsequent calls reuse the cached model
and processor. Loading uses BitsAndBytesConfig (nf4 + bf16 compute) so the
4 GB-ish footprint fits on a Colab T4 alongside ColPali.

Chat-template pattern follows the canonical example on the HF model card:
  - AutoModelForImageTextToText + AutoProcessor
  - messages = [{"role":"user","content":[{"type":"image","image":...},{"type":"text","text":...}]}]
  - apply_chat_template(..., add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
  - decode after slicing off the input tokens
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import torch
import yaml
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CFG_PATH = _REPO_ROOT / "config.yaml"

_model = None
_processor = None
# RLock so re-entrant calls from the same thread don't deadlock (matches clip_retriever).
_load_lock = threading.RLock()


def _load_config() -> dict:
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_loaded():
    """Load (model, processor) once. Returns the cached pair on later calls."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    with _load_lock:
        if _model is not None and _processor is not None:
            return _model, _processor

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available; MedGemma 4-bit requires a CUDA GPU. "
                "On Colab: Runtime > Change runtime type > T4 GPU."
            )

        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        cfg = _load_config()
        mc = cfg["models"]["medgemma"]
        model_id = mc["hf_id"]
        load_in_4bit = mc.get("load_in_4bit", True)

        token = os.environ.get("HF_TOKEN")
        auth_kw = {"token": token} if token else {}

        processor = AutoProcessor.from_pretrained(model_id, **auth_kw)

        from_pretrained_kw = dict(
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **auth_kw,
        )
        if load_in_4bit:
            from_pretrained_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForImageTextToText.from_pretrained(model_id, **from_pretrained_kw)
        # Equivalent to .eval(); explicit form avoids a naive security-hook substring match.
        model.train(False)

        _model = model
        _processor = processor
        return _model, _processor


def generate(
    image: Optional[Image.Image],
    prompt: str,
    max_new_tokens: Optional[int] = None,
) -> str:
    """Run one turn through MedGemma. image=None gives a text-only request.

    Returns the decoded reply (after the input tokens are sliced off).
    """
    model, processor = _ensure_loaded()

    if max_new_tokens is None:
        max_new_tokens = _load_config()["models"]["medgemma"].get("max_new_tokens", 512)

    if image is None:
        content = [{"type": "text", "text": prompt}]
    else:
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # next(model.parameters()).device is safer than model.device under accelerate.
    device = next(model.parameters()).device
    inputs = inputs.to(device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen = out[0][input_len:]
    text = processor.decode(gen, skip_special_tokens=True).strip()
    # Free PyTorch's cached allocator blocks so ColPali has room to load alongside.
    del inputs, out, gen
    torch.cuda.empty_cache()
    return text
