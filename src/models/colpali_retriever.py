"""ColPali v1.3 retrieval over rendered report pages.

Each radiology report is rendered as a single PNG page using PIL + basic
typography. ColPali (PaliGemma-based, multi-vector late-interaction) embeds
each page via `colpali_engine.models.ColPali`; queries are scored via
processor.score_multi_vector.

Public API:
    build_index(manifest_path=None) -> None
    query(question: str, top_k: int = 3) -> list[dict]

Persistence: data/sample/colpali_index/doc_embeddings.pt (torch tensor) and
manifest.csv (aligned to embedding rows). The model itself uses the HF cache.

Note: colpali-engine MUST be installed from GitHub source for transformers 5.x
LoRA-remapping fixes. PyPI 0.3.16 lags. Per maintainer guidance:
    pip install git+https://github.com/illuin-tech/colpali
"""

from __future__ import annotations

import os
import textwrap
import threading
import time as _t
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CFG_PATH = _REPO_ROOT / "config.yaml"

_state: dict = {
    "model": None,
    "processor": None,
    "doc_embeddings": None,
    "manifest": None,
}
# RLock so _ensure_index -> build_index -> _ensure_model on the same thread
# doesn't deadlock.
_lock = threading.RLock()


def _load_config() -> dict:
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _index_dir() -> Path:
    cfg = _load_config()
    return _REPO_ROOT / cfg["models"]["colpali"]["index_path"]


def _ensure_model():
    if _state["model"] is not None:
        return _state["model"], _state["processor"]
    with _lock:
        if _state["model"] is not None:
            return _state["model"], _state["processor"]
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required for ColPali. On Colab: Runtime > Change runtime type > T4 GPU."
            )

        from colpali_engine.models import ColPali, ColPaliProcessor

        cfg = _load_config()
        mc = cfg["models"]["colpali"]
        model_id = mc["hf_id"]
        load_in_4bit = mc.get("load_in_4bit", False)
        token = os.environ.get("HF_TOKEN")
        auth_kw = {"token": token} if token else {}

        print(f"[colpali] loading {model_id} (4bit={load_in_4bit})...", flush=True)
        t0 = _t.time()
        from_pretrained_kw = dict(
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            **auth_kw,
        )
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            from_pretrained_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = ColPali.from_pretrained(model_id, **from_pretrained_kw)
        model.train(False)
        print(f"[colpali] model ready in {_t.time()-t0:.1f}s", flush=True)

        processor = ColPaliProcessor.from_pretrained(model_id, **auth_kw)
        _state["model"] = model
        _state["processor"] = processor
        return model, processor


def render_report(text: str, size: tuple[int, int]) -> Image.Image:
    """Render a report as a single-page PNG with basic typography (PIL only)."""
    width, height = size
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    try:
        char_w = font.getbbox("M")[2]
    except Exception:
        char_w = 12
    wrap_chars = max(40, (width - 80) // max(1, char_w))

    lines: list[str] = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=wrap_chars) or [""]
        lines.extend(wrapped)

    y = 40
    line_h = 28
    for line in lines:
        if y + line_h > height - 40:
            break
        draw.text((40, y), line, fill="black", font=font)
        y += line_h
    return img


def build_index(manifest_path: Optional[Path] = None, batch_size: int = 1) -> None:
    """Render each report -> ColPali-embed -> persist tensor + manifest snapshot."""
    cfg = _load_config()
    if manifest_path is None:
        manifest_path = _REPO_ROOT / cfg["data"]["manifest_index"]
    page_size = tuple(cfg["models"]["colpali"]["report_image_size"])
    idx_dir = _index_dir()
    idx_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    model, processor = _ensure_model()
    device = next(model.parameters()).device

    chunks: list[torch.Tensor] = []
    keep_rows: list[int] = []
    img_buf: list[Image.Image] = []
    idx_buf: list[int] = []

    def _flush():
        if not img_buf:
            return
        inputs = processor.process_images(img_buf).to(device)
        with torch.no_grad():
            embs = model(**inputs)
        chunks.append(embs.detach().to("cpu"))
        keep_rows.extend(idx_buf)
        img_buf.clear()
        idx_buf.clear()
        del inputs, embs
        torch.cuda.empty_cache()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="ColPali render+embed"):
        try:
            page = render_report(str(row["report"]), page_size)
        except Exception as e:
            print(f"WARN: failed to render {row['id']}: {e}")
            continue
        img_buf.append(page)
        idx_buf.append(i)
        if len(img_buf) >= batch_size:
            _flush()
    _flush()

    if not chunks:
        raise RuntimeError("No ColPali embeddings produced.")
    doc_emb = torch.cat(chunks, dim=0)
    df_aligned = df.iloc[keep_rows].reset_index(drop=True)

    torch.save(doc_emb, idx_dir / "doc_embeddings.pt")
    df_aligned.to_csv(idx_dir / "manifest.csv", index=False)
    _state["doc_embeddings"] = doc_emb
    _state["manifest"] = df_aligned
    print(
        f"[colpali] indexed {doc_emb.shape[0]} pages; "
        f"embeddings shape {tuple(doc_emb.shape)} -> {idx_dir}"
    )


def _ensure_index() -> tuple:
    if _state["doc_embeddings"] is not None and _state["manifest"] is not None:
        return _state["doc_embeddings"], _state["manifest"]
    with _lock:
        if _state["doc_embeddings"] is not None and _state["manifest"] is not None:
            return _state["doc_embeddings"], _state["manifest"]
        idx_dir = _index_dir()
        emb_path = idx_dir / "doc_embeddings.pt"
        manifest_path = idx_dir / "manifest.csv"
        if emb_path.exists() and manifest_path.exists():
            _state["doc_embeddings"] = torch.load(
                emb_path, map_location="cpu", weights_only=True
            )
            _state["manifest"] = pd.read_csv(manifest_path)
        else:
            build_index()
        return _state["doc_embeddings"], _state["manifest"]


def query(question: str, top_k: int = 3) -> list[dict]:
    """ColPali late-interaction retrieval over the persisted doc embeddings."""
    doc_emb, manifest = _ensure_index()
    model, processor = _ensure_model()
    device = next(model.parameters()).device

    if doc_emb.device != device:
        doc_emb = doc_emb.to(device)
        _state["doc_embeddings"] = doc_emb

    q_inputs = processor.process_queries([question]).to(device)
    with torch.no_grad():
        q_emb = model(**q_inputs)

    scores = processor.score_multi_vector(q_emb, doc_emb)  # [1, n_docs]
    scores_np = scores[0].detach().to("cpu").float().numpy()
    top_idx = scores_np.argsort()[::-1][:top_k]

    out: list[dict] = []
    for idx in top_idx:
        row = manifest.iloc[int(idx)]
        out.append({
            "id": row["id"],
            "report": row["report"],
            "score": float(scores_np[idx]),
            "image_path": row["image_path"],
        })
    return out


if __name__ == "__main__":
    build_index()
