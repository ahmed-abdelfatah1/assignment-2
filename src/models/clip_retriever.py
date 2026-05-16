"""OpenCLIP ViT-B/32 image-embedding retrieval baseline.

Used as the lighter-weight comparison for MedGemma report generation:
embed each training image, FAISS-index them, and for a new image return
the nearest training image's report verbatim.

Public API:
    build_index(manifest_path=None) -> None
    query(image: PIL.Image, top_k: int = 1) -> list[dict]

The index is persisted alongside a manifest snapshot so retrieval doesn't
depend on the source manifest CSV being unchanged between builds.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CFG_PATH = _REPO_ROOT / "config.yaml"

_state: dict = {"model": None, "preprocess": None, "index": None, "manifest": None}
# RLock (not Lock): _ensure_index acquires it and calls build_index, which calls
# _ensure_model, which tries to acquire it again on the same thread. A plain
# threading.Lock would deadlock there.
_lock = threading.RLock()


def _load_config() -> dict:
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. On Colab: Runtime > Change runtime type > T4 GPU."
        )
    return "cuda"


def _ensure_model():
    if _state["model"] is not None:
        return _state["model"], _state["preprocess"]
    with _lock:
        if _state["model"] is not None:
            return _state["model"], _state["preprocess"]
        import time as _t
        cfg = _load_config()
        mc = cfg["models"]["clip"]
        # hf-hub:* models (e.g. BiomedCLIP) load via create_model_from_pretrained
        # (returns 2 values); classic open_clip checkpoints use
        # create_model_and_transforms (returns 3).
        print(f"[clip] loading {mc['model_name']} (pretrained={mc.get('pretrained')})...", flush=True)
        t0 = _t.time()
        if mc["model_name"].startswith("hf-hub:"):
            model, preprocess = open_clip.create_model_from_pretrained(mc["model_name"])
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                mc["model_name"], pretrained=mc["pretrained"]
            )
        print(f"[clip] model built in {_t.time()-t0:.1f}s", flush=True)
        device = _device()
        print(f"[clip] moving to {device}...", flush=True)
        t0 = _t.time()
        model = model.to(device)
        model.train(False)
        print(f"[clip] on {device} in {_t.time()-t0:.1f}s", flush=True)
        _state["model"] = model
        _state["preprocess"] = preprocess
        return model, preprocess


def _embed_image(img: Image.Image) -> np.ndarray:
    """Return a 1xD float32 L2-normalized embedding for cosine-sim retrieval."""
    model, preprocess = _ensure_model()
    device = _device()
    with torch.inference_mode():
        x = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
        feats = model.encode_image(x).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


def _index_paths() -> tuple[Path, Path]:
    cfg = _load_config()
    index_path = _REPO_ROOT / cfg["models"]["clip"]["index_path"]
    manifest_snapshot = index_path.with_suffix(".manifest.csv")
    return index_path, manifest_snapshot


def build_index(manifest_path: Optional[Path] = None, batch_size: Optional[int] = None) -> None:
    """Embed every image in the manifest and persist a FAISS index + manifest snapshot.

    Batched encoding: preprocess images on CPU, stack into a tensor, run a single
    forward pass per batch. On a T4, batch=32 takes single-digit seconds for 300 images.
    """
    cfg = _load_config()
    if manifest_path is None:
        manifest_path = _REPO_ROOT / cfg["data"]["manifest_index"]
    if batch_size is None:
        batch_size = cfg["models"]["clip"].get("embed_batch_size", 32)
    index_path, manifest_snapshot = _index_paths()

    df = pd.read_csv(manifest_path)
    model, preprocess = _ensure_model()
    device = _device()

    keep_rows: list[int] = []
    embeddings_chunks: list[np.ndarray] = []
    pending_tensors: list[torch.Tensor] = []
    pending_idxs: list[int] = []

    def _flush():
        if not pending_tensors:
            return
        batch = torch.stack(pending_tensors).to(device)
        with torch.inference_mode():
            feats = model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings_chunks.append(feats.cpu().numpy().astype("float32"))
        keep_rows.extend(pending_idxs)
        pending_tensors.clear()
        pending_idxs.clear()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="CLIP embedding"):
        img_path = _REPO_ROOT / row["image_path"]
        try:
            with Image.open(img_path) as im:
                tensor = preprocess(im.convert("RGB"))
        except Exception as e:
            print(f"WARN: skipping {img_path}: {e}")
            continue
        pending_tensors.append(tensor)
        pending_idxs.append(i)
        if len(pending_tensors) >= batch_size:
            _flush()
    _flush()

    if not embeddings_chunks:
        raise RuntimeError(f"No embeddable images under {manifest_path}.")
    X = np.vstack(embeddings_chunks)
    df_aligned = df.iloc[keep_rows].reset_index(drop=True)

    index = faiss.IndexFlatIP(X.shape[1])  # cosine sim via L2-normalized inner product
    index.add(X)
    _state["index"] = index
    _state["manifest"] = df_aligned

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    df_aligned.to_csv(manifest_snapshot, index=False)
    print(f"Wrote {index.ntotal} vectors to {index_path}")


def _ensure_index() -> tuple:
    if _state["index"] is not None and _state["manifest"] is not None:
        return _state["index"], _state["manifest"]
    with _lock:
        if _state["index"] is not None and _state["manifest"] is not None:
            return _state["index"], _state["manifest"]
        index_path, manifest_snapshot = _index_paths()
        if index_path.exists() and manifest_snapshot.exists():
            _state["index"] = faiss.read_index(str(index_path))
            _state["manifest"] = pd.read_csv(manifest_snapshot)
        else:
            build_index()
        return _state["index"], _state["manifest"]


def query(image: Image.Image, top_k: int = 1) -> list[dict]:
    """Return top_k nearest training rows for the query image."""
    index, manifest = _ensure_index()
    emb = _embed_image(image)
    sims, idxs = index.search(emb, top_k)
    out: list[dict] = []
    for sim, idx in zip(sims[0], idxs[0]):
        row = manifest.iloc[int(idx)]
        out.append({
            "id": row["id"],
            "report": row["report"],
            "similarity": float(sim),
            "image_path": row["image_path"],
        })
    return out


if __name__ == "__main__":
    build_index()
