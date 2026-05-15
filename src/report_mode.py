"""Phase 2: Report-generation interface.

Two functions, common return shape:
    {"model": str, "report": str, "latency_s": float, "extras": dict}

Run as `python -m src.report_mode` (or directly) to see a side-by-side
comparison of MedGemma vs CLIP retrieval on the first 10 test images.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

# Allow `python src/report_mode.py` as well as `python -m src.report_mode`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CFG_PATH = _REPO_ROOT / "config.yaml"

REPORT_PROMPT = (
    "You are an expert radiologist. Write a concise chest X-ray report for the image. "
    "Use two sections: 'Findings:' (one paragraph) and 'Impression:' (one short paragraph). "
    "Do not invent patient identifiers or clinical history."
)


def _load_config() -> dict:
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_report_medgemma(image: Image.Image) -> dict:
    """Zero-shot MedGemma report from the image."""
    from src.models import medgemma  # lazy import so CLIP path doesn't load MedGemma
    cfg = _load_config()
    t0 = time.perf_counter()
    text = medgemma.generate(image, REPORT_PROMPT)
    return {
        "model": cfg["models"]["medgemma"]["hf_id"],
        "report": text,
        "latency_s": time.perf_counter() - t0,
        "extras": {},
    }


def generate_report_clip(image: Image.Image) -> dict:
    """Retrieval baseline: nearest training image's report verbatim."""
    from src.models import clip_retriever
    t0 = time.perf_counter()
    top = clip_retriever.query(image, top_k=1)[0]
    return {
        "model": "openclip-vit-b-32-retrieval",
        "report": top["report"],
        "latency_s": time.perf_counter() - t0,
        "extras": {
            "retrieved_id": top["id"],
            "similarity": top["similarity"],
            "retrieved_image": top["image_path"],
        },
    }


def _truncate(s: str, n: int = 400) -> str:
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def _main() -> None:
    cfg = _load_config()
    test_csv = _REPO_ROOT / cfg["data"]["manifest_test"]
    df = pd.read_csv(test_csv).head(10).reset_index(drop=True)
    print(f"Comparing on {len(df)} test images from {test_csv.name}")

    # Explicit warmup: CLIP first (model + index), then MedGemma. Keeps the
    # per-image latency numbers below clean (no first-call load bleed) and
    # makes the "what's loading right now" picture obvious in the log.
    from src.models import clip_retriever, medgemma

    print("\n[warmup] Loading CLIP (model + index)...")
    t0 = time.perf_counter()
    clip_retriever._ensure_index()
    print(f"[warmup] CLIP ready in {time.perf_counter() - t0:.1f}s")

    print("[warmup] Loading MedGemma (4-bit)...")
    t0 = time.perf_counter()
    medgemma._ensure_loaded()
    print(f"[warmup] MedGemma ready in {time.perf_counter() - t0:.1f}s\n")

    for i, row in df.iterrows():
        img_path = _REPO_ROOT / row["image_path"]
        try:
            with Image.open(img_path) as raw:
                img = raw.convert("RGB").copy()
        except Exception as e:
            print(f"\n[{i+1}/{len(df)}] {row['id']}  IMAGE LOAD FAILED: {e}")
            continue

        print(f"\n=== [{i+1}/{len(df)}] id={row['id']}  view={row.get('view_used','?')} ===")
        print(f"--- gold ---\n{_truncate(row['report'])}")

        try:
            r1 = generate_report_clip(img)
            sim = r1["extras"]["similarity"]
            print(f"--- {r1['model']}  ({r1['latency_s']:.2f}s, sim={sim:.3f}) ---")
            print(_truncate(r1["report"]))
        except Exception as e:
            print(f"CLIP error: {e}")

        try:
            r2 = generate_report_medgemma(img)
            print(f"--- {r2['model']}  ({r2['latency_s']:.2f}s) ---")
            print(_truncate(r2["report"]))
        except Exception as e:
            print(f"MedGemma error: {e}")


if __name__ == "__main__":
    _main()
