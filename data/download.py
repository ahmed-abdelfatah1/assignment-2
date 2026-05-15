"""Phase 1: pull the MIMIC-CXR Kaggle dataset, sample 400 pairs, build manifests.

Usage:
    python data/download.py                # full run: 400 pairs, 300 index / 100 test
    python data/download.py --inspect      # print dataset tree summary and exit
    python data/download.py --n 20         # quick smoke test with 20 pairs

Auth:
    Reads KAGGLE_USERNAME / KAGGLE_KEY from a local .env file (python-dotenv) so
    the same script works locally and on Colab. On Colab the kaggle vars are
    typically set from userdata in the notebook before this script runs;
    load_dotenv() is a no-op there.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# Load .env at import time so KAGGLE_* and HF_TOKEN are visible to kagglehub
# and any downstream HF calls regardless of how this script is invoked.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("download")

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "config.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# Columns we'll probe for in any candidate metadata CSV.
REPORT_TEXT_COLS = ["text", "report", "findings", "impression", "Report", "REPORT", "FINDINGS"]
IMAGE_REF_COLS = ["image", "image_path", "dicom_id", "filename", "file_name", "id", "Image", "path"]


def load_config() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inspect_dataset(root: Path) -> None:
    """Print a per-extension breakdown of the downloaded dataset."""
    log.info("Inspecting %s", root)
    by_ext: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file():
            by_ext.setdefault(p.suffix.lower(), []).append(p)
    for ext, files in sorted(by_ext.items(), key=lambda kv: -len(kv[1])):
        log.info("  %-8s %5d files", ext or "<none>", len(files))
        for sample in files[:3]:
            log.info("           e.g. %s", sample.relative_to(root))


def find_metadata_csv(root: Path) -> Path | None:
    """Return the best-scoring CSV with plausible report-text + image-ref columns."""
    csvs = list(root.rglob("*.csv"))
    if not csvs:
        return None
    scored = []
    for c in csvs:
        try:
            head = pd.read_csv(c, nrows=5)
        except Exception as e:
            log.warning("  unreadable CSV %s: %s", c.name, e)
            continue
        cols = set(head.columns)
        has_text = any(col in cols for col in REPORT_TEXT_COLS)
        has_ref = any(col in cols for col in IMAGE_REF_COLS)
        scored.append(((has_text, has_ref, c.stat().st_size), c))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    if scored and scored[0][0][0]:  # require at least a text column
        winner = scored[0][1]
        log.info("Picked metadata CSV: %s", winner.relative_to(root))
        return winner
    return None


def index_images(root: Path) -> dict[str, Path]:
    """Map filename-stem -> first matching image path under root."""
    out: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.setdefault(p.stem, p)
    return out


def build_pairs_from_csv(csv_path: Path, dataset_root: Path) -> list[dict]:
    log.info("Reading metadata CSV: %s", csv_path.name)
    df = pd.read_csv(csv_path)
    log.info("  columns: %s", list(df.columns))
    log.info("  rows   : %d", len(df))

    text_col = next((c for c in REPORT_TEXT_COLS if c in df.columns), None)
    ref_col = next((c for c in IMAGE_REF_COLS if c in df.columns), None)
    if text_col is None:
        raise RuntimeError(
            f"No report-text column in {csv_path.name}; tried {REPORT_TEXT_COLS}"
        )
    if ref_col is None:
        log.warning("No image-ref column; will pair by row index / scanning images by stem")

    images_by_stem = index_images(dataset_root)
    log.info("Indexed %d images by stem", len(images_by_stem))

    pairs: list[dict] = []
    skipped = {"empty_report": 0, "no_image": 0}
    for _, row in df.iterrows():
        report = ""
        if pd.notna(row[text_col]):
            report = str(row[text_col]).strip()
        if not report:
            skipped["empty_report"] += 1
            continue

        stem = None
        if ref_col is not None and pd.notna(row[ref_col]):
            stem = Path(str(row[ref_col]).strip()).stem
        img = images_by_stem.get(stem) if stem else None
        if img is None:
            skipped["no_image"] += 1
            continue
        pairs.append({"id": stem, "image_src": img, "report": report})

    log.info("  paired : %d", len(pairs))
    log.info("  skipped: %s", skipped)
    return pairs


def build_pairs_from_txt(dataset_root: Path) -> list[dict]:
    """Fallback when no metadata CSV exists: pair each image with same-stem .txt."""
    log.info("No usable CSV — falling back to image<->.txt pairing by stem")
    images_by_stem = index_images(dataset_root)
    txt_by_stem: dict[str, Path] = {}
    for p in dataset_root.rglob("*.txt"):
        txt_by_stem.setdefault(p.stem, p)

    pairs: list[dict] = []
    skipped = {"empty_report": 0, "no_txt": 0}
    for stem, img in images_by_stem.items():
        txt = txt_by_stem.get(stem)
        if txt is None:
            skipped["no_txt"] += 1
            continue
        report = txt.read_text(encoding="utf-8", errors="ignore").strip()
        if not report:
            skipped["empty_report"] += 1
            continue
        pairs.append({"id": stem, "image_src": img, "report": report})
    log.info("  paired : %d", len(pairs))
    log.info("  skipped: %s", skipped)
    return pairs


def image_is_loadable(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inspect", action="store_true",
        help="download then print directory summary and exit",
    )
    ap.add_argument(
        "--n", type=int, default=None,
        help="override total sample size (split is rescaled proportionally)",
    )
    args = ap.parse_args()

    cfg = load_config()
    dc = cfg["data"]

    total_n = args.n or dc["total_samples"]
    if args.n:
        index_n = max(1, int(round(total_n * dc["index_split"] / dc["total_samples"])))
        test_n = max(1, total_n - index_n)
    else:
        index_n, test_n = dc["index_split"], dc["test_split"]
    seed = dc["seed"]

    sample_dir = REPO_ROOT / dc["sample_dir"]
    images_dir = REPO_ROOT / dc["images_dir"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError:
        log.error("kagglehub not installed — run `pip install -r requirements.txt`")
        sys.exit(1)

    log.info("kagglehub: downloading %s", dc["kaggle_slug"])
    dataset_path = Path(kagglehub.dataset_download(dc["kaggle_slug"]))
    log.info("kagglehub: dataset_path = %s", dataset_path)

    if args.inspect:
        inspect_dataset(dataset_path)
        return

    csv_path = find_metadata_csv(dataset_path)
    pairs = build_pairs_from_csv(csv_path, dataset_path) if csv_path else build_pairs_from_txt(dataset_path)
    if not pairs:
        log.error("No (image, report) pairs found — rerun with --inspect to debug layout.")
        sys.exit(1)

    # Deterministic sample of `total_n` pairs.
    df_pairs = pd.DataFrame(pairs)
    if total_n > len(df_pairs):
        log.warning("Requested %d but only %d pairs available; using all of them.", total_n, len(df_pairs))
        total_n = len(df_pairs)
        index_n = min(index_n, max(1, int(round(total_n * 0.75))))
        test_n = total_n - index_n
    df_sampled = df_pairs.sample(n=total_n, random_state=seed).reset_index(drop=True)

    # Copy + validate images, building the final manifest rows.
    rows: list[dict] = []
    bad_imgs = 0
    for entry in tqdm(df_sampled.to_dict(orient="records"), desc="copying images"):
        src: Path = entry["image_src"]
        if not image_is_loadable(src):
            bad_imgs += 1
            continue
        dst = images_dir / f"{entry['id']}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)
        rows.append({
            "id": entry["id"],
            "image_path": dst.relative_to(REPO_ROOT).as_posix(),
            "report": entry["report"],
        })
    log.info("Copied %d images; skipped %d unreadable", len(rows), bad_imgs)

    if not rows:
        log.error("All sampled images failed validation.")
        sys.exit(1)

    # Write manifests. Re-shuffle once with the same seed so the index/test split
    # is independent of the sampling order.
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    actual_index_n = min(index_n, max(1, len(df) - 1))
    actual_test_n = min(test_n, len(df) - actual_index_n)

    df_index = df.iloc[:actual_index_n].copy()
    df_test = df.iloc[actual_index_n:actual_index_n + actual_test_n].copy()

    manifest_all = REPO_ROOT / dc["manifest_all"]
    manifest_index = REPO_ROOT / dc["manifest_index"]
    manifest_test = REPO_ROOT / dc["manifest_test"]

    df.to_csv(manifest_all, index=False, quoting=csv.QUOTE_ALL)
    df_index.to_csv(manifest_index, index=False, quoting=csv.QUOTE_ALL)
    df_test.to_csv(manifest_test, index=False, quoting=csv.QUOTE_ALL)

    log.info("manifest_all   : %d rows -> %s", len(df), manifest_all.relative_to(REPO_ROOT))
    log.info("manifest_index : %d rows -> %s", len(df_index), manifest_index.relative_to(REPO_ROOT))
    log.info("manifest_test  : %d rows -> %s", len(df_test), manifest_test.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
