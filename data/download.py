"""Phase 1: pull simhadrisadaram/mimic-cxr-dataset and build train/test manifests.

Schema (known): each row is one patient. The CSVs have these columns:
    subject_id (scalar)
    image, view, AP, PA, Lateral, text, text_augment   (stringified Python lists)

Parsing strategy (per user spec):
  1. Concat mimic_cxr_aug_train.csv + mimic_cxr_aug_validate.csv.
  2. Parse the `image`, `PA`, `AP`, `text` cells via Python's ast module
     (safe literal parsing — no code execution). Skip rows that fail.
  3. Keep rows whose parsed `text` list has EXACTLY ONE non-empty entry, where
     "non-empty" means stripped string with 'Findings:' substring OR len > 50.
     This drops shells like 'Findings:  Impression: '.
  4. Pick image by view priority: PA[0] -> AP[0] -> image[0]. Skip if all empty.
  5. Resolve relative path (e.g. files/p10/p10003502/...) under the kagglehub
     dataset root and validate with PIL.Image.verify().
  6. Sample 400 with seed=42, split 300 index / 100 test.

`text_augment` is ignored — those are paraphrases, not gold reports.

Usage:
    python data/download.py                # full run
    python data/download.py --inspect      # tree summary by extension
    python data/download.py --inspect-csv  # parse funnel, no copying
    python data/download.py --n 20         # quick smoke test (split rescales)

Auth: KAGGLE_USERNAME / KAGGLE_KEY (and HF_TOKEN) loaded from .env locally; on
Colab, the notebook pushes Secrets into os.environ before invoking this script.
"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("download")

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "config.yaml"

LIST_COLUMNS = ["image", "PA", "AP", "text"]


def load_config() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inspect_dataset(root: Path) -> None:
    """Print per-extension file counts under the dataset root."""
    log.info("Inspecting %s", root)
    by_ext: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file():
            by_ext.setdefault(p.suffix.lower(), []).append(p)
    for ext, files in sorted(by_ext.items(), key=lambda kv: -len(kv[1])):
        log.info("  %-8s %5d files", ext or "<none>", len(files))
        for sample in files[:3]:
            log.info("           e.g. %s", sample.relative_to(root))


def parse_list_cell(cell: Any) -> list | None:
    """Parse a stringified Python list literal safely (parse-only, no execution).

    Uses ast.parse() and inspects the resulting tree; rejects anything that isn't
    a plain list of constants. Equivalent in scope to ast.literal_eval but routed
    through the lower-level building block.
    """
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str):
        return None
    try:
        tree = ast.parse(cell.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    list_node = tree.body[0].value
    if not isinstance(list_node, ast.List):
        return None
    out: list = []
    for elt in list_node.elts:
        if isinstance(elt, ast.Constant):
            out.append(elt.value)
        else:
            # Reject anything fancy (calls, names, expressions).
            return None
    return out


def filter_reports(text_list: list) -> list[str]:
    """Keep entries that look like a real report.

    Non-empty == stripped, AND ('Findings:' in t OR len > 50). Filters out
    shells like 'Findings:  Impression: ' that have no actual content.
    """
    out: list[str] = []
    for t in text_list:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        if "Findings:" in s or len(s) > 50:
            out.append(s)
    return out


def pick_view(pa: list, ap: list, image_list: list) -> tuple[str, str] | None:
    """Return (image_relpath, view_label). Priority PA -> AP -> OTHER. None if all empty."""
    for lst, label in [(pa, "PA"), (ap, "AP"), (image_list, "OTHER")]:
        if not lst:
            continue
        first = lst[0]
        if isinstance(first, str) and first.strip():
            return first.strip(), label
    return None


def find_csvs(root: Path) -> tuple[Path | None, Path | None]:
    """Locate mimic_cxr_aug_train.csv and mimic_cxr_aug_validate.csv under root."""
    train: Path | None = None
    val: Path | None = None
    for c in root.rglob("*.csv"):
        n = c.name.lower()
        if train is None and "train" in n:
            train = c
        elif val is None and ("val" in n or "valid" in n):
            val = c
    return train, val


def find_images_root(kagglehub_root: Path) -> Path:
    """Find the directory whose 'files/p1*' subtree exists.

    Depth-limited: checks root and immediate subdirs only - no recursion.
    On Colab the dataset is mounted via FUSE at /kaggle/input/... and a
    recursive rglob over 260K JPGs hangs; the prefix we want lives at most
    one directory deep, so a shallow probe is sufficient.
    """
    candidates = [kagglehub_root] + [p for p in kagglehub_root.iterdir() if p.is_dir()]
    for base in candidates:
        files_dir = base / "files"
        if files_dir.is_dir():
            # Cheap probe: does it have any p1*/p2* patient prefix dir?
            try:
                next(files_dir.glob("p1*"))
                return base
            except StopIteration:
                continue
    raise FileNotFoundError(
        f"Could not locate 'files/pXX/' under {kagglehub_root} (checked root + 1 level deep)."
    )


def build_eligible(df_all: pd.DataFrame, images_root: Path) -> tuple[list[dict], dict]:
    """Parse + filter + view-pick + path-resolve. Returns (eligible_rows, funnel)."""
    funnel = {
        "total": len(df_all),
        "parseable": 0,
        "exactly_one_report": 0,
        "has_pa_or_ap_image": 0,
        "has_any_image": 0,
        "image_file_resolves": 0,
    }
    eligible: list[dict] = []

    for _, row in df_all.iterrows():
        parsed = {col: parse_list_cell(row.get(col)) for col in LIST_COLUMNS}
        if any(parsed[c] is None for c in LIST_COLUMNS):
            continue
        funnel["parseable"] += 1

        reports = filter_reports(parsed["text"])
        if len(reports) != 1:
            continue
        funnel["exactly_one_report"] += 1
        report = reports[0]

        pa_ok = bool(parsed["PA"]) and isinstance(parsed["PA"][0], str) and parsed["PA"][0].strip()
        ap_ok = bool(parsed["AP"]) and isinstance(parsed["AP"][0], str) and parsed["AP"][0].strip()
        if pa_ok or ap_ok:
            funnel["has_pa_or_ap_image"] += 1

        picked = pick_view(parsed["PA"], parsed["AP"], parsed["image"])
        if picked is None:
            continue
        funnel["has_any_image"] += 1
        img_relpath, view_used = picked

        abs_path = images_root / img_relpath
        if not abs_path.is_file():
            continue
        funnel["image_file_resolves"] += 1

        eligible.append({
            "subject_id": str(row.get("subject_id")),
            "image_abspath": abs_path,
            "view_used": view_used,
            "report": report,
        })

    return eligible, funnel


def print_funnel(funnel: dict, images_root: Path) -> None:
    log.info("Patient funnel:")
    log.info("  [1] total rows                    : %d", funnel["total"])
    log.info("  [2] with all lists parseable      : %d", funnel["parseable"])
    log.info("  [3] exactly-one non-empty report  : %d", funnel["exactly_one_report"])
    log.info("  [4] has >=1 PA or AP image        : %d", funnel["has_pa_or_ap_image"])
    log.info("  [4.5] images_root detected as     : %s", images_root)
    log.info("  [.] has any usable image (any view): %d", funnel["has_any_image"])
    log.info("  [.] image file resolves on disk   : %d  (== eligible pool)", funnel["image_file_resolves"])


def image_is_loadable(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="download, then print per-extension file counts and exit")
    ap.add_argument("--inspect-csv", action="store_true", dest="inspect_csv",
                    help="run parsing + filtering + view-pick, print funnel, and exit (no copying)")
    ap.add_argument("--n", type=int, default=None,
                    help="override total sample size (split rescaled proportionally)")
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
        log.error("kagglehub not installed - run `pip install -r requirements.txt`")
        sys.exit(1)

    log.info("kagglehub: downloading %s", dc["kaggle_slug"])
    dataset_path = Path(kagglehub.dataset_download(dc["kaggle_slug"]))
    log.info("kagglehub: dataset_path = %s", dataset_path)

    if args.inspect:
        inspect_dataset(dataset_path)
        return

    train_csv, val_csv = find_csvs(dataset_path)
    if train_csv is None:
        log.error("Could not locate train CSV (expected mimic_cxr_aug_train.csv).")
        log.error("Run with --inspect to see the dataset layout.")
        sys.exit(1)

    log.info("train CSV: %s", train_csv.relative_to(dataset_path))
    if val_csv is not None:
        log.info("val   CSV: %s", val_csv.relative_to(dataset_path))
    else:
        log.warning("No validate CSV found - proceeding with train only.")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv) if val_csv is not None else pd.DataFrame()
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    log.info("Combined rows: %d (train=%d, val=%d)",
             len(df_all), len(df_train), len(df_val))

    try:
        images_root = find_images_root(dataset_path)
    except FileNotFoundError as e:
        log.error("%s", e)
        log.error("Run with --inspect to see the dataset layout.")
        sys.exit(1)
    log.info("images_root: %s", images_root)

    eligible, funnel = build_eligible(df_all, images_root)
    print_funnel(funnel, images_root)

    if args.inspect_csv:
        return

    if not eligible:
        log.error("Zero eligible rows - rerun with --inspect-csv to debug.")
        sys.exit(1)

    # Deterministic sample of `total_n` patients.
    df_pool = pd.DataFrame(eligible)
    if total_n > len(df_pool):
        log.warning("Requested %d but only %d eligible - using all available.",
                    total_n, len(df_pool))
        total_n = len(df_pool)
        index_n = max(1, int(round(total_n * 0.75)))
        test_n = total_n - index_n
    df_sample = df_pool.sample(n=total_n, random_state=seed).reset_index(drop=True)

    # Copy + validate.
    rows: list[dict] = []
    skipped_bad = 0
    seen_ids: set[str] = set()
    for entry in tqdm(df_sample.to_dict(orient="records"), desc="copying images"):
        src: Path = entry["image_abspath"]
        if not image_is_loadable(src):
            skipped_bad += 1
            continue
        # The image stem is typically a DICOM UUID and unique in practice.
        # Disambiguate the rare collision by prefixing the subject_id.
        sample_id = src.stem
        if sample_id in seen_ids:
            sample_id = f"{entry['subject_id']}_{src.stem}"
        seen_ids.add(sample_id)

        dst = images_dir / f"{sample_id}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)
        rows.append({
            "id": sample_id,
            "image_path": dst.relative_to(REPO_ROOT).as_posix(),
            "report": entry["report"],
            "subject_id": entry["subject_id"],
            "view_used": entry["view_used"],
        })
    log.info("Copied %d images; skipped %d unreadable", len(rows), skipped_bad)
    if not rows:
        log.error("All sampled images failed validation.")
        sys.exit(1)

    # Shuffle once with the same seed, then split.
    df = pd.DataFrame(rows, columns=["id", "image_path", "report", "subject_id", "view_used"])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
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

    log.info("view_used distribution (manifest_all):")
    for v, c in df["view_used"].value_counts().items():
        log.info("  %-6s %d", v, c)


if __name__ == "__main__":
    main()
