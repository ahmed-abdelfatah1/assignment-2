"""Phase 3: Generate clinical QA pairs from reports via MedGemma.

For each sampled report from manifest_index.csv, prompt MedGemma to emit
exactly 3 QA pairs (yes/no, open-ended, location/laterality) as strict JSON,
then write to data/sample/qa_dataset.json with shape:

    [
      {
        "report_id": str,
        "image_path": str,
        "report": str,
        "qa_pairs": [{"q": str, "a": str}, ...]
      },
      ...
    ]

Usage:
    python data/build_qa_dataset.py              # full run (config.qa_generation.reports_to_use)
    python data/build_qa_dataset.py --limit 5    # quick sanity-check pass
    python data/build_qa_dataset.py --out path   # override output path
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "config.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("build_qa")


QA_PROMPT_TEMPLATE = (
    "From the radiology report below, generate exactly 3 clinical question-answer pairs. "
    "Mix question types: one yes/no (finding presence), one open-ended (description), "
    "one location/laterality. "
    # Double braces -> literal '{' '}' after .format(), so MedGemma sees real JSON.
    'Output strict JSON: [{{"q": ..., "a": ...}}, ...]. '
    "Report: {report}"
)


def _load_config() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|```\s*$", re.MULTILINE)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json_array(text: str) -> list | None:
    """Pull a JSON list out of an LLM response, handling fences and prose."""
    cleaned = _FENCE_RE.sub("", text).strip()

    # Try direct parse first.
    try:
        v = json.loads(cleaned)
        if isinstance(v, list):
            return v
    except json.JSONDecodeError:
        pass

    # Greedy bracket match - covers cases like "Here's the JSON: [...]"
    m = _ARRAY_RE.search(cleaned)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass

    return None


def _validate_qa_pairs(raw: object) -> list[dict] | None:
    """Ensure we got a list of {q, a} dicts with non-empty strings.

    Tolerates 'question'/'answer' aliases as a fallback. Returns None on any
    structural failure so the caller can skip the row.
    """
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        q = item.get("q") or item.get("question")
        a = item.get("a") or item.get("answer")
        if not isinstance(q, str) or not isinstance(a, str):
            return None
        q, a = q.strip(), a.strip()
        if not q or not a:
            return None
        out.append({"q": q, "a": a})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on reports (default: cfg.qa_generation.reports_to_use)")
    ap.add_argument("--out", type=Path, default=None,
                    help="override output path (default: cfg.data.qa_dataset)")
    args = ap.parse_args()

    cfg = _load_config()
    qg = cfg["qa_generation"]
    n_reports = args.limit or qg["reports_to_use"]
    out_path = args.out or (REPO_ROOT / cfg["data"]["qa_dataset"])
    manifest = REPO_ROOT / cfg["data"]["manifest_index"]
    seed = cfg["data"]["seed"]

    log.info("manifest : %s", manifest)
    log.info("output   : %s", out_path)
    df = pd.read_csv(manifest)
    log.info("manifest rows: %d, sampling: %d, seed: %d", len(df), n_reports, seed)
    if n_reports > len(df):
        log.warning("Requested %d but only %d rows in manifest_index; using all.",
                    n_reports, len(df))
        n_reports = len(df)
    df_sample = df.sample(n=n_reports, random_state=seed).reset_index(drop=True)

    from src.models import medgemma
    log.info("Loading MedGemma (first call only)...")
    medgemma._ensure_loaded()
    log.info("MedGemma ready.")

    results: list[dict] = []
    fail_unparseable = 0
    fail_generate = 0

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="generating QA"):
        prompt = QA_PROMPT_TEMPLATE.format(report=row["report"])
        try:
            response = medgemma.generate(None, prompt, max_new_tokens=512)
        except Exception as e:
            log.warning("MedGemma error on %s: %s", row["id"], e)
            fail_generate += 1
            continue

        raw_parsed = _extract_json_array(response)
        validated = _validate_qa_pairs(raw_parsed) if raw_parsed is not None else None
        if validated is None:
            log.warning("Unparseable QA for %s; raw[:200]=%r", row["id"], response[:200])
            fail_unparseable += 1
            continue

        results.append({
            "report_id": str(row["id"]),
            "image_path": str(row["image_path"]),
            "report": str(row["report"]),
            "qa_pairs": validated,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info("Summary:")
    log.info("  rows requested      : %d", len(df_sample))
    log.info("  rows written        : %d", len(results))
    log.info("  generation failures : %d", fail_generate)
    log.info("  parse failures      : %d", fail_unparseable)
    log.info("  output              : %s", out_path)

    # QA-type distribution audit (best-effort: lowercase-substring heuristics).
    if results:
        bucket = {"yes_no": 0, "location": 0, "open_ended": 0}
        for entry in results:
            for qa in entry["qa_pairs"]:
                q = qa["q"].lower()
                if any(q.startswith(s) for s in ("is ", "are ", "does ", "do ", "has ", "have ")):
                    bucket["yes_no"] += 1
                elif any(k in q for k in ("where", "which side", "right", "left", "laterality", "location")):
                    bucket["location"] += 1
                else:
                    bucket["open_ended"] += 1
        log.info("  qa-type rough split : %s", bucket)


if __name__ == "__main__":
    main()
