# Chest X-Ray Multi-Modal Intelligence System

**DSAI 413 — Assignment 2**

---

## 1. Problem

Two operating modes over the MIMIC-CXR dataset:

1. **Report generation** — given a chest X-ray image, produce a radiology report (Findings + Impression).
2. **Question answering (RAG)** — given a natural-language clinical question, retrieve relevant indexed reports and generate a grounded answer.

The work compares one **mandatory model** against one **lighter baseline** in each mode, all zero-shot, on a 400-sample MIMIC-CXR subset (300 indexed, 100 held-out test).

| Mode | Mandatory | Baseline |
|---|---|---|
| Report generation | MedGemma 1.5-4B-IT (multimodal LM) | OpenCLIP ViT-B/32 nearest-neighbor retrieval |
| QA retrieval | ColPali v1.3 (vision-LM late interaction) | sentence-transformers MiniLM-L6 (text bi-encoder) |
| QA answer | MedGemma 1.5-4B-IT (text-only with retrieved context) | — |

---

## 2. Architecture

```mermaid
flowchart TB
    subgraph Data
        K[Kaggle: MIMIC-CXR<br/>simhadrisadaram] --> D[data/download.py]
        D --> M[manifest.csv<br/>id, image_path, report,<br/>subject_id, view_used]
        M --> MI[manifest_index.csv<br/>300 train]
        M --> MT[manifest_test.csv<br/>100 test]
    end

    subgraph "Report Mode"
        IMG[Chest X-ray]
        IMG --> MG[MedGemma 4-bit]
        MG --> R1[Report]
        IMG --> CLIP[OpenCLIP ViT-B/32]
        CLIP --> FAISS_IMG[FAISS IndexFlatIP<br/>over 300 imgs]
        FAISS_IMG --> R2[Nearest report]
    end

    subgraph "QA RAG Mode"
        Q[Question] --> CP[ColPali v1.3<br/>multi-vector]
        MI -.render report PNG.-> CPI[ColPali index<br/>doc_embeddings.pt]
        CP --> CPI
        CPI --> TOPK1[top-3 reports]
        Q --> ML[MiniLM-L6]
        MI -.encode text.-> MLI[FAISS IndexFlatIP<br/>text_index.faiss]
        ML --> MLI
        MLI --> TOPK2[top-3 reports]
        TOPK1 --> MG2[MedGemma context+Q]
        TOPK2 --> MG2
        MG2 --> ANS[Answer]
    end

    subgraph Evaluation
        R1 & R2 --> METR1[ROUGE-L + BERTScore]
        ANS --> METR2[Recall@3 + LLM-judge]
    end
```

The same `MedGemma` singleton serves three jobs (report generation, RAG answering, LLM-as-judge in eval), avoiding repeated 4-bit loads.

---

## 3. Model choices

### MedGemma 1.5-4B-IT (`google/medgemma-1.5-4b-it`)
- Mandatory per the assignment.
- Loaded in **4-bit nf4** with `bitsandbytes` so it fits alongside ColPali on a Colab T4 (15 GB).
- Multimodal: handles image+text for report generation; text-only for QA answering and judging.
- Generation: greedy decoding (`do_sample=False`) for reproducibility.

### ColPali v1.3 (`vidore/colpali-v1.3-merged`)
- Mandatory per the assignment. We use the `-merged` variant (LoRA pre-merged into the base PaliGemma weights) to avoid PEFT adapter-key remap incompatibilities with current transformers.
- 4-bit nf4 quantized.
- Reports are rendered as single-page PNGs (PIL + DejaVu Sans 20px) at 1024×1280, then embedded as multi-vector representations. Queries are scored via late-interaction MaxSim through `processor.score_multi_vector`.

### Baselines
- **BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) for image-to-report retrieval. ViT-B/16 + PubMedBERT, pretrained by Microsoft on 15M PubMed (image, caption) pairs. L2-normalized 512-d embeddings, FAISS `IndexFlatIP` over the 300 train images. (Initial implementation used vanilla OpenCLIP ViT-B/32 with `openai` weights; this was qualitatively unable to surface medical features like sternotomy hardware or cardiomegaly because its general-internet pretraining gives every chest X-ray ~0.97 similarity. BiomedCLIP was substituted as a deliberate methodology choice while staying zero-shot — the medical fine-tune was done by Microsoft, not by us.)
- **sentence-transformers/all-MiniLM-L6-v2** for text-to-report retrieval. L2-normalized 384-d embeddings, FAISS `IndexFlatIP` over the 300 train reports.

---

## 4. Dataset and QA generation

### MIMIC-CXR subset
- **Source:** `simhadrisadaram/mimic-cxr-dataset` via `kagglehub`. Two CSVs (`mimic_cxr_aug_train.csv`, `mimic_cxr_aug_validate.csv`) with ~65k patient-rows.
- **Each row = one patient**, with stringified Python lists in `image`, `view`, `AP`, `PA`, `Lateral`, `text`. Parsed via `ast.parse` (safe literal eval) into Python lists.
- **Filtering funnel** (logged by `data/download.py --inspect-csv`):

  | Stage | Count |
  |---|---:|
  | Total rows | 65,086 |
  | All lists parseable | 65,086 |
  | Exactly-one non-empty report (`'Findings:' in t OR len > 50`) | 32,680 |
  | Has ≥ 1 PA or AP image | 31,563 |
  | Image file resolves on disk | **22,783 (eligible pool)** |

- **Sampling:** 400 patients drawn with `seed=42`, view priority `PA > AP > other`. Split into 300 index / 100 test (re-shuffled with same seed).
- **Manifest schema:** `id, image_path, report, subject_id, view_used`.

### QA dataset
- For 25 reports sampled from `manifest_index.csv`, MedGemma is prompted (text-only) to emit a strict JSON list of 3 QA pairs per report — one yes/no, one open-ended, one location/laterality.
- The parser is defensive: tries direct `json.loads`, strips ` ```json ` fences, falls back to `json.JSONDecoder().raw_decode` scanning every `[` (survives MedGemma 1.5's `<unused94>thought` preambles).
- Output: `data/sample/qa_dataset.json` with `{report_id, image_path, report, qa_pairs:[{q, a}, …]}`.
- 25 reports × 3 pairs = **75 QA items**, comfortably covering the 50-item eval budget after parse-failure attrition.

---

## 5. Evaluation methodology

### Report mode (n=30 test images)
- **ROUGE-L F1** (per-pair, then mean) via `rouge_score`.
- **BERTScore F1** (vectorized over all pairs) via `bert_score`, `lang="en"` (RoBERTa-large encoder).

### QA RAG mode (n=15 questions, drawn from `qa_dataset.json`)
- **Recall@3**: fraction of questions where the gold patient's report appears among the top-3 retrieved IDs.
- **LLM-as-judge**: for each (question, predicted answer, gold answer) triple, MedGemma is asked to emit `VERDICT: <1|2|3>` (1=correct / 2=partial / 3=wrong). Parser uses the explicit `VERDICT:` regex with a last-word fallback to handle thinking-mode preambles. **Judge accuracy** = correct / n.

### Honest limitations of the protocol
- **Same model is both system-under-test AND judge.** Standard LLM-as-judge weakness; the judge may favor outputs in MedGemma's style. The Recall@k metric is independent of this.
- **Generic clinical questions** (e.g. "Is there a pleural effusion?") have no patient-specific anchor, so retrieval Recall@k is expected to be low across both retrievers (~0). This biases the comparison toward the answering step rather than retrieval quality per se.

---

## 6. Results

_Numbers below come from `results/comparison.json` (populated by `python -m src.eval`)._

### Report Generation (n=30)

| Model | ROUGE-L F1 | BERTScore F1 | Latency mean (s) |
|---|---:|---:|---:|
| MedGemma 1.5-4B (zero-shot, sampling T=0.3) | 0.253 | 0.859 | 21.55 |
| BiomedCLIP retrieval (top-1) | **0.276** | **0.881** | **0.36** |

**BiomedCLIP edges MedGemma on both quality metrics**: +9% relative ROUGE-L (0.276 vs 0.253) and +0.022 BERTScore, with **60× lower latency**. This is the opposite of what one might naively expect from a 4B-parameter multimodal LM beating a 90M-parameter image-text encoder — and it tells an important story about model selection under data and compute constraints.

For context, the initial baseline (vanilla OpenCLIP ViT-B/32 with `openai` weights) was at ROUGE-L 0.283, and MedGemma beat *it* by +23%. Swapping the baseline to BiomedCLIP — same architecture family, same zero-shot constraint, but 15M PubMed image-text pairs of medical pretraining done by Microsoft instead of general-internet pretraining — closed the gap and inverted the result.

**Why BiomedCLIP wins at this scale:**

1. **Medical pretraining is the dominant lever.** ROUGE-L and BERTScore both reward outputs that match the gold's clinical phrasing. BiomedCLIP retrieves verbatim reports written by radiologists, so when its top match is clinically similar, the metric overlap is high. MedGemma generates novel text that's clinically sensible but uses different word choices than the gold.
2. **Generation introduces format drift.** Our prompt asks MedGemma for per-region commentary (lungs L/R, pleura, heart, mediastinum, bones, devices). This produces longer, more enumerated outputs (~21s latency = ~2× the original concise-prompt run). Gold MIMIC reports are terse paragraphs, so a longer output dilutes n-gram overlap even when content is correct.
3. **300 indexed reports is enough for retrieval, too small for fine-tuning.** Most test images have a clinically similar training neighbor, so retrieval has high ceiling. Fine-tuning either model at this scale would overfit.

**Where MedGemma still wins:** clinical findings that don't have a close neighbor in the 300-image index. The pre-eval qualitative test on a sternotomy + cardiomegaly + edema X-ray: MedGemma identified all three findings explicitly; even BiomedCLIP retrieval found a *similar* patient but not the same specific combination of devices. For open-ended report generation on rare findings, MedGemma's generation is more flexible than retrieval's nearest-neighbor.

### QA RAG (n=15)

| Retriever | Recall@3 | Judge accuracy | correct | partial | wrong | unparseable+err | latency mean (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ColPali v1.3 | 0.000 | 0.400 | 6 | **3** | 5 | 1 | 70.17 |
| MiniLM-L6 text | **0.133** | **0.467** | **7** | 1 | 7 | 0 | **11.01** |

**MiniLM-L6 narrowly beats ColPali v1.3** on Recall@3 (0.133 vs 0.000), strict-correct judge accuracy (46.7% vs 40.0%), and end-to-end latency (6.4× faster). ColPali pulls ahead on lenient "correct + partial" (60% vs 53%) by producing more hedged answers. At n=15 the gap is within sampling noise, but the direction is consistent across three independent measures and the latency penalty for ColPali is real.

### Qualitative observations

**Side-by-side on a real test image** (median sternotomy + mitral valve replacement + moderate cardiomegaly + mild pulmonary edema):

| Component | Output |
|---|---|
| Gold report | "Patient is status post median sternotomy and mitral valve replacement. Moderate cardiomegaly... Mild pulmonary edema is present..." |
| **Vanilla OpenCLIP top-3** | All three matches at sim ~0.97, all describing *normal* lungs / *no acute cardiopulmonary process* / *cardiac silhouette normal*. None mention sternotomy, cardiomegaly, or edema. |
| **BiomedCLIP top-1** | "moderate cardiomegaly... small bilateral pleural effusions... most suggestive of moderate pulmonary edema." Hits the two main findings (cardiomegaly + edema). Similarity 0.916, well-separated from rank-2/3 (0.904 / 0.902). |
| **MedGemma** | Identifies sternal wires, mild cardiomegaly, and pulmonary edema with anatomically specific language. |

This is the clearest demonstration in the project of *why medical pretraining matters*: vanilla OpenCLIP has no concept of sternotomy hardware or cardiomegaly because those features are below its general-internet pretraining noise floor. BiomedCLIP's 15M PubMed pairs include thousands of similar pathology images, so the features are within distribution.

**Other observations:**

- **MedGemma's reports** follow the requested `FINDINGS: … IMPRESSION: …` structure consistently. With `temperature=0.3` sampling enabled, outputs vary between images; under greedy decoding (used in the very first eval pass) they collapsed to the same "No acute cardiopulmonary process" boilerplate regardless of input.
- **The per-region prompt is a double-edged sword.** Forcing MedGemma to comment on lungs/pleura/heart/mediastinum/bones/devices catches more findings on abnormal X-rays but also lengthens output by ~2×, hurting ROUGE-L when the gold is short. The original concise prompt scored higher on metrics but produced less informative outputs.
- **ColPali's rendered-page strategy is the wrong tool for this corpus.** ColPali was trained on real document scans with rich layout (PDFs, tables, charts). Our reports are short clinical paragraphs rendered as plain DejaVu Sans on a white canvas — there's no layout signal to exploit. The model's multi-vector representation also costs ~6× the latency of MiniLM with no observable retrieval benefit here.
- **MiniLM beats ColPali on retrieval and downstream answer accuracy** precisely because clinical text-to-text similarity is what the QA task is actually about, and that's what MiniLM was trained for.
- **Recall@3 is structurally low (≤ 13%)** for both retrievers because the auto-generated QA questions are generic ("Is there pneumonia?", "Is there a pleural effusion?") with no patient-specific anchor. The retrievers do their job correctly given the inputs; the limitation is in the question generation, not the retrievers. Despite low retrieval recall, judge accuracy lands at 40–47% because generic clinical questions often have answers consistent across many reports in the corpus.

---

## 7. Limitations

1. **Subset, not full MIMIC-CXR.** 400 pairs is statistically small; results indicate trend, not absolute capability. The full dataset has ~250k images.
2. **Zero-shot only.** No domain fine-tuning of MedGemma or ColPali on MIMIC-CXR; an in-domain finetune would likely materially improve both modes.
3. **LLM-as-judge with same model.** MedGemma judging its own outputs is a known bias source; cross-model judges (e.g. GPT-4, Claude) would be more rigorous if API access were in scope.
4. **Rendered-report ColPali pages are synthetic**, not native scanned documents. ColPali was trained on real document pages; basic PIL typography on a blank canvas may not exercise its layout-understanding strength.
5. **Generic clinical questions.** Auto-generated QA pairs are not patient-specific (no anchor to identify which patient's report is gold), so retrieval Recall@k is structurally low regardless of retriever quality.
6. **MedGemma 1.5 thinking-mode preambles.** The `<unused94>thought` token leaks structured reasoning before JSON or verdict outputs. Mitigated by defensive parsers and bumped `max_new_tokens`, but ~5–15% of judge outputs still need fallback parsing.
7. **Single Colab T4 footprint.** MedGemma 4-bit + ColPali 4-bit fits tightly in 15 GB VRAM; bf16 ColPali would not coexist. On larger GPUs (A100), 8-bit or bf16 ColPali would give cleaner numbers.
8. **Automated metric vs. clinical informativeness trade-off.** Forcing MedGemma into per-region commentary helps the demo produce informative reports but hurts ROUGE-L because gold MIMIC reports are terse. A shorter prompt scores higher on metrics but produces "no acute cardiopulmonary process" boilerplate at greedy decoding. The eval numbers reported here use the informative prompt + `T=0.3` sampling — i.e. they are pessimistic relative to a metric-optimized prompt would yield, but optimistic for clinical utility.
9. **BiomedCLIP beats MedGemma on metrics but only on this corpus structure.** The 300-image retrieval pool happens to include enough clinically diverse reports that nearest-neighbor finds a plausible match for most test images. On rarer pathology with no close neighbor in the index, retrieval would fall apart and MedGemma's flexibility wins. A larger and more pathology-diverse index would shift the balance further toward retrieval; a smaller/rarer test distribution would shift it back toward generation.

---

## 8. Future work

- **Patient-grounded QA generation:** include patient-specific markers (view, modality, any imaging finding token) in generated questions so retrieval has something to anchor on.
- **Cross-model judges:** add a second judge (e.g. Claude or GPT-4) to debias the LLM-judge accuracy.
- **In-domain fine-tune:** LoRA-tune MedGemma on a subset of MIMIC-CXR (image + report pairs) and re-evaluate; this is the lever most likely to move ROUGE-L meaningfully.
- **Stronger baseline:** swap MiniLM for a clinical-text encoder (e.g. `emilyalsentzer/Bio_ClinicalBERT` mean-pooled) — better in-domain semantics for medical text.
- **Native PDF rendering for ColPali:** convert reports to actual PDF (e.g. via `reportlab`) so ColPali sees document-like layouts rather than plain text on white.
- **Confidence calibration:** have MedGemma emit uncertainty alongside the answer; couple with retrieval similarity to abstain when neither retrieval nor confidence support an answer.

---

## Appendix: Repo layout

```
chest-xray-system/
├── README.md, config.yaml, requirements.txt, .env.example
├── data/
│   ├── download.py            # Kaggle pull + filter + 400-sample split
│   ├── build_qa_dataset.py    # MedGemma generates 3 QA pairs per report
│   └── sample/                # gitignored at-runtime data + indexes
├── src/
│   ├── models/
│   │   ├── medgemma.py           # 4-bit MedGemma singleton
│   │   ├── colpali_retriever.py  # ColPali index + multi-vector scoring
│   │   ├── clip_retriever.py     # OpenCLIP image retrieval
│   │   └── text_retriever.py     # MiniLM text retrieval
│   ├── report_mode.py            # generate_report_medgemma / _clip
│   ├── qa_mode.py                # qa_colpali_rag / qa_text_rag
│   └── eval.py                   # ROUGE + BERTScore + judge
├── app/app.py                    # Gradio demo (Report / QA tabs)
├── notebooks/run_on_colab.ipynb  # Colab launch flow
├── results/                      # comparison.json + comparison.md
└── report/REPORT.md              # this document
```
