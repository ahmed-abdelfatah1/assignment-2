"""Phase 6: Gradio demo with two tabs.

Tabs:
  - Report Generation: upload chest X-ray -> generated report (MedGemma or CLIP).
  - QA: type a question -> answer + retrieved report IDs (ColPali or text-RAG).

Launches with share=True (per config.demo.share) so the Colab share link is
publicly reachable.

Run:
    python app/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import yaml
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_CFG_PATH = _REPO_ROOT / "config.yaml"


def _load_config() -> dict:
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Lazy imports so opening the app doesn't load every model at import time.
def _report_medgemma(img):
    from src.report_mode import generate_report_medgemma
    return generate_report_medgemma(img)


def _report_clip(img):
    from src.report_mode import generate_report_clip
    # Demo shows top-3 with similarity scores so the user sees CLIP's alternatives
    # instead of always landing on the same modal "normal" report. Eval still
    # uses top_k=1 to keep the head-to-head with MedGemma fair.
    return generate_report_clip(img, top_k=3)


def _qa_colpali(q, k):
    from src.qa_mode import qa_colpali_rag
    return qa_colpali_rag(q, top_k=k)


def _qa_text(q, k):
    from src.qa_mode import qa_text_rag
    return qa_text_rag(q, top_k=k)


REPORT_MODELS = {
    "MedGemma 1.5-4B (multimodal)": _report_medgemma,
    "BiomedCLIP ViT-B/16 (retrieval, top-3)": _report_clip,
}

QA_RETRIEVERS = {
    "ColPali v1.3 (vision-LM)": _qa_colpali,
    "MiniLM-L6 (text)": _qa_text,
}


def run_report(image, model_name: str):
    """Handler for the Report tab."""
    if image is None:
        return "Please upload an image.", ""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    fn = REPORT_MODELS[model_name]
    try:
        result = fn(image.convert("RGB"))
    except Exception as e:
        return f"Error: {e}", ""
    return result["report"], f"{result['latency_s']:.2f}s"


def run_qa(question: str, retriever_name: str, top_k: int):
    """Handler for the QA tab."""
    q = (question or "").strip()
    if not q:
        return "Please enter a question.", "", ""
    fn = QA_RETRIEVERS[retriever_name]
    try:
        result = fn(q, top_k)
    except Exception as e:
        return f"Error: {e}", "", ""
    retrieved = result["retrieved"]
    retrieved_lines = "\n".join(
        f"- id={r['id']}  score={r['score']:.3f}" for r in retrieved
    )
    return result["answer"], retrieved_lines, f"{result['latency_s']:.2f}s"


def _build_ui() -> gr.Blocks:
    cfg = _load_config()
    default_k = cfg["retrieval"]["top_k"]

    with gr.Blocks(title="Chest X-Ray Multi-Modal System") as demo:
        gr.Markdown(
            "# Chest X-Ray Multi-Modal Intelligence System\n"
            "DSAI 413 - Assignment 2. Image-to-report generation and RAG QA over a "
            "300-report MIMIC-CXR subset. Models: MedGemma 1.5-4B, ColPali v1.3."
        )

        with gr.Tabs():
            with gr.Tab("Report Generation"):
                gr.Markdown(
                    "Upload a chest X-ray and pick a model. **MedGemma** generates "
                    "a per-region radiology report from scratch (~10 s/image). "
                    "**BiomedCLIP retrieval** returns the top-3 most visually similar "
                    "training reports with their similarity scores (~50 ms)."
                )
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(type="pil", label="Chest X-ray", height=400)
                        model_select = gr.Dropdown(
                            list(REPORT_MODELS.keys()),
                            value=list(REPORT_MODELS.keys())[0],
                            label="Model",
                        )
                        report_btn = gr.Button("Generate report", variant="primary")
                    with gr.Column():
                        report_out = gr.Textbox(label="Generated report", lines=14)
                        report_latency = gr.Textbox(label="Latency", interactive=False)
                report_btn.click(
                    run_report,
                    inputs=[img_input, model_select],
                    outputs=[report_out, report_latency],
                )

            with gr.Tab("QA over reports"):
                gr.Markdown(
                    "Ask a clinical question. The retriever pulls the top-k most "
                    "relevant reports from the indexed corpus; MedGemma answers "
                    "using ONLY those reports as context."
                )
                with gr.Row():
                    with gr.Column():
                        q_input = gr.Textbox(
                            label="Question",
                            placeholder="e.g. Is there a pleural effusion? Describe the cardiac silhouette.",
                            lines=2,
                        )
                        retriever_select = gr.Dropdown(
                            list(QA_RETRIEVERS.keys()),
                            value=list(QA_RETRIEVERS.keys())[0],
                            label="Retriever",
                        )
                        k_slider = gr.Slider(
                            minimum=1, maximum=10, value=default_k, step=1,
                            label="top_k retrieved reports",
                        )
                        qa_btn = gr.Button("Ask", variant="primary")
                    with gr.Column():
                        answer_out = gr.Textbox(label="Answer", lines=8)
                        retrieved_out = gr.Textbox(label="Retrieved report IDs", lines=6)
                        qa_latency = gr.Textbox(label="Latency", interactive=False)
                qa_btn.click(
                    run_qa,
                    inputs=[q_input, retriever_select, k_slider],
                    outputs=[answer_out, retrieved_out, qa_latency],
                )

        gr.Markdown(
            "_First query per pipeline loads its model + index - expect ~60 s for "
            "MedGemma 4-bit and ~15 s for ColPali. Subsequent queries are fast._"
        )

    return demo


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    cfg = _load_config()
    demo = _build_ui()
    demo.queue()  # serialize requests so models aren't re-entered concurrently
    demo.launch(
        share=cfg["demo"].get("share", True),
        server_name=cfg["demo"].get("server_name", "0.0.0.0"),
        server_port=cfg["demo"].get("server_port", 7860),
    )


if __name__ == "__main__":
    main()
