"""Gradio demo for the image-to-manga converter."""

import os
import threading
from pathlib import Path
from typing import List, Optional

import gradio as gr

import convert_manga
import generate_screentone

DEFAULT_SCREENTONE_SELECTION = [
    "dot",
    "grid",
    "sand-white",
    "line45",
    "directional-noise",
]

_SCREENTONES_READY = False
_SCREENTONES_LOCK = threading.Lock()


def _screentones_exist() -> bool:
    tone_dir = Path(convert_manga.SCREENTONE_DIR)
    if not tone_dir.exists():
        return False

    for tone_type in convert_manga.SCREENTONE_TYPES:
        if list(tone_dir.glob(f"{tone_type}_*.png")):
            continue
        return False
    return True


def ensure_screentones() -> Optional[str]:
    """Generate screentones once if they are missing."""

    global _SCREENTONES_READY
    with _SCREENTONES_LOCK:
        if _SCREENTONES_READY:
            return None

        if _screentones_exist():
            _SCREENTONES_READY = True
            return None

        os.makedirs(convert_manga.SCREENTONE_DIR, exist_ok=True)
        generate_screentone.generate_and_save_all(convert_manga.SCREENTONE_DIR)
        _SCREENTONES_READY = True
        return "Generated screentones."


def run_pipeline(
    image_path: Optional[str],
    screentone_types: List[str],
    histogram_equalization: bool,
    sobel_thresh: float,
):
    ensure_msg = ensure_screentones()
    if ensure_msg:
        gr.Info(ensure_msg)

    if not image_path:
        raise gr.Error("Please upload an image to start the conversion.")

    if not screentone_types:
        raise gr.Error("Please select at least one screentone type.")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "gradio_result.png"

    try:
        result = convert_manga.convert_to_manga(
            input_path=image_path,
            output_path=str(output_path),
            screentone_types=screentone_types,
            histogram_equalization=histogram_equalization,
            sobel_thresh=sobel_thresh,
        )
    except Exception as exc:  # pragma: no cover - surfaced to UI
        raise gr.Error(str(exc))

    gr.Info(f"Saved latest output to {output_path}.")
    return result


def build_interface():
    with gr.Blocks(title="Image to Manga Converter") as demo:
        gr.Markdown(
            """
            ## Manga-Style Conversion of Natural Images via Adaptive Screentones
            Upload a photo to convert it into manga style.

            Note: At the first run, screentone assets will be generated automatically, which may take a bit longer.
            """
        )

        with gr.Row():
            input_image = gr.Image(
                label="Upload image",
                type="filepath",
                height=400,
            )
            output_image = gr.Image(
                label="Manga output",
                type="numpy",
                height=400,
            )

        with gr.Row():
            with gr.Column(scale=2):
                screentone_selector = gr.CheckboxGroup(
                    label="Screentone types",
                    choices=convert_manga.SCREENTONE_TYPES,
                    value=DEFAULT_SCREENTONE_SELECTION,
                )
            with gr.Column(scale=1):
                hist_eq_checkbox = gr.Checkbox(
                    label="Histogram equalization",
                    value=False,
                )
                sobel_slider = gr.Slider(
                    label="Sobel threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.2,
                )

        run_button = gr.Button("Convert Image")

        example_files = sorted(Path("pictures").glob("input_*"))
        if example_files:
            gr.Examples(
                examples=[[str(example)] for example in example_files],
                inputs=[input_image],
                label="Examples",
            )

        run_button.click(
            fn=run_pipeline,
            inputs=[input_image, screentone_selector, hist_eq_checkbox, sobel_slider],
            outputs=[output_image],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch()
