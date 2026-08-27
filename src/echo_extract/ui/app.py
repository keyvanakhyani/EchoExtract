"""Gradio web interface for EchoExtract."""

import logging
from pathlib import Path

import gradio as gr

from echo_extract.core.config import settings
from echo_extract.core.logging_config import setup_logging
from echo_extract.engines.faster_whisper_engine import FasterWhisperEngine
from echo_extract.pipeline import transcribe_media

logger = logging.getLogger(__name__)

# Load the model ONCE when the app starts, not per request.
setup_logging()
logger.info("Loading transcription model (this happens once)...")
engine = FasterWhisperEngine()
logger.info("Model loaded and ready.")


def process(
    media_path: str,
    language: str,
    translate: bool,
    formats: list[str],
) -> tuple[str, str, list[str]]:
    """Transcribe an uploaded file and return text, status, and output files.

    Args:
        media_path: Path to the uploaded video or audio file.
        language: Forced source language, or "auto" to detect.
        translate: Whether to also translate to English.
        formats: Which output formats the user selected.

    Returns:
        A tuple of (transcript text, status message, list of output file paths).
    """
    if not media_path:
        return "", "Please upload a file first.", []

    if not formats:
        return "", "Please select at least one output format.", []

    source = Path(media_path)
    lang = None if language == "auto" else language
    translate_to = "en" if translate else None

    logger.info("Processing %s", source.name)

    result = transcribe_media(
        source_path=source,
        engine=engine,
        output_dir=source.parent,
        formats=formats,
        language=lang,
        translate_to=translate_to,
    )

    # Collect the paths of all files that were written, so Gradio can offer
    # them for download.
    base_name = source.stem
    output_files: list[str] = []
    for fmt in formats:
        path = source.parent / f"{base_name}.{result.language}.{fmt}"
        if path.exists():
            output_files.append(str(path))
        # If translation was produced, include the English files too.
        if translate_to:
            en_path = source.parent / f"{base_name}.en.{fmt}"
            if en_path.exists():
                output_files.append(str(en_path))

    status = (
        f"Done. Detected language: {result.language} "
        f"({len(result.segments)} segments)."
    )
    return result.full_text, status, output_files


def build_interface() -> gr.Blocks:
    """Build and return the Gradio interface."""
    with gr.Blocks(title="EchoExtract") as demo:
        gr.Markdown("# 🎧 EchoExtract\nTranscribe video and audio to text, locally.")

        with gr.Row():
            with gr.Column():
                media_input = gr.File(
                    label="Upload video or audio",
                    type="filepath",
                )
                language_input = gr.Dropdown(
                    choices=["auto", "fa", "en"],
                    value="auto",
                    label="Language",
                )
                formats_input = gr.CheckboxGroup(
                    choices=["txt", "srt", "vtt", "json"],
                    value=["txt", "srt"],
                    label="Output formats",
                )
                translate_input = gr.Checkbox(
                    label="Also translate to English",
                    value=False,
                )
                run_button = gr.Button("Transcribe", variant="primary")

            with gr.Column():
                status_output = gr.Textbox(label="Status", interactive=False)
                text_output = gr.Textbox(label="Transcript", lines=15)
                files_output = gr.File(label="Download files", file_count="multiple")

        run_button.click(
            fn=process,
            inputs=[media_input, language_input, translate_input, formats_input],
            outputs=[text_output, status_output, files_output],
        )

    return demo


def main() -> None:
    """Launch the Gradio app."""
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()