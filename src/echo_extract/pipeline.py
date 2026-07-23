"""Orchestrates the full transcription workflow: video -> audio -> text."""

from pathlib import Path

from echo_extract.core.models import TranscriptionResult
from echo_extract.engines.base import TranscriptionEngine
from echo_extract.io.audio import prepare_audio
from echo_extract.io.writers import write_formats
import logging

logger = logging.getLogger(__name__)

def transcribe_media(
    source_path: Path,
    engine: TranscriptionEngine,
    output_dir: Path,
    formats: list[str],
    language: str | None = None,
    translate_to: str | None = None,
    start_time: float | None = None,
    duration: float | None = None,
    keep_audio: bool = False,
) -> TranscriptionResult:
    """Run the full pipeline: extract audio, transcribe, optionally translate.

    Args:
        video_path: Path to the source video file.
        engine: Any transcription engine implementing the interface.
        output_dir: Directory where output files will be written.
        formats: Which output formats to write.
        language: Optional source language code. None means auto-detect.
        translate_to: If set, also write a translation in this language.
        start_time: Optional start time in seconds.
        duration: Optional duration in seconds.
        keep_audio: If False, the intermediate WAV file is deleted afterward.

    Returns:
        The TranscriptionResult in the source language.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = source_path.stem

    # Step 1: extract audio, optionally limited to a time range.
    logger.info("Starting pipeline for %s", source_path.name)
    audio_path = output_dir / f"{base_name}.wav"
    prepare_audio(source_path, audio_path, start_time=start_time, duration=duration)

    # Step 2: transcribe in the source language.
    logger.info("Transcribing audio (language=%s)", language or "auto")
    result = engine.transcribe(audio_path, language=language, task="transcribe")

    logger.info("Detected language: %s (%.2f)", result.language, result.language_probability)

    # Step 3: write source-language outputs, suffixed with the detected language.
    written = write_formats(result, output_dir, base_name, formats, suffix=result.language)
    logger.info("Wrote %d file(s) for source language", len(written))

    # Step 4: optionally translate and write a second set of files.
    if translate_to is not None:
        logger.info("Translating to %s", translate_to)
        translated = engine.transcribe(audio_path, language=language, task="translate")
        write_formats(translated, output_dir, base_name, formats, suffix=translate_to)

    # Step 5: clean up the intermediate audio file.
    if not keep_audio:
        audio_path.unlink(missing_ok=True)
        logger.debug("Removed intermediate audio file")

    return result