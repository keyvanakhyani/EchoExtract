"""Orchestrates the full transcription workflow: video -> audio -> text."""

from pathlib import Path

from echo_extract.core.models import TranscriptionResult
from echo_extract.engines.base import TranscriptionEngine
from echo_extract.io.audio import extract_audio
from echo_extract.io.writers import write_formats


def transcribe_video(
    video_path: Path,
    engine: TranscriptionEngine,
    output_dir: Path,
    formats: list[str],
    keep_audio: bool = False,
) -> TranscriptionResult:
    """Run the full pipeline: extract audio, transcribe, write all formats.

    Args:
        video_path: Path to the source video file.
        engine: Any transcription engine implementing the interface.
        output_dir: Directory where output files will be written.
        keep_audio: If False, the intermediate WAV file is deleted afterward.

    Returns:
        The TranscriptionResult produced by the engine.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a base name from the video file (e.g. "lecture.mp4" -> "lecture").
    base_name = video_path.stem

    # Step 1: extract audio to a temporary WAV next to the outputs.
    audio_path = output_dir / f"{base_name}.wav"
    extract_audio(video_path, audio_path)

    # Step 2: transcribe the audio.
    result = engine.transcribe(audio_path)

    # Step 3: write every output format using the shared base name.
    write_formats(result, output_dir, base_name, formats)

    # Step 4: optionally clean up the intermediate audio file.
    if not keep_audio:
        audio_path.unlink(missing_ok=True)

    return result