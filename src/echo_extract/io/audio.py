"""Audio extraction from video files using ffmpeg."""

import subprocess
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

# Common audio file extensions. Anything else is treated as video.
AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".wav", ".ogg", ".oga", ".m4a", ".flac", ".aac", ".wma", ".opus"}
)


def is_audio_file(path: Path) -> bool:
    """Return True if the path points to an audio file rather than a video."""
    return path.suffix.lower() in AUDIO_EXTENSIONS

class AudioExtractionError(Exception):
    """Raised when ffmpeg fails to extract audio from a video file."""


def prepare_audio(
    source_path: Path,
    output_path: Path,
    start_time: float | None = None,
    duration: float | None = None,
) -> Path:
    """Convert a video or audio file into mono 16kHz WAV for Whisper.

    Works for both video files (audio is extracted) and audio files
    (audio is simply re-encoded to the required format).

    Args:
        source_path: Path to the source video or audio file.
        output_path: Path where the WAV file will be written.
        start_time: Optional start offset in seconds.
        duration: Optional duration in seconds.

    Returns:
        The path to the created WAV file.

    Raises:
        FileNotFoundError: If the source file does not exist.
        AudioExtractionError: If ffmpeg fails.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    kind = "audio" if is_audio_file(source_path) else "video"
    logger.info("Preparing audio from %s file: %s", kind, source_path.name)

    command = ["ffmpeg"]

    if start_time is not None:
        command += ["-ss", str(start_time)]

    command += ["-i", str(source_path)]

    if duration is not None:
        command += ["-t", str(duration)]

    # -vn is harmless for audio-only inputs and drops video streams otherwise.
    command += [
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-y",
        str(output_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("ffmpeg failed for %s", source_path)
        raise AudioExtractionError(
            f"ffmpeg failed for {source_path}:\n{result.stderr}"
        )

    return output_path