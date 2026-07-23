"""Audio extraction from video files using ffmpeg."""

import subprocess
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

class AudioExtractionError(Exception):
    """Raised when ffmpeg fails to extract audio from a video file."""


def extract_audio(
    video_path: Path,
    output_path: Path,
    start_time: float | None = None,
    duration: float | None = None,
) -> Path:
    """Extract mono 16kHz WAV audio from a video file using ffmpeg.

    Whisper models are trained on 16kHz mono audio, so we convert to that
    exact format for best transcription accuracy.

    Args:
        video_path: Path to the source video file (e.g. an mp4).
        output_path: Path where the extracted WAV file will be written.

    Returns:
        The path to the created WAV file.

    Raises:
        FileNotFoundError: If the source video does not exist.
        AudioExtractionError: If ffmpeg fails for any reason.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Ensure the output directory exists before ffmpeg writes to it.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = ["ffmpeg"]

    # Seek to start time before input for fast seeking (if provided).
    if start_time is not None:
        command += ["-ss", str(start_time)]

    command += ["-i", str(video_path)]

    # Limit duration (if provided).
    if duration is not None:
        command += ["-t", str(duration)]

    command += [
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-y",
        str(output_path),
    ]

    # Run ffmpeg. capture_output hides its verbose logs unless we need them.
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    # A non-zero return code means ffmpeg failed; surface its error message.
    if result.returncode != 0:
        logger.error("ffmpeg failed for %s", video_path)
        raise AudioExtractionError(
            f"ffmpeg failed for {video_path}:\n{result.stderr}"
        )

    return output_path