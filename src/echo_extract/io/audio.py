"""Audio extraction from video files using ffmpeg."""

import subprocess
from pathlib import Path


class AudioExtractionError(Exception):
    """Raised when ffmpeg fails to extract audio from a video file."""


def extract_audio(video_path: Path, output_path: Path) -> Path:
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

    # Build the ffmpeg command as a list so paths with spaces are safe.
    command = [
        "ffmpeg",
        "-i", str(video_path),   # input file
        "-vn",                   # drop the video stream
        "-ar", "16000",          # audio sample rate: 16kHz
        "-ac", "1",              # audio channels: mono
        "-c:a", "pcm_s16le",     # standard 16-bit WAV codec
        "-y",                    # overwrite output if it exists
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
        raise AudioExtractionError(
            f"ffmpeg failed for {video_path}:\n{result.stderr}"
        )

    return output_path