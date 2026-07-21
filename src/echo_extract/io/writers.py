"""Write transcription results to txt, srt, vtt, and json files."""

import json
from pathlib import Path

from echo_extract.core.models import TranscriptionResult


def _format_timestamp(seconds: float, use_comma: bool = True) -> str:
    """Format a time in seconds as HH:MM:SS,mmm (srt) or HH:MM:SS.mmm (vtt).

    Args:
        seconds: The time in seconds.
        use_comma: If True use a comma before milliseconds (srt style),
            otherwise use a dot (vtt style).

    Returns:
        A formatted timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    separator = "," if use_comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def write_txt(result: TranscriptionResult, output_path: Path) -> Path:
    """Write the full transcript as plain text."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.full_text, encoding="utf-8")
    return output_path


def write_srt(result: TranscriptionResult, output_path: Path) -> Path:
    """Write the transcript as an SRT subtitle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for index, segment in enumerate(result.segments, start=1):
        start = _format_timestamp(segment.start, use_comma=True)
        end = _format_timestamp(segment.end, use_comma=True)
        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(segment.text.strip())
        lines.append("")  # blank line between entries

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_vtt(result: TranscriptionResult, output_path: Path) -> Path:
    """Write the transcript as a WebVTT subtitle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["WEBVTT", ""]  # required header
    for segment in result.segments:
        start = _format_timestamp(segment.start, use_comma=False)
        end = _format_timestamp(segment.end, use_comma=False)
        lines.append(f"{start} --> {end}")
        lines.append(segment.text.strip())
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_json(result: TranscriptionResult, output_path: Path) -> Path:
    """Write the full result (language + segments) as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pydantic's model_dump gives us a clean serializable dict.
    data = result.model_dump()
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path

# A registry mapping each format name to its writer function.
WRITERS = {
    "txt": write_txt,
    "srt": write_srt,
    "vtt": write_vtt,
    "json": write_json,
}


def write_formats(
    result: TranscriptionResult,
    output_dir: Path,
    base_name: str,
    formats: list[str],
) -> list[Path]:
    """Write the result in the requested formats only.

    Args:
        result: The transcription result to write.
        output_dir: Directory for the output files.
        base_name: File name without extension.
        formats: Which formats to write, e.g. ["srt", "json"].

    Returns:
        The list of file paths that were written.
    """
    written: list[Path] = []
    for fmt in formats:
        writer = WRITERS[fmt]
        path = writer(result, output_dir / f"{base_name}.{fmt}")
        written.append(path)
    return written