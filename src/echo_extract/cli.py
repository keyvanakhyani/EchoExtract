"""Command-line interface for EchoExtract."""

import argparse
from pathlib import Path

from echo_extract.engines.faster_whisper_engine import FasterWhisperEngine
from echo_extract.pipeline import transcribe_video


def main() -> None:
    """Parse arguments and run the transcription pipeline."""
    parser = argparse.ArgumentParser(
        description="Transcribe videos to text using local Whisper."
    )
    parser.add_argument("video", type=Path, help="Path to the video file.")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output folder. Defaults to the video's own folder.",
    )
    parser.add_argument(
        "-m", "--model", default="large-v3", help="Whisper model size."
    )
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'.")
    parser.add_argument(
        "--keep-audio", action="store_true", help="Keep the intermediate WAV file."
    )
    parser.add_argument(
        "-f", "--format", action="append", dest="formats",
        choices=["txt", "srt", "vtt", "json"],
        help="Output format (repeatable). Defaults to all four.",
    )
    args = parser.parse_args()

    # If no format was given, default to all four.
    formats = args.formats if args.formats else ["txt", "srt", "vtt", "json"]

    if not args.video.exists():
        print(f"Error: video not found: {args.video}")
        raise SystemExit(1)

    output_dir = args.output if args.output is not None else args.video.parent

    print(f"Loading model '{args.model}' on {args.device}...")
    engine = FasterWhisperEngine(model_size=args.model, device=args.device)

    print(f"Transcribing {args.video.name}...")
    result = transcribe_video(
        args.video, engine, output_dir, formats, keep_audio=args.keep_audio
    )

    print(
        f"Done. Language: {result.language} "
        f"({result.language_probability:.2f}), {len(result.segments)} segments."
    )
    print(f"Outputs written to {output_dir}/")


if __name__ == "__main__":
    main()