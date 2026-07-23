"""Command-line interface for EchoExtract."""

import argparse
from pathlib import Path

from echo_extract.core.config import settings
from echo_extract.engines.faster_whisper_engine import FasterWhisperEngine
from echo_extract.pipeline import transcribe_video
from echo_extract.core.logging_config import setup_logging


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
        "-m", "--model", default=settings.model_size, help="Whisper model size."
    )
    parser.add_argument(
        "--device", default=settings.device, help="'cuda' or 'cpu'."
    )
    parser.add_argument(
        "--keep-audio", action="store_true", help="Keep the intermediate WAV file."
    )
    parser.add_argument(
        "-f", "--format", action="append", dest="formats",
        choices=["txt", "srt", "vtt", "json"],
        help="Output format (repeatable). Defaults to all four.",
    )
    parser.add_argument(
        "-l", "--language", default=None,
        help="Force input language (e.g. 'fa', 'en'). Omit to auto-detect.",
    )
    parser.add_argument(
        "--start", type=float, default=None,
        help="Start time in seconds (e.g. 300 for 5:00).",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Duration in seconds to transcribe (e.g. 180 for 3 minutes).",
    )
    parser.add_argument(
        "--translate-to", default=None, dest="translate_to",
        help="Translate output to this language. Currently only 'en' is supported.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed debug logs.",
    )
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if args.translate_to is not None and args.translate_to != "en":
        print(
            f"Error: translation to '{args.translate_to}' is not supported yet. "
            "Whisper can only translate to English ('en')."
        )
        raise SystemExit(1)

    # If no format was given, default to all four.
    formats = args.formats if args.formats else settings.default_formats

    if not args.video.exists():
        print(f"Error: video not found: {args.video}")
        raise SystemExit(1)

    output_dir = args.output if args.output is not None else args.video.parent

    print(f"Loading model '{args.model}' on {args.device}...")
    engine = FasterWhisperEngine(model_size=args.model, device=args.device)

    print(f"Transcribing {args.video.name}...")
    result = transcribe_video(
        args.video, engine, output_dir, formats,
        language=args.language,
        translate_to=args.translate_to,
        start_time=args.start,
        duration=args.duration,
        keep_audio=args.keep_audio,
    )

    print(
        f"Done. Language: {result.language} "
        f"({result.language_probability:.2f}), {len(result.segments)} segments."
    )
    print(f"Outputs written to {output_dir}/")


if __name__ == "__main__":
    main()