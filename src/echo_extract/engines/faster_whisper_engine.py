"""Local transcription engine powered by faster-whisper."""

import os
import sysconfig
from pathlib import Path

from echo_extract.core.config import settings

# Point HuggingFace cache to the location from settings so downloaded
# models don't fill up drive C.
os.environ.setdefault("HF_HOME", str(settings.hf_home))

# faster-whisper needs NVIDIA's cublas/cudnn DLLs at runtime. They ship
# inside the installed nvidia-* pip packages under site-packages/nvidia.
# We locate site-packages dynamically so this works on any machine, then
# register the DLL folders both via add_dll_directory and PATH (ctranslate2
# resolves cublas lazily at run time via PATH).
_site_packages = Path(sysconfig.get_paths()["purelib"])
_nvidia_root = _site_packages / "nvidia"

for _sub in ("cublas", "cudnn"):
    _dll_dir = _nvidia_root / _sub / "bin"
    if _dll_dir.exists():
        os.add_dll_directory(str(_dll_dir))
        os.environ["PATH"] = str(_dll_dir) + os.pathsep + os.environ["PATH"]

# Import faster_whisper AFTER the DLL directories are registered above.
from faster_whisper import WhisperModel, BatchedInferencePipeline

from echo_extract.core.models import Segment, TranscriptionResult
from echo_extract.engines.base import TranscriptionEngine
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn


class FasterWhisperEngine(TranscriptionEngine):
    """Transcription engine that runs a Whisper model locally."""

    def __init__(
        self,
        model_size: str = settings.model_size,
        device: str = settings.device,
        compute_type: str = settings.compute_type,
    ) -> None:
        """Load the Whisper model and wrap it in a batched pipeline."""
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        # Batched pipeline processes multiple chunks in parallel on the GPU.
        self.batched_model = BatchedInferencePipeline(model=self.model)

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """Transcribe or translate audio, showing live progress."""
        segments_iter, info = self.batched_model.transcribe(
            str(audio_path),
            beam_size=settings.beam_size,
            language=language,
            task=task,
            vad_filter=True,
            batch_size=settings.batch_size,
        )

        segments: list[Segment] = []
        total = info.duration  # total audio length in seconds

        with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            job = progress.add_task(f"{task.capitalize()}", total=total)

            # The iterator yields segments lazily as audio is processed,
            # so we can report progress based on each segment's end time.
            for seg in segments_iter:
                segments.append(
                    Segment(start=seg.start, end=seg.end, text=seg.text)
                )
                progress.update(job, completed=min(seg.end, total))

            progress.update(job, completed=total)

        return TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            segments=segments,
        )