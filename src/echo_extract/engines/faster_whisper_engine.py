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
from faster_whisper import WhisperModel

from echo_extract.core.models import Segment, TranscriptionResult
from echo_extract.engines.base import TranscriptionEngine


class FasterWhisperEngine(TranscriptionEngine):
    """Transcription engine that runs a Whisper model locally."""

    def __init__(
        self,
        model_size: str = settings.model_size,
        device: str = settings.device,
        compute_type: str = settings.compute_type,
    ) -> None:
        """Load the Whisper model into memory."""
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """Transcribe audio and return a structured result.

        Args:
            audio_path: Path to a WAV audio file.
            language: Optional language code (e.g. 'fa', 'en').
                If None, Whisper auto-detects the language.
        """
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            beam_size=settings.beam_size,
            language=language,
            task=task,
        )

        segments = [
            Segment(start=seg.start, end=seg.end, text=seg.text)
            for seg in segments_iter
        ]

        return TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            segments=segments,
        )