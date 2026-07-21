"""Local transcription engine powered by faster-whisper."""

import os

# Point HuggingFace cache to drive D so models don't fill up drive C.
os.environ["HF_HOME"] = r"D:\whisper_models\huggingface"
from pathlib import Path

# CUDA DLLs must be discoverable BEFORE faster_whisper is imported.
# os.add_dll_directory handles the initial load, but ctranslate2 also
# resolves cublas lazily at run time via the PATH env var, so we add the
# directories to PATH as well to cover both cases.
_nvidia_root = Path(r"D:\python workspace\EchoExtract\venv\Lib\site-packages\nvidia")
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
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        download_root="D:/whisper_models",
    ) -> None:
        """Load the Whisper model into memory."""
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio and return a structured result."""
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
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