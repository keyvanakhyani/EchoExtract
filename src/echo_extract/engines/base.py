"""Abstract base class defining the transcription engine contract."""

from abc import ABC, abstractmethod
from pathlib import Path

from echo_extract.core.models import TranscriptionResult


class TranscriptionEngine(ABC):
    """Contract that every transcription backend must follow.

    Using a shared interface lets us swap engines (local Whisper, a cloud
    API, etc.) without changing the rest of the application.
    """

    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file into a structured result.

        Args:
            audio_path: Path to a WAV audio file.

        Returns:
            A TranscriptionResult with detected language and segments.
        """
        ...