"""Domain models for transcription results."""

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A single time-stamped chunk of transcribed speech."""

    start: float = Field(..., description="Segment start time in seconds.")
    end: float = Field(..., description="Segment end time in seconds.")
    text: str = Field(..., description="Transcribed text for this segment.")


class TranscriptionResult(BaseModel):
    """The full result of transcribing one audio file."""

    language: str = Field(..., description="Detected language code, e.g. 'fa' or 'en'.")
    language_probability: float = Field(
        ..., description="Confidence of the language detection (0-1)."
    )
    segments: list[Segment] = Field(
        default_factory=list, description="Ordered time-stamped segments."
    )

    @property
    def full_text(self) -> str:
        """Join all segment texts into one continuous transcript."""
        return " ".join(segment.text.strip() for segment in self.segments)