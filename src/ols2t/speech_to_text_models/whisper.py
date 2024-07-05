from collections.abc import Generator

from faster_whisper import WhisperModel

from ols2t.models import BaseStream, Segment

from ..settings import (
    WhisperSpeechToTextModelLanguage,
    WhisperSpeechToTextModelPathOrModelSize,
    WhisperSpeechToTextModelSize,
)
from .base import BaseSpeechToTextModel


class WhisperSpeechToTextModel(BaseSpeechToTextModel):
    def __init__(
        self, path_or_model_size: WhisperSpeechToTextModelPathOrModelSize, language: WhisperSpeechToTextModelLanguage
    ):
        self._path_or_model_size = path_or_model_size
        self._language = language
        self._model_cache: WhisperModel | None = None

    @property
    def model_cache(self) -> WhisperModel:
        if self._model_cache is None:
            self._model_cache = WhisperModel(
                model_size_or_path=(
                    self._path_or_model_size
                    if isinstance(self._path_or_model_size, WhisperSpeechToTextModelSize)
                    else str(self._path_or_model_size)
                ),
            )
        return self._model_cache

    def transcribe(self, input_stream: BaseStream) -> Generator[Segment, None, None]:
        with input_stream as s:
            segments, info = self.model_cache.transcribe(s)
            for segment in segments:
                yield Segment(start=segment.start, end=segment.end, text=segment.text, probability=segment.avg_logprob)
