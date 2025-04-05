from collections.abc import Generator

from faster_whisper import WhisperModel

from ols2t.models import BaseStream, Segment

from ..settings import (
    WhisperSpeechToTextModelDevice,
    WhisperSpeechToTextModelLanguage,
    WhisperSpeechToTextModelPathOrModelSize,
    WhisperSpeechToTextModelSize,
)
from .base import BaseSpeechToTextModel


class WhisperSpeechToTextModel(BaseSpeechToTextModel):
    def __init__(
        self,
        path_or_model_size: WhisperSpeechToTextModelPathOrModelSize,
        language: WhisperSpeechToTextModelLanguage,
        device: WhisperSpeechToTextModelDevice = WhisperSpeechToTextModelDevice.CPU,
    ):
        self._path_or_model_size = path_or_model_size
        self._language = language
        self._model_cache: WhisperModel | None = None
        self._device = device

    @property
    def model_cache(self) -> WhisperModel:
        if self._model_cache is None:
            self._model_cache = WhisperModel(
                model_size_or_path=(
                    self._path_or_model_size.value
                    if isinstance(self._path_or_model_size, WhisperSpeechToTextModelSize)
                    else str(self._path_or_model_size)
                ),
                device=self._device.value,
            )
        return self._model_cache

    def transcribe(self, input_stream: BaseStream) -> Generator[Segment, None, None]:
        with input_stream as s:
            for chunk in s:
                segments, _ = self.model_cache.transcribe(
                    chunk,
                    language=self._language.value,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters={
                        "threshold": 0.3,
                        "min_speech_duration_ms": 100,
                        "max_speech_duration_s": 10,
                        "min_silence_duration_ms": 100,
                    },
                )
                for segment in segments:
                    for word in segment.words:
                        yield Segment(start=word.start, end=word.end, text=word.word, probability=word.probability)
