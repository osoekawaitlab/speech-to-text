from collections.abc import Generator

from .models import BaseStream, Segment
from .settings import SpeechToTextCoreSettings
from .speech_to_text_models.base import BaseSpeechToTextModel
from .speech_to_text_models.factory import create_speech_to_text_model


class SpeechToTextCore:
    def __init__(self, model: BaseSpeechToTextModel) -> None:
        self._model = model

    @property
    def model(self) -> BaseSpeechToTextModel:
        return self._model

    def transcribe(self, input_stream: BaseStream) -> Generator[Segment, None, None]:
        yield from self.model.transcribe(input_stream=input_stream)

    @classmethod
    def create(cls, settings: SpeechToTextCoreSettings) -> "SpeechToTextCore":
        return cls(model=create_speech_to_text_model(settings=settings.speech_to_text_model_settings))
