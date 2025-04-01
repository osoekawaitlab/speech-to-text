from ..settings import (
    SegmentMergingSpeechToTextModelSettings,
    SpeechToTextModelSettings,
    WhisperSpeechToTextModelSettings,
)
from .base import BaseSpeechToTextModel
from .segment_merging import SegmentMergingSpeechToTextModel
from .whisper import WhisperSpeechToTextModel


def create_speech_to_text_model(settings: SpeechToTextModelSettings) -> BaseSpeechToTextModel:
    if isinstance(settings, WhisperSpeechToTextModelSettings):
        return WhisperSpeechToTextModel(path_or_model_size=settings.path_or_model_size, language=settings.language)
    elif isinstance(settings, SegmentMergingSpeechToTextModelSettings):
        model = create_speech_to_text_model(settings=settings.speech_to_text_model_settings)
        return SegmentMergingSpeechToTextModel(model=model)
    raise ValueError(f"Unknown model type: {settings.type}")
