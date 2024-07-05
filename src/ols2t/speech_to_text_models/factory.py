from ..settings import SpeechToTextModelSettings, WhisperSpeechToTextModelSettings
from .base import BaseSpeechToTextModel
from .whisper import WhisperSpeechToTextModel


def create_speech_to_text_model(settings: SpeechToTextModelSettings) -> BaseSpeechToTextModel:
    if isinstance(settings, WhisperSpeechToTextModelSettings):
        return WhisperSpeechToTextModel(path_or_model_size=settings.path_or_model_size, language=settings.language)
    raise ValueError(f"Unknown model type: {settings.type}")
