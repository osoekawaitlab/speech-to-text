from enum import Enum
from typing import Annotated, Literal

from oltl.settings import BaseSettings as OltlBaseSettings
from pydantic import DirectoryPath, Field
from pydantic_settings import SettingsConfigDict


class BaseSettings(OltlBaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLS2T_")


class SpeechToTextModelType(str, Enum):
    WHISPER = "WHISPER"


class WhisperSpeechToTextModelSize(str, Enum):
    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    DISTIL_SMALL_EN = "distil-small.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    DISTIL_MEDIUM_EN = "distil-medium.en"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE = "large"
    DISTIL_LARGE_V2 = "distil-large-v2"
    DISTIL_LARGE_V3 = "distil-large-v3"


class WhisperSpeechToTextModelLanguage(str, Enum):
    EN = "en"
    JA = "ja"


class BaseSpeechToTextModelSettings(BaseSettings):
    type: SpeechToTextModelType


WhisperSpeechToTextModelPathOrModelSize = WhisperSpeechToTextModelSize | DirectoryPath


class WhisperSpeechToTextModelSettings(BaseSpeechToTextModelSettings):
    type: Literal[SpeechToTextModelType.WHISPER] = SpeechToTextModelType.WHISPER
    path_or_model_size: WhisperSpeechToTextModelPathOrModelSize
    language: WhisperSpeechToTextModelLanguage


SpeechToTextModelSettings = Annotated[WhisperSpeechToTextModelSettings, Field(discriminator="type")]


class SpeechToTextCoreSettings(BaseSettings):
    speech_to_text_model_settings: SpeechToTextModelSettings
