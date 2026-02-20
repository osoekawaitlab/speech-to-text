from enum import Enum
from typing import Annotated, Literal, Union

from oltl.settings import BaseSettings as OltlBaseSettings
from pydantic import DirectoryPath, Field
from pydantic_settings import SettingsConfigDict


class BaseSettings(OltlBaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLS2T_")


class SpeechToTextModelType(str, Enum):
    WHISPER = "WHISPER"
    SEGMENT_MERGING = "SEGMENT_MERGING"


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
    LARGE_V3_TURBO = "large-v3-turbo"
    LARGE = "large"
    DISTIL_LARGE_V2 = "distil-large-v2"
    DISTIL_LARGE_V3 = "distil-large-v3"


class WhisperSpeechToTextModelLanguage(str, Enum):
    EN = "en"
    JA = "ja"


class WhisperSpeechToTextModelDevice(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class BaseSpeechToTextModelSettings(BaseSettings):
    type: SpeechToTextModelType


WhisperSpeechToTextModelPathOrModelSize = WhisperSpeechToTextModelSize | DirectoryPath


class WhisperSpeechToTextModelSettings(BaseSpeechToTextModelSettings):
    type: Literal[SpeechToTextModelType.WHISPER] = SpeechToTextModelType.WHISPER
    path_or_model_size: WhisperSpeechToTextModelPathOrModelSize
    language: WhisperSpeechToTextModelLanguage
    device: WhisperSpeechToTextModelDevice = WhisperSpeechToTextModelDevice.CPU


class SegmentMergingSpeechToTextModelSettings(BaseSpeechToTextModelSettings):
    type: Literal[SpeechToTextModelType.SEGMENT_MERGING] = SpeechToTextModelType.SEGMENT_MERGING
    speech_to_text_model_settings: "SpeechToTextModelSettings"


SpeechToTextModelSettings = Annotated[
    Union[WhisperSpeechToTextModelSettings, SegmentMergingSpeechToTextModelSettings], Field(discriminator="type")
]


class SpeechToTextCoreSettings(BaseSettings):
    speech_to_text_model_settings: SpeechToTextModelSettings


class InterfaceType(str, Enum):
    CLI = "CLI"
    HTTP_API = "HTTP_API"


class BaseInterfaceSettings(BaseSettings):
    type: InterfaceType


class CliSettings(BaseInterfaceSettings):
    type: Literal[InterfaceType.CLI] = InterfaceType.CLI


class HttpApiSettings(BaseInterfaceSettings):
    type: Literal[InterfaceType.HTTP_API] = InterfaceType.HTTP_API
    host: str = "0.0.0.0"
    port: int = 8000


InterfaceSettings = Annotated[Union[CliSettings, HttpApiSettings], Field(discriminator="type")]


class SpeechToTextAppSettings(BaseSettings):
    core_settings: SpeechToTextCoreSettings
    interface_settings: InterfaceSettings
