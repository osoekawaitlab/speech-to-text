from pytest_mock import MockerFixture

from ols2t.core import SpeechToTextCore
from ols2t.models import FileStream, Segment
from ols2t.settings import (
    SpeechToTextCoreSettings,
    SpeechToTextModelType,
    WhisperSpeechToTextModelLanguage,
    WhisperSpeechToTextModelSize,
)
from ols2t.speech_to_text_models.base import BaseSpeechToTextModel


def test_speech_to_text_core_create(mocker: MockerFixture) -> None:
    create_speech_to_text_model = mocker.patch("ols2t.core.create_speech_to_text_model")

    settings = SpeechToTextCoreSettings(
        speech_to_text_model_settings={
            "type": SpeechToTextModelType.WHISPER,
            "path_or_model_size": WhisperSpeechToTextModelSize.TINY,
            "language": WhisperSpeechToTextModelLanguage.EN,
        }
    )
    actual = SpeechToTextCore.create(settings=settings)
    assert isinstance(actual, SpeechToTextCore)
    create_speech_to_text_model.assert_called_once_with(settings=settings.speech_to_text_model_settings)
    assert actual.model == create_speech_to_text_model.return_value


def test_speech_to_text_core_transcribe(mocker: MockerFixture, hello_fixture: FileStream) -> None:
    model = mocker.Mock(spec=BaseSpeechToTextModel)
    segments = [
        Segment(text="こんにちは", start=0.0, end=2.0, probability=0.9),
        Segment(text="さようなら", start=0.0, end=2.0, probability=0.9),
    ]

    model.transcribe.return_value = iter(segments)
    core = SpeechToTextCore(model=model)

    actual = core.transcribe(input_stream=hello_fixture)
    assert list(actual) == segments
    model.transcribe.assert_called_once_with(input_stream=hello_fixture)
