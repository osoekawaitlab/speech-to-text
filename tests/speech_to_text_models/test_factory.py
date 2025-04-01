import pytest
from pytest_mock import MockerFixture

from ols2t.settings import (
    BaseSpeechToTextModelSettings,
    SegmentMergingSpeechToTextModelSettings,
    SpeechToTextModelType,
    WhisperSpeechToTextModelLanguage,
    WhisperSpeechToTextModelSettings,
    WhisperSpeechToTextModelSize,
)
from ols2t.speech_to_text_models.factory import create_speech_to_text_model


def test_factory_generates_whisper_speech_to_text_model(mocker: MockerFixture) -> None:
    WhiepserSpeechToTextModel = mocker.patch("ols2t.speech_to_text_models.factory.WhisperSpeechToTextModel")
    settings = WhisperSpeechToTextModelSettings(
        path_or_model_size=WhisperSpeechToTextModelSize.TINY,
        language=WhisperSpeechToTextModelLanguage.JA,
    )
    create_speech_to_text_model(settings=settings)
    WhiepserSpeechToTextModel.assert_called_once_with(
        path_or_model_size=WhisperSpeechToTextModelSize.TINY, language=WhisperSpeechToTextModelLanguage.JA
    )


def test_factory_raises_value_error_when_model_type_is_unknown(mocker: MockerFixture) -> None:
    settings = BaseSpeechToTextModelSettings(type=SpeechToTextModelType.WHISPER)
    with pytest.raises(ValueError):
        create_speech_to_text_model(settings=settings)  # type: ignore[arg-type]


def test_factory_generates_segment_merging_speech_to_text_model(mocker: MockerFixture) -> None:
    WhisperSpeechToTextModel = mocker.patch("ols2t.speech_to_text_models.factory.WhisperSpeechToTextModel")
    SegmentMergingSpeechToTextModel = mocker.patch(
        "ols2t.speech_to_text_models.factory.SegmentMergingSpeechToTextModel"
    )
    settings = SegmentMergingSpeechToTextModelSettings(
        model_settings=WhisperSpeechToTextModelSettings(
            path_or_model_size=WhisperSpeechToTextModelSize.TINY,
            language=WhisperSpeechToTextModelLanguage.JA,
        )
    )
    create_speech_to_text_model(settings=settings)
    SegmentMergingSpeechToTextModel.assert_called_once_with(model=WhisperSpeechToTextModel.return_value)
    WhisperSpeechToTextModel.assert_called_once_with(
        path_or_model_size=WhisperSpeechToTextModelSize.TINY, language=WhisperSpeechToTextModelLanguage.JA
    )
