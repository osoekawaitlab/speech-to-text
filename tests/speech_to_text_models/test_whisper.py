from ols2t.models import FileStream, Segment
from ols2t.settings import (
    WhisperSpeechToTextModelLanguage,
    WhisperSpeechToTextModelSize,
)
from ols2t.speech_to_text_models.whisper import WhisperSpeechToTextModel


def test_whisper_speech_to_text_model_transcribe(hello_fixture: FileStream) -> None:
    model = WhisperSpeechToTextModel(
        path_or_model_size=WhisperSpeechToTextModelSize.SMALL, language=WhisperSpeechToTextModelLanguage.JA
    )
    expected = [Segment(start=0.0, end=1.04, text="こんにちは", probability=0.96)]
    actual = model.transcribe(input_stream=hello_fixture)
    cnt = 0
    for a, e in zip(actual, expected):
        assert abs(a.start - e.start) < 0.01
        assert abs(a.end - e.end) < 0.01
        assert abs(a.probability - e.probability) < 0.1
        assert a.text == e.text
        cnt += 1
    assert cnt == 1
