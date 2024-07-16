import pytest

import ols2t


@pytest.mark.slow
def test_ols2t_as_module_transcribe(hello_fixture: ols2t.FileStream) -> None:
    settings = ols2t.SpeechToTextCoreSettings(
        speech_to_text_model_settings={"type": "WHISPER", "path_or_model_size": "small", "language": "ja"}
    )
    core = ols2t.SpeechToTextCore.create(settings=settings)
    actual = core.transcribe(input_stream=hello_fixture)
    assert "".join(s.text for s in actual) == "こんにちは"
