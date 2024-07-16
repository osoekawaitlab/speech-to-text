from pytest_mock import MockerFixture

from ols2t.app import SpeechToTextApp
from ols2t.settings import SpeechToTextAppSettings


def test_create_app(mocker: MockerFixture) -> None:
    SpeechToTextCore = mocker.patch("ols2t.app.SpeechToTextCore")
    create_interface = mocker.patch("ols2t.app.create_interface")
    settings = SpeechToTextAppSettings(
        core_settings={
            "speech_to_text_model_settings": {"type": "WHISPER", "path_or_model_size": "small", "language": "ja"}
        },
        interface_settings={"type": "CLI"},
    )
    app = SpeechToTextApp.create(settings=settings)
    app.run()
    create_interface.return_value.run.assert_called_once_with()

    SpeechToTextCore.create.assert_called_once_with(settings=settings.core_settings)
    create_interface.assert_called_once_with(
        settings=settings.interface_settings, core=SpeechToTextCore.create.return_value
    )
