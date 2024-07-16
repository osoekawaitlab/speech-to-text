from argparse import ArgumentParser

from pytest_mock import MockerFixture

from ols2t.app import SpeechToTextApp
from ols2t.settings import SpeechToTextAppSettings


def test_create_app(mocker: MockerFixture) -> None:
    SpeechToTextCore = mocker.patch("ols2t.app.SpeechToTextCore")
    core = SpeechToTextCore.create.return_value
    create_interface = mocker.patch("ols2t.app.create_interface")
    interface = create_interface.return_value
    settings = SpeechToTextAppSettings(
        core_settings={
            "speech_to_text_model_settings": {"type": "WHISPER", "path_or_model_size": "small", "language": "ja"}
        },
        interface_settings={"type": "CLI"},
    )
    basic_argument_parser = mocker.MagicMock(spec=ArgumentParser)
    app = SpeechToTextApp.create(settings=settings, basic_argument_parser=basic_argument_parser)
    app.run()
    interface.run.assert_called_once_with()

    SpeechToTextCore.create.assert_called_once_with(settings=settings.core_settings)
    create_interface.assert_called_once_with(
        settings=settings.interface_settings, core=core, basic_argument_parser=basic_argument_parser
    )
