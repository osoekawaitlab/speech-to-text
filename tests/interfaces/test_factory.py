from argparse import ArgumentParser

import pytest
from pytest_mock import MockerFixture

from ols2t.app import SpeechToTextApp
from ols2t.interfaces.factory import create_interface
from ols2t.settings import BaseInterfaceSettings, CliInterfaceSettings, InterfaceType


def test_create_interface_creates_cli(mocker: MockerFixture) -> None:
    CliInterface = mocker.patch("ols2t.interfaces.factory.CliInterface")
    settings = CliInterfaceSettings()
    basic_argument_parser = mocker.MagicMock(spec=ArgumentParser)
    core = mocker.MagicMock(spec=SpeechToTextApp)
    actual = create_interface(settings=settings, core=core, basic_argument_parser=basic_argument_parser)
    assert actual == CliInterface.return_value
    CliInterface.assert_called_once_with(core=core, basic_argument_parser=basic_argument_parser)


def test_create_interface_raises_value_error(mocker: MockerFixture) -> None:
    bad_settings = BaseInterfaceSettings(type=InterfaceType.CLI)
    basic_argument_parser = mocker.MagicMock(spec=ArgumentParser)
    core = mocker.MagicMock(spec=SpeechToTextApp)
    with pytest.raises(ValueError):
        create_interface(
            settings=bad_settings, core=core, basic_argument_parser=basic_argument_parser  # type: ignore[arg-type]
        )
