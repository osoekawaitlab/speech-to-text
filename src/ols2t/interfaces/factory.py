from argparse import ArgumentParser

from ..core import SpeechToTextCore
from ..settings import CliSettings, HttpApiSettings, InterfaceSettings
from .base import BaseInterface
from .cli import Cli


def create_interface(
    settings: InterfaceSettings, core: SpeechToTextCore, basic_argument_parser: ArgumentParser
) -> BaseInterface:
    if isinstance(settings, CliSettings):
        return Cli(core=core, basic_argument_parser=basic_argument_parser)
    if isinstance(settings, HttpApiSettings):
        from .http_api import HttpApi

        return HttpApi(core=core, settings=settings)
    raise ValueError(f"Unsupported interface type: {settings.type}")
