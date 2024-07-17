from argparse import ArgumentParser

from ..core import SpeechToTextCore
from ..settings import CliSettings, InterfaceSettings
from .base import BaseInterface
from .cli import Cli


def create_interface(
    settings: InterfaceSettings, core: SpeechToTextCore, basic_argument_parser: ArgumentParser
) -> BaseInterface:
    if isinstance(settings, CliSettings):
        return Cli(core=core, basic_argument_parser=basic_argument_parser)
    raise ValueError(f"Unsupported interface type: {settings.type}")
