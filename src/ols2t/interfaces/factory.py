from argparse import ArgumentParser

from ..core import SpeechToTextCore
from ..settings import CliInterfaceSettings, InterfaceSettings
from .base import BaseInterface
from .cli import CliInterface


def create_interface(
    settings: InterfaceSettings, core: SpeechToTextCore, basic_argument_parser: ArgumentParser
) -> BaseInterface:
    if isinstance(settings, CliInterfaceSettings):
        return CliInterface(core=core, basic_argument_parser=basic_argument_parser)
    raise ValueError(f"Unsupported interface type: {settings.type}")
