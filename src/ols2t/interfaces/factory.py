from ..core import SpeechToTextCore
from ..settings import InterfaceSettings
from .base import BaseInterface


def create_interface(settings: InterfaceSettings, core: SpeechToTextCore) -> BaseInterface:
    raise NotImplementedError()
