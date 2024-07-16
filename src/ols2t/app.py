from .core import SpeechToTextCore
from .interfaces.base import BaseInterface
from .interfaces.factory import create_interface
from .settings import SpeechToTextAppSettings


class SpeechToTextApp:
    def __init__(self, core: SpeechToTextCore, interface: BaseInterface):
        self._core = core
        self._interface = interface

    @property
    def core(self) -> SpeechToTextCore:
        return self._core

    @property
    def interface(self) -> BaseInterface:
        return self._interface

    def run(self) -> None:
        self.interface.run()

    @classmethod
    def create(cls, settings: SpeechToTextAppSettings) -> "SpeechToTextApp":
        core = SpeechToTextCore.create(settings=settings.core_settings)
        interface = create_interface(settings=settings.interface_settings, core=core)
        return cls(core=core, interface=interface)
