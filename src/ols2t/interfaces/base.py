from abc import ABC, abstractmethod

from ..core import SpeechToTextCore


class BaseInterface(ABC):
    def __init__(self, core: SpeechToTextCore) -> None:
        self._core = core

    @property
    def core(self) -> SpeechToTextCore:
        return self._core

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError
