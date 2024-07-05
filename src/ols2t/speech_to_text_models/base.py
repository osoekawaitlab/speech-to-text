from abc import ABC, abstractmethod
from collections.abc import Generator

from ..models import BaseStream, Segment


class BaseSpeechToTextModel(ABC):
    @abstractmethod
    def transcribe(self, input_stream: BaseStream) -> Generator[Segment, None, None]:
        raise NotImplementedError
