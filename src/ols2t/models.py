from contextlib import AbstractContextManager
from enum import Enum
from io import BufferedReader
from types import TracebackType
from typing import Literal, Type, TypeAlias

import numpy as np
from numpy.typing import NDArray
from oltl import BaseModel
from pydantic import FilePath

FRAME_ARRAY_T: TypeAlias = NDArray[np.float32]


class AbstractTranscriber: ...


class StreamType(str, Enum):
    FILE = "FILE"


class BaseStream(BaseModel, AbstractContextManager[BufferedReader]):
    type: StreamType


class FileStream(BaseStream):
    type: Literal[StreamType.FILE] = StreamType.FILE
    path: FilePath

    def __enter__(self) -> BufferedReader:
        self._fp = open(self.path, "rb")
        return self._fp

    def __exit__(
        self, exc_type: Type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        self._fp.close()
        return super().__exit__(exc_type, exc_value, traceback)


class Segment(BaseModel):
    """A segment of transcribed text.

    Attributes:
        text (str): The transcribed text.
        start (float): The start time of the segment in seconds.
        end (float): The end time of the segment in seconds.
        probability (float): The probability of the transcribed text.

    >>> Segment(text="こんにちは", start=0.0, end=2.0, probability=0.9)
    Segment(text='こんにちは', start=0.0, end=2.0, probability=0.9)
    """

    text: str
    start: float
    end: float
    probability: float
