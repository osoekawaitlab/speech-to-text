from collections.abc import Iterable
from contextlib import AbstractContextManager
from enum import Enum
from itertools import count
from types import TracebackType
from typing import Literal, Type, TypeAlias

import numpy as np
from faster_whisper.audio import decode_audio
from numpy.typing import NDArray
from oltl import BaseModel
from pyaudio import PyAudio, paFloat32
from pydantic import FilePath

AudioSample: TypeAlias = np.float32
AudioFrameChunk: TypeAlias = NDArray[AudioSample]


SamplingRate: TypeAlias = int


class AudioChunkStream(Iterable[AudioFrameChunk]):
    """
    A stream of audio chunks.

    Attributes:
        sampling_rate (SamplingRate): The sampling rate of the audio.
        data (Iterable[AudioFrameChunk]): The audio data.

    >>> x = AudioChunkStream(sampling_rate=2, data=iter([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]))
    >>> x.sampling_rate
    2
    >>> x.current_frame
    0
    >>> x.offset
    0.0
    >>> next(x)
    array([1., 2., 3.])
    >>> x.current_frame
    3
    >>> x.offset
    1.5
    >>> next(x)
    array([4., 5., 6.])
    >>> x.current_frame
    6
    >>> x.offset
    3.0
    >>> next(x)
    Traceback (most recent call last):
     ...
    StopIteration
    """

    def __init__(self, sampling_rate: SamplingRate, data: Iterable[AudioFrameChunk]) -> None:
        self._data = iter(data)
        self._sampling_rate = sampling_rate
        self._current_frame = 0
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    @property
    def sampling_rate(self) -> SamplingRate:
        return self._sampling_rate

    @property
    def offset(self) -> float:
        return self.current_frame / self.sampling_rate

    @property
    def current_frame(self) -> int:
        return self._current_frame

    def __next__(self) -> AudioFrameChunk:
        if self._stop:
            raise StopIteration
        for d in self._data:
            self._current_frame += len(d)
            return d
        raise StopIteration

    def __iter__(self) -> "AudioChunkStream":
        return self


class AbstractTranscriber: ...


class StreamType(str, Enum):
    FILE = "FILE"
    MICROPHONE = "MICROPHONE"


class BaseStream(BaseModel, AbstractContextManager[AudioChunkStream]):
    type: StreamType


class FileStream(BaseStream):
    type: Literal[StreamType.FILE] = StreamType.FILE
    path: FilePath

    def __enter__(self) -> AudioChunkStream:
        self._fp = open(self.path, "rb")
        sampling_rate = 16000
        return AudioChunkStream(sampling_rate, iter((decode_audio(self._fp, sampling_rate=sampling_rate),)))

    def __exit__(
        self, exc_type: Type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        self._fp.close()
        return super().__exit__(exc_type, exc_value, traceback)


class MicrophoneStream(BaseStream):
    type: Literal[StreamType.MICROPHONE] = StreamType.MICROPHONE

    def __enter__(self) -> AudioChunkStream:
        sampling_rate = 16000
        self._audio = PyAudio()
        self._stream = self._audio.open(
            format=paFloat32, channels=1, rate=sampling_rate, input=True, frames_per_buffer=1024
        )
        return AudioChunkStream(sampling_rate, (np.frombuffer(self._stream.read(16000), np.float32) for _ in count()))

    def __exit__(
        self, exc_type: Type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
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
