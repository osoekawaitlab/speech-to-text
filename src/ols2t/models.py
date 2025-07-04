from collections.abc import Generator, Iterable
from contextlib import AbstractContextManager
from enum import Enum
from multiprocessing import Event as MPEvent
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from multiprocessing.synchronize import Event as EventClass
from queue import Empty as QueueEmptyException
from types import TracebackType
from typing import Literal, Type, TypeAlias

import numpy as np
from av import container as av_container
from faster_whisper.audio import decode_audio
from oltl import BaseModel
from pyaudio import PyAudio, paFloat32
from pydantic import FilePath

from .types import AudioFrameChunk, ContinuousBufferReader

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
    AUDIO_FRAME = "AUDIO_FRAME"
    BYTES_CHUNK = "BYTES_CHUNK"


class BaseStream(BaseModel, AbstractContextManager[AudioChunkStream]):
    type: StreamType


class AudioFrameStream(BaseStream):
    type: Literal[StreamType.AUDIO_FRAME] = StreamType.AUDIO_FRAME
    chunks: Iterable[AudioFrameChunk]
    sampling_rate: SamplingRate

    def __enter__(self) -> AudioChunkStream:
        return AudioChunkStream(
            sampling_rate=self.sampling_rate, data=iter((np.concatenate(list(self.chunks)).view(AudioFrameChunk),))
        )

    def __exit__(
        self, exc_type: Type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        return super().__exit__(exc_type, exc_value, traceback)


class FileStream(BaseStream):
    type: Literal[StreamType.FILE] = StreamType.FILE
    path: FilePath

    def __enter__(self) -> AudioChunkStream:
        self._fp = open(self.path, "rb")
        sampling_rate = 16000
        return AudioChunkStream(
            sampling_rate, iter((AudioFrameChunk(decode_audio(self._fp, sampling_rate=sampling_rate)),))
        )

    def __exit__(
        self, exc_type: Type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        self._fp.close()
        return super().__exit__(exc_type, exc_value, traceback)


def recording_process(queue: "MPQueue[AudioFrameChunk | Exception | None]", stop_event: EventClass) -> None:
    try:
        audio = PyAudio()
        sampling_rate = 16000
        stream = audio.open(format=paFloat32, channels=1, rate=sampling_rate, input=True, frames_per_buffer=1024)

        while not stop_event.is_set():
            if stream.is_active():
                data = stream.read(16000, exception_on_overflow=False)
                queue.put(AudioFrameChunk(data), block=False)
            else:
                break

    except Exception as e:
        queue.put(Exception(f"Recording process error: {str(e)}"))
    finally:
        if "stream" in locals() and stream:
            stream.stop_stream()
            stream.close()
        if "audio" in locals() and audio:
            audio.terminate()

        queue.put(None)


class MicrophoneStream(BaseStream):
    type: Literal[StreamType.MICROPHONE] = StreamType.MICROPHONE

    def __init__(self, type: StreamType = StreamType.MICROPHONE) -> None:
        super(MicrophoneStream, self).__init__(type=type)
        self._max_queue_size = 256
        self._process: Process | None = None
        self._queue: "MPQueue[AudioFrameChunk | Exception | None]" | None = None
        self._stop_event: EventClass | None = None

    def __enter__(self) -> AudioChunkStream:
        self._queue = MPQueue(maxsize=self._max_queue_size)
        self._stop_event = MPEvent()

        self._process = Process(target=recording_process, args=(self._queue, self._stop_event))
        self._process.daemon = True
        self._process.start()

        sampling_rate = 16000
        return AudioChunkStream(sampling_rate, self._iter_chunks())

    def _iter_chunks(self) -> Generator[AudioFrameChunk, None, None]:
        if self._queue is None:
            raise RuntimeError("Queue is not initialized. Did you call __enter__?")
        if self._stop_event is None:
            raise RuntimeError("Stop event is not initialized. Did you call __enter__?")
        if self._process is None:
            raise RuntimeError("Process is not initialized. Did you call __enter__?")
        if self._process.is_alive() is False:
            raise RuntimeError("Process is not alive. Did you call __enter__?")
        while True:
            try:
                chunk = self._queue.get(timeout=1.0)
                if chunk is None:
                    break

                if isinstance(chunk, Exception):
                    raise chunk

                yield chunk
            except QueueEmptyException:
                continue
            except Exception:
                break

    def __exit__(
        self, exc_type: Type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        if self._stop_event:
            self._stop_event.set()

        if self._process and self._process.is_alive():
            self._process.join(timeout=0.1)
            if self._process.is_alive():
                self._process.terminate()

        if self._queue:
            self._queue.close()
            self._queue.join_thread()

        if self._process:
            self._process.join()
            self._process.close()

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


def decoding_process(
    queue: "MPQueue[bytes]", output_queue: "MPQueue[AudioFrameChunk | Exception | None]", stop_event: EventClass
) -> None:
    reader = ContinuousBufferReader(queue, stop_event)
    container = av_container.open(reader, "r")
    audio_stream = container.streams.audio[0]
    for packet in container.decode(audio_stream):
        output_queue.put(AudioFrameChunk(packet.to_ndarray()[0]))


class BytesChunkStream(BaseStream):
    type: Literal[StreamType.BYTES_CHUNK] = StreamType.BYTES_CHUNK

    def __init__(
        self, chunk_queue: "MPQueue[bytes]", stop_event: EventClass, type: StreamType = StreamType.BYTES_CHUNK
    ) -> None:
        super(BytesChunkStream, self).__init__(type=type)
        self._max_queue_size = 256
        self._process: Process | None = None
        self._output_queue: "MPQueue[AudioFrameChunk | Exception | None]" | None = None
        self._stop_event = stop_event
        self._chunk_queue: "MPQueue[bytes]" = chunk_queue

    def __enter__(self) -> AudioChunkStream:
        self._output_queue = MPQueue(maxsize=self._max_queue_size)
        self._process = Process(target=decoding_process, args=(self._chunk_queue, self._output_queue, self._stop_event))
        self._process.daemon = True
        self._process.start()

        sampling_rate = 16000
        return AudioChunkStream(sampling_rate, self._iter_chunks())

    def _iter_chunks(self) -> Generator[AudioFrameChunk, None, None]:
        if self._output_queue is None:
            raise RuntimeError("Queue is not initialized. Did you call __enter__?")
        if self._process is None:
            raise RuntimeError("Process is not initialized. Did you call __enter__?")
        if self._process.is_alive() is False:
            raise RuntimeError("Process is not alive. Did you call __enter__?")
        while self._process.is_alive():
            try:
                chunk = self._output_queue.get(timeout=1.0)
                if chunk is None:
                    break

                if isinstance(chunk, Exception):
                    raise chunk

                yield chunk
            except QueueEmptyException:
                continue
            except Exception:
                break

    def __exit__(
        self, exc_type: Type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:

        if self._process and self._process.is_alive():
            self._process.join(timeout=0.1)
            if self._process.is_alive():
                self._process.terminate()

        if self._output_queue:
            self._output_queue.close()
            self._output_queue.join_thread()

        if self._process:
            self._process.join()
            self._process.close()

        return super().__exit__(exc_type, exc_value, traceback)
