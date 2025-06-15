import time
from base64 import b64decode
from io import BytesIO
from multiprocessing import Queue as MPQueue
from multiprocessing.synchronize import Event as EventClass
from queue import Empty as QueueEmptyException
from typing import Any, List, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

AudioSample: TypeAlias = np.float32


class AudioFrameChunk(NDArray[AudioSample]):
    r"""
    AudioFrameChunk is an array of AudioSample class which compatible with pydantic.

    >>> AudioFrameChunk([1.0, 2.0, 3.0])
    AudioFrameChunk([1., 2., 3.], dtype=float32)
    >>> AudioFrameChunk(b'\x00\x00\x80?\x00\x00\x00@\x00\x00@@')
    AudioFrameChunk([1., 2., 3.], dtype=float32)
    >>> AudioFrameChunk("AACAPwAAAEAAAEBA")
    AudioFrameChunk([1., 2., 3.], dtype=float32)
    >>> from pydantic import BaseModel
    >>> class AudioFrameChunkTester(BaseModel):
    ...   value: AudioFrameChunk
    >>> AudioFrameChunkTester(value=[1.0, 2.0, 3.0]).model_dump()
    {'value': AudioFrameChunk([1., 2., 3.], dtype=float32)}
    >>> AudioFrameChunkTester(value=b'\x00\x00\x80?\x00\x00\x00@\x00\x00@@').model_dump()
    {'value': AudioFrameChunk([1., 2., 3.], dtype=float32)}
    >>> AudioFrameChunkTester(value="AACAPwAAAEAAAEBA").model_dump()
    {'value': AudioFrameChunk([1., 2., 3.], dtype=float32)}
    >>> x = AudioFrameChunkTester(value=[1.0, 2.0, 3.0])
    >>> x.model_dump_json()
    '{"value":[1.0,2.0,3.0]}'
    >>> x.model_validate_json('{"value":[1.0,2.0,3.0]}')
    AudioFrameChunkTester(value=AudioFrameChunk([1., 2., 3.], dtype=float32))
    """  # noqa: E501

    def __new__(cls, value: Any) -> "AudioFrameChunk":
        if isinstance(value, cls):
            return value
        if isinstance(value, np.ndarray):
            return np.array(value, dtype=AudioSample).view(cls)
        if isinstance(value, bytes):
            return np.frombuffer(value, dtype=AudioSample).view(cls)
        if isinstance(value, str):
            return np.frombuffer(b64decode(value), dtype=AudioSample).view(cls)
        return np.array(value, dtype=np.float32).view(cls)

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls.validate,
            serialization=core_schema.plain_serializer_function_ser_schema(cls.serialize, when_used="json"),
        )

    def __get_pydantic_json_schema__(self, _handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        return {"format": "base64EncodedString", "type": "string"}

    @classmethod
    def validate(cls, value: Any) -> NDArray[AudioSample]:
        if isinstance(value, (bytes, str)):
            return cls(value)
        return cls(np.array(value, dtype=np.float32))

    def serialize(self) -> List[float]:
        return cast(List[float], self.tolist())


class ContinuousBufferReader(BytesIO):
    """
    >>> from multiprocessing import Process
    >>> from multiprocessing import Event as MPEvent
    >>> queue = MPQueue(maxsize=256)
    >>> stop_event = MPEvent()
    >>> reader = ContinuousBufferReader(queue, stop_event)
    >>> queue.put(b"1234567890")
    >>> reader.read(10)
    b'1234567890'
    >>> def adding_chunks(queue) -> None:
    ...     for i in range(3):
    ...         time.sleep(0.1)
    ...         queue.put(b"1234567890")
    >>> p = Process(target=adding_chunks, args=(queue,))
    >>> p.start()
    >>> reader.read(13)
    b'1234567890123'
    >>> p.join()
    >>> stop_event.set()
    >>> reader.read(30)
    b'45678901234567890'
    """

    def __init__(self, queue: "MPQueue[bytes]", stop_event: EventClass) -> None:
        super(ContinuousBufferReader, self).__init__(b"")
        self._queue: "MPQueue[bytes]" = queue
        self._stop_event: EventClass = stop_event

    def read(self, size: int | None = -1) -> bytes:
        if size is None or size < 0:
            raise ValueError("size must be positive")
        while self.tell() < size:
            try:
                chunk = self._queue.get(timeout=1.0)
                if chunk is None:
                    time.sleep(0.01)
                self.write(chunk)
            except QueueEmptyException:
                if self._stop_event.is_set():
                    break
                time.sleep(0.01)
        current_buffer = self.getvalue()
        return_value: bytes = current_buffer[:size]
        self.seek(0)
        self.truncate()
        self.write(current_buffer[size:])
        return return_value
