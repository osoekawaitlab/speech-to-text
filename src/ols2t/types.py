from base64 import b64decode
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
