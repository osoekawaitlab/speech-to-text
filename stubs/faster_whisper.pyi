from io import BufferedReader
from typing import Any, Iterable, List, NamedTuple, Optional, Tuple, Union

class TranscriptionInfo(NamedTuple):
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]]
    ...

class Word(NamedTuple):
    start: float
    end: float
    word: str
    probability: float

class Segment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]

class WhisperModel:
    def __init__(self, model_size_or_path: str) -> None: ...
    def transcribe(self, stream: BufferedReader) -> Tuple[List[Segment], TranscriptionInfo]: ...
