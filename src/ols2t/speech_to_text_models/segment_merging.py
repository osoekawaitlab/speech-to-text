from collections.abc import Generator
from typing import List

from ..models import AudioFrameStream, BaseStream, Segment
from ..types import AudioFrameChunk
from .base import BaseSpeechToTextModel


class SegmentMergingSpeechToTextModel(BaseSpeechToTextModel):
    def __init__(self, model: BaseSpeechToTextModel):
        super(SegmentMergingSpeechToTextModel, self).__init__()
        self._model = model
        self._buffer_length = 3
        self._margin = 0.5
        self._probability_threshold = 0.2

    @property
    def model(self) -> BaseSpeechToTextModel:
        return self._model

    @property
    def buffer_length(self) -> int:
        return self._buffer_length

    @property
    def margin(self) -> float:
        return self._margin

    @property
    def probability_threshold(self) -> float:
        return self._probability_threshold

    def transcribe(self, input_stream: BaseStream) -> Generator[Segment, None, None]:
        chunk_buffer: List[AudioFrameChunk] = []
        segment_buffer: List[Segment] = []
        offset = 0.0
        with input_stream as chunks:
            for chunk in chunks:
                chunk_buffer.append(chunk)
                if len(chunk_buffer) > self.buffer_length:
                    x = chunk_buffer.pop(0)
                    offset += len(x) / chunks.sampling_rate
                for segment in self.model.transcribe(
                    AudioFrameStream(chunks=chunk_buffer.copy(), sampling_rate=chunks.sampling_rate)
                ):
                    if segment.probability < self.probability_threshold:
                        continue
                    segment_buffer.append(
                        Segment(
                            start=segment.start + offset,
                            end=segment.end + offset,
                            probability=segment.probability,
                            text=segment.text,
                        )
                    )
                current_best = self.merge_segments(segment_buffer)
                for segment in current_best:
                    if segment.end < offset - self.margin:
                        yield segment
                    else:
                        break
                segment_buffer = [segment for segment in segment_buffer if segment.end >= offset - self.margin]
        for segment in self.merge_segments(segment_buffer):
            yield segment

    def compute_segment_weight(self, segment: Segment) -> float:
        return (segment.end - segment.start) * segment.probability

    def merge_segments(self, segments: List[Segment]) -> List[Segment]:
        segments.sort(key=lambda x: x.end)
        bests = [self.compute_segment_weight(segments[0])]
        selected_segments = [True]
        last_indices = [-1]
        for i in range(1, len(segments)):
            last_index = i - 1
            while last_index >= 0 and segments[last_index].end > segments[i].start:
                last_index -= 1
            last_indices.append(last_index)
            if last_index == -1:
                include_current = self.compute_segment_weight(segments[i])
            else:
                include_current = bests[last_index] + self.compute_segment_weight(segments[i])
            exclude_current = bests[i - 1]
            if include_current > exclude_current:
                bests.append(include_current)
                selected_segments.append(True)
            else:
                bests.append(exclude_current)
                selected_segments.append(False)
        selected_indices = []
        i = len(segments) - 1
        while i >= 0:
            if selected_segments[i]:
                selected_indices.append(i)
                i = last_indices[i]
            else:
                i -= 1
        return [segments[idx] for idx in reversed(selected_indices)]
