from collections.abc import Generator
from typing import List
from unittest.mock import MagicMock, call

import pytest
from pytest import fixture
from pytest_mock import MockerFixture

from ols2t.models import BaseStream, FileStream, Segment
from ols2t.settings import (
    WhisperSpeechToTextModelLanguage,
    WhisperSpeechToTextModelSize,
)
from ols2t.speech_to_text_models import segment_merging
from ols2t.speech_to_text_models.base import BaseSpeechToTextModel
from ols2t.speech_to_text_models.whisper import WhisperSpeechToTextModel
from ols2t.types import AudioFrameChunk


@fixture
def transcribed_data() -> Generator[Generator[List[Segment], None, None], None, None]:
    def ret() -> Generator[List[Segment], None, None]:
        for sl in [
            [
                Segment(text="ご", start=0.0, end=0.7, probability=0.020458625629544258),
                Segment(text="視", start=0.7, end=0.78, probability=0.9624325037002563),
                Segment(text="聴", start=0.78, end=0.78, probability=0.9999885559082031),
                Segment(text="ありがとうございました", start=0.78, end=0.78, probability=0.9556659460067749),
            ],
            [
                Segment(text="ご", start=0.0, end=0.68, probability=0.010929318144917488),
                Segment(text="視", start=0.68, end=0.82, probability=0.938345730304718),
                Segment(text="聴", start=0.82, end=1.58, probability=0.9999861717224121),
                Segment(text="ありがとうございました", start=1.58, end=1.6, probability=0.9128336310386658),
            ],
            [
                Segment(text="お", start=0.0, end=2.28, probability=0.1288357973098755),
                Segment(text="や", start=2.28, end=2.4, probability=0.4278242588043213),
                Segment(text="す", start=2.4, end=2.4, probability=0.9886988997459412),
                Segment(text="み", start=2.4, end=2.4, probability=0.9997223019599915),
                Segment(text="な", start=2.4, end=2.4, probability=0.9930731654167175),
                Segment(text="さい", start=2.4, end=2.4, probability=0.9951319694519043),
            ],
            [Segment(text="こんにちは", start=0.0, end=1.78, probability=0.9733014702796936)],
            [
                Segment(text="こんにちは。", start=0.0, end=1.04, probability=0.9457034468650818),
                Segment(text="これは", start=1.42, end=1.94, probability=0.917963981628418),
                Segment(text="テ", start=1.94, end=2.2, probability=0.95941561460495),
                Segment(text="スト。", start=2.2, end=2.36, probability=0.9912213087081909),
            ],
            [
                Segment(text="こんにちは。", start=0.0, end=0.3, probability=0.9215718507766724),
                Segment(text="これは", start=0.76, end=1.16, probability=0.987838625907898),
                Segment(text="テ", start=1.16, end=1.36, probability=0.9989131689071655),
                Segment(text="スト", start=1.36, end=1.58, probability=0.9999743700027466),
                Segment(text="用", start=1.58, end=1.78, probability=0.9995704293251038),
                Segment(text="の", start=1.78, end=1.96, probability=0.9997976422309875),
                Segment(text="音", start=1.96, end=2.1, probability=0.9989535808563232),
                Segment(text="声", start=2.1, end=2.34, probability=0.9997398257255554),
                Segment(text="です。", start=2.34, end=2.38, probability=0.636759340763092),
            ],
            [
                Segment(text="これは", start=0.0, end=0.4, probability=0.9834764003753662),
                Segment(text="テ", start=0.4, end=0.58, probability=0.9702237844467163),
                Segment(text="スト", start=0.58, end=0.76, probability=0.9999737739562988),
                Segment(text="用", start=0.76, end=0.98, probability=0.9993946552276611),
                Segment(text="の", start=0.98, end=1.16, probability=0.9991241097450256),
                Segment(text="音", start=1.16, end=1.28, probability=0.9982801675796509),
                Segment(text="声", start=1.28, end=1.54, probability=0.9995442032814026),
                Segment(text="デ", start=1.54, end=1.66, probability=0.9992497563362122),
                Segment(text="ー", start=1.66, end=1.78, probability=0.9999369382858276),
                Segment(text="タ", start=1.78, end=1.96, probability=0.9999675750732422),
                Segment(text="です。", start=1.96, end=2.08, probability=0.998569130897522),
            ],
            [
                Segment(text="音", start=0.0, end=0.24, probability=0.4136035442352295),
                Segment(text="声", start=0.24, end=0.72, probability=0.9855781197547913),
                Segment(text="デ", start=0.72, end=0.86, probability=0.9894005656242371),
                Segment(text="ー", start=0.86, end=1.0, probability=0.9996438026428223),
                Segment(text="タ", start=1.0, end=1.14, probability=0.9996758699417114),
                Segment(text="です", start=1.14, end=1.26, probability=0.9705048203468323),
            ],
        ]:
            yield sl

    yield ret()


@fixture
def speech_to_text_model_mock(
    transcribed_data: Generator[List[Segment], None, None],
) -> Generator[MagicMock, None, None]:
    mock = MagicMock(spec=BaseSpeechToTextModel)
    mock.transcribe.side_effect = transcribed_data
    yield mock


@fixture
def dummy_input_segments() -> Generator[List[MagicMock], None, None]:
    def gen_mock() -> MagicMock:
        m = MagicMock(spec=AudioFrameChunk)
        m.__len__.return_value = 12971
        return m

    fragments = [gen_mock() for _ in range(8)]
    yield fragments


@fixture
def dummy_input_stream(dummy_input_segments: List[MagicMock]) -> Generator[MagicMock, None, None]:
    mock = MagicMock(spec=BaseStream)
    mock.__enter__.return_value.__iter__.return_value = iter(dummy_input_segments)
    mock.__enter__.return_value.sampling_rate = 16000
    yield mock


@fixture
def audio_frame_stream_values() -> Generator[List[MagicMock], None, None]:
    yield [MagicMock(spec=AudioFrameChunk) for _ in range(8)]


def test_segment_merging_transcribe(
    mocker: MockerFixture,
    speech_to_text_model_mock: MagicMock,
    dummy_input_stream: MagicMock,
    dummy_input_segments: List[MagicMock],
    audio_frame_stream_values: List[MagicMock],
) -> None:
    AudioFrameStream = mocker.patch("ols2t.speech_to_text_models.segment_merging.AudioFrameStream")
    AudioFrameStream.side_effect = audio_frame_stream_values
    sut = segment_merging.SegmentMergingSpeechToTextModel(model=speech_to_text_model_mock)
    expected = [
        Segment(text="視", start=0.68, end=0.82, probability=0.938345730304718),
        Segment(text="聴", start=0.82, end=1.58, probability=0.9999861717224121),
        Segment(text="ありがとうございました", start=1.58, end=1.6, probability=0.9128336310386658),
        Segment(text="こんにちは", start=0.8106875, end=2.5906875, probability=0.9733014702796936),
        Segment(text="これは", start=3.041375, end=3.561375, probability=0.917963981628418),
        Segment(text="テ", start=3.561375, end=3.821375, probability=0.95941561460495),
        Segment(text="スト", start=3.82275, end=4.00275, probability=0.9999737739562988),
        Segment(text="用", start=4.00275, end=4.22275, probability=0.9993946552276611),
        Segment(text="の", start=4.22275, end=4.40275, probability=0.9991241097450256),
        Segment(text="音", start=4.40275, end=4.52275, probability=0.9982801675796509),
        Segment(text="声", start=4.52275, end=4.78275, probability=0.9995442032814026),
        Segment(text="デ", start=4.78275, end=4.90275, probability=0.9992497563362122),
        Segment(text="ー", start=4.90275, end=5.02275, probability=0.9999369382858276),
        Segment(text="タ", start=5.02275, end=5.20275, probability=0.9999675750732422),
        Segment(text="です。", start=5.20275, end=5.32275, probability=0.998569130897522),
    ]

    actual = list(sut.transcribe(input_stream=dummy_input_stream))
    assert actual == expected
    assert speech_to_text_model_mock.transcribe.call_count == 8
    speech_to_text_model_mock.transcribe.assert_has_calls([call(v) for v in audio_frame_stream_values])
    AudioFrameStream.assert_has_calls(
        [
            call(chunks=[dummy_input_segments[0]], sampling_rate=16000),
            call(chunks=[dummy_input_segments[0], dummy_input_segments[1]], sampling_rate=16000),
            call(
                chunks=[dummy_input_segments[0], dummy_input_segments[1], dummy_input_segments[2]], sampling_rate=16000
            ),
            call(
                chunks=[dummy_input_segments[1], dummy_input_segments[2], dummy_input_segments[3]], sampling_rate=16000
            ),
            call(
                chunks=[dummy_input_segments[2], dummy_input_segments[3], dummy_input_segments[4]], sampling_rate=16000
            ),
            call(
                chunks=[dummy_input_segments[3], dummy_input_segments[4], dummy_input_segments[5]], sampling_rate=16000
            ),
            call(
                chunks=[dummy_input_segments[4], dummy_input_segments[5], dummy_input_segments[6]], sampling_rate=16000
            ),
            call(
                chunks=[dummy_input_segments[5], dummy_input_segments[6], dummy_input_segments[7]], sampling_rate=16000
            ),
        ]
    )


@pytest.mark.parametrize(
    ("segments", "expected"),
    [
        [
            [
                Segment(text="a", start=0.0, end=1.0, probability=0.9),
                Segment(text="b", start=1.0, end=2.0, probability=0.9),
                Segment(text="c", start=2.0, end=3.0, probability=0.9),
            ],
            [
                Segment(text="a", start=0.0, end=1.0, probability=0.9),
                Segment(text="b", start=1.0, end=2.0, probability=0.9),
                Segment(text="c", start=2.0, end=3.0, probability=0.9),
            ],
        ],
        [
            [
                Segment(text="a", start=0.0, end=1.0, probability=0.9),
                Segment(text="b", start=1.0, end=2.0, probability=0.9),
                Segment(text="c", start=2.0, end=3.0, probability=0.9),
                Segment(text="d", start=0.0, end=2.0, probability=0.99),
            ],
            [
                Segment(text="d", start=0.0, end=2.0, probability=0.99),
                Segment(text="c", start=2.0, end=3.0, probability=0.9),
            ],
        ],
        [[], []],
    ],
)
def test_merge_segments(segments: List[Segment], expected: List[Segment]) -> None:
    sut = segment_merging.SegmentMergingSpeechToTextModel(model=MagicMock(spec=BaseSpeechToTextModel))
    actual = sut.merge_segments(segments)
    assert actual == expected


@pytest.mark.slow
def test_segment_merging_speech_to_text_model_transcribe(hello_fixture: FileStream) -> None:
    model = segment_merging.SegmentMergingSpeechToTextModel(
        model=WhisperSpeechToTextModel(
            path_or_model_size=WhisperSpeechToTextModelSize.SMALL, language=WhisperSpeechToTextModelLanguage.JA
        )
    )
    expected = [Segment(start=0.0, end=1.04, text="こんにちは", probability=0.96)]
    actual = model.transcribe(input_stream=hello_fixture)
    cnt = 0
    for a, e in zip(actual, expected):
        assert abs(a.start - e.start) < 0.01
        assert abs(a.end - e.end) < 0.01
        assert abs(a.probability - e.probability) < 0.1
        assert a.text == e.text
        cnt += 1
    assert cnt == 1
