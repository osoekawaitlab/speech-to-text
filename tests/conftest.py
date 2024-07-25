import os
from collections.abc import Generator, Iterable

import numpy as np
import pytest
from pytest import Config
from pytest_mock import MockerFixture

from ols2t.models import FileStream


@pytest.fixture
def test_dir() -> Generator[str, None, None]:
    yield os.path.abspath(os.path.dirname(__file__))


@pytest.fixture
def fixture_dir(test_dir: str) -> Generator[str, None, None]:
    yield os.path.join(test_dir, "fixtures")


@pytest.fixture
def hello_fixture(fixture_dir: str) -> Generator[FileStream, None, None]:
    yield FileStream(path=os.path.join(fixture_dir, "hello_ja.wav"))


@pytest.fixture
def patch_empty_environment_variables(mocker: MockerFixture) -> Generator[None, None, None]:
    current_environment_variables = os.environ.copy()
    mocker.patch("os.environ", {k: v for k, v in current_environment_variables.items() if not k.startswith("OLS2T_")})
    yield


@pytest.fixture
def longtext_all_decoded_fixture_path(fixture_dir: str) -> Generator[str, None, None]:
    yield os.path.join(fixture_dir, "longtext_all_decoded.npy")


@pytest.fixture
def longtext_all_decoded_one_second_chunks(
    longtext_all_decoded_fixture_path: str,
) -> Generator[Iterable[bytes], None, None]:
    with open(longtext_all_decoded_fixture_path, "rb") as f:
        x = np.load(f)
        yield (x[i : i + 16000].tobytes() for i in range(0, len(x), 16000))


@pytest.fixture
def patch_whisper_small_environment_variable(
    patch_empty_environment_variables: None, mocker: MockerFixture
) -> Generator[None, None, None]:
    mocker.patch(
        "os.environ",
        dict(
            os.environ,
            OLS2T_CORE_SETTINGS__SPEECH_TO_TEXT_MODEL_SETTINGS__TYPE="WHISPER",
            OLS2T_CORE_SETTINGS__SPEECH_TO_TEXT_MODEL_SETTINGS__PATH_OR_MODEL_SIZE="small",
            OLS2T_CORE_SETTINGS__SPEECH_TO_TEXT_MODEL_SETTINGS__LANGUAGE="ja",
        ),
    )
    yield


@pytest.fixture
def patch_cli_environment_variable(
    patch_empty_environment_variables: None, mocker: MockerFixture
) -> Generator[None, None, None]:
    mocker.patch("os.environ", dict(os.environ, OLS2T_INTERFACE_SETTINGS__TYPE="CLI"))
    yield


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow")
