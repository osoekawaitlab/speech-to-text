import os
from collections.abc import Generator

import pytest

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
