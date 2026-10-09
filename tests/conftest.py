import pytest

from nepkev import FIXTURES
from nepkev.records import load_records
from nepkev.taxonomy import load_taxonomy

FIXTURE_FILE = FIXTURES / "fixtures-v1.canonical.jsonl"


@pytest.fixture(scope="session")
def tax():
    return load_taxonomy()


@pytest.fixture
def fixtures():
    return load_records(FIXTURE_FILE)
