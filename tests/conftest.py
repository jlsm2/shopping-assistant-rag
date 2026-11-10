from collections.abc import Callable

import pytest

from app.assistant import Assistant
from app.catalog import JsonCatalog
from app.config import DEFAULT_DATA_DIR, Store
from app.llm import LLMResponse
from app.retrieval import Index, load_chunks
from tests.fakes import FakeLLM


@pytest.fixture(scope="session")
def index() -> Index:
    return Index(load_chunks(DEFAULT_DATA_DIR / "knowledge"))


@pytest.fixture(scope="session")
def catalog() -> JsonCatalog:
    return JsonCatalog(DEFAULT_DATA_DIR / "catalog.json")


@pytest.fixture(scope="session")
def store() -> Store:
    return Store.load(DEFAULT_DATA_DIR / "store.json")


@pytest.fixture
def make_assistant(index, catalog, store) -> Callable[..., tuple[Assistant, FakeLLM]]:
    def make(*script: LLMResponse | Exception, **overrides) -> tuple[Assistant, FakeLLM]:
        llm = FakeLLM(*script)
        parts = {"index": index, "catalog": catalog, "store": store, **overrides}
        return Assistant(llm=llm, **parts), llm

    return make
