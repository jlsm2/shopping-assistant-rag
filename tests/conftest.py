import os
from collections.abc import Callable

# Tests must not read the developer's .env. Set before anything imports app.api,
# which builds its settings at import time.
os.environ["ENV_FILE"] = os.devnull

import pytest  # noqa: E402

from app.assistant import Assistant  # noqa: E402
from app.catalog import JsonCatalog  # noqa: E402
from app.config import DEFAULT_DATA_DIR, Store  # noqa: E402
from app.llm import LLMResponse  # noqa: E402
from app.retrieval import Index, load_chunks  # noqa: E402
from tests.fakes import FakeLLM  # noqa: E402


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
