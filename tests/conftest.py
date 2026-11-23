import os

import pytest

# To record new cassettes:
#   PYTEST_ZAI_API_KEY="$(llm keys get zai)" uv run pytest --record-mode once
ZAI_API_KEY = os.getenv("PYTEST_ZAI_API_KEY", "zai-test-key")


def drop_cookies(response):
    response["headers"] = {
        name: value
        for name, value in response["headers"].items()
        if name.lower() != "set-cookie"
    }
    return response


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "cookie"],
        "before_record_response": drop_cookies,
        "decode_compressed_response": True,
    }


@pytest.fixture
def user_path(tmp_path):
    path = tmp_path / "llm.datasette.io"
    path.mkdir()
    return path


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, user_path):
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    monkeypatch.setenv("ZAI_API_KEY", ZAI_API_KEY)
