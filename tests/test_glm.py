import json
import os
import time
from types import SimpleNamespace

import httpx2
import llm
import llm_glm
import pytest
from click.testing import CliRunner
from llm.cli import cli

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\xa6\x00\x00\x01\x1a"
    b"\x02\x03\x00\x00\x00\xe6\x99\xc4^\x00\x00\x00\tPLTE\xff\xff\xff"
    b"\x00\xff\x00\xfe\x01\x00\x12t\x01J\x00\x00\x00GIDATx\xda\xed\xd81\x11"
    b"\x000\x08\xc0\xc0.]\xea\xaf&Q\x89\x04V\xe0>\xf3+\xc8\x91Z\xf4\xa2\x08EQ\x14E"
    b"Q\x14EQ\x14EQ\xd4B\x91$I3\xbb\xbf\x08EQ\x14EQ\x14EQ\x14E\xd1\xa5"
    b"\xd4\x17\x91\xc6\x95\x05\x15\x0f\x9f\xc5\t\x9f\xa4\x00\x00\x00\x00IEND\xaeB`"
    b"\x82"
)

MODELS = [
    {"id": "glm-5-turbo", "details": {"supported_reasoning_levels": []}},
    {"id": "glm-5.2"},
    {
        "id": "glm-5.3",
        "details": {
            "input_modalities": ["text"],
            "supported_reasoning_levels": [
                {"effort": "low"},
                {"effort": "high"},
                {"effort": "max"},
            ],
            "description": "Flagship model",
            "default_reasoning_level": "max",
            "context_window": 1048576,
        },
    },
    {"id": "glm-5.3-flash", "details": {"input_modalities": ["text", "image"]}},
    {"id": "glm-5.3-flashx"},
]


@pytest.fixture
def fake_models(monkeypatch):
    "Serve MODELS instead of talking to Z.ai, recording each fetch"
    calls = []

    def fetch_models(key):
        calls.append(key)
        return MODELS

    monkeypatch.setattr(llm_glm, "fetch_models", fetch_models)
    return calls


# Tests that talk to Z.ai, replayed from cassettes


@pytest.mark.vcr
@pytest.mark.parametrize("stream", (True, False))
def test_prompt(stream):
    model = llm.get_model("zai-glm/glm-5.3-flash")
    response = model.prompt(
        "Say hi in three words", stream=stream, reasoning_effort="low"
    )
    assert response.text()
    usage = response.usage()
    assert usage.input > 0
    assert usage.output > 0
    assert "reasoning_tokens" in usage.details["output_tokens_details"]
    assert response.response_json["model"] == "glm-5.3-flash"


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (True, False))
async def test_async_prompt(stream):
    model = llm.get_async_model("zai-glm/glm-5.3-flash")
    response = model.prompt(
        "Say hi in three words", stream=stream, reasoning_effort="low"
    )
    assert await response.text()
    usage = await response.usage()
    assert usage.input > 0
    assert usage.output > 0


@pytest.mark.vcr
def test_reasoning_is_streamed():
    model = llm.get_model("zai-glm/glm-5.3-flash")
    response = model.prompt(
        "What is 17 * 23? Reply with just the number.", reasoning_effort="high"
    )
    reasoning = ""
    text = ""
    for event in response.stream_events():
        if event.type == "reasoning":
            reasoning += event.chunk or ""
        elif event.type == "text":
            text += event.chunk
    assert reasoning.strip()
    assert "391" in text


@pytest.mark.vcr
def test_system_prompt_and_conversation():
    model = llm.get_model("zai-glm/glm-5.3-flash")
    conversation = model.conversation()
    first = conversation.prompt(
        "My favourite number is 7919. Just say OK.",
        system="You are terse.",
        reasoning_effort="low",
    )
    assert first.text()
    second = conversation.prompt(
        "What is my favourite number?", reasoning_effort="low"
    )
    assert "7919" in second.text()


@pytest.mark.vcr
def test_tools():
    calls = []

    def get_weather(city: str) -> str:
        "Get the weather for a city"
        calls.append(city)
        return f"Sunny and 21C in {city}"

    model = llm.get_model("zai-glm/glm-5.3-flash")
    chain = model.chain(
        "What is the weather in Paris and in Tokyo? Use the tool.",
        tools=[get_weather],
        options={"reasoning_effort": "low"},
    )
    text = chain.text()
    assert sorted(calls) == ["Paris", "Tokyo"]
    assert "21" in text


@pytest.mark.vcr
def test_image_attachment():
    model = llm.get_model("zai-glm/glm-5.3-flash")
    response = model.prompt(
        "What two colors are in this image? Reply with just the two colors.",
        attachments=[llm.Attachment(content=TINY_PNG)],
        reasoning_effort="low",
    )
    text = response.text().lower()
    assert "red" in text
    assert "green" in text


@pytest.mark.vcr
def test_models_are_fetched_and_cached(user_path):
    model_ids = [model.model_id for model in llm.get_models()]
    assert "zai-glm/glm-5.3" in model_ids
    assert "zai-glm/glm-5.3-flash" in model_ids
    cached = {
        model["id"]: model
        for model in json.loads((user_path / "glm_models.json").read_text())
    }
    assert cached["glm-5.3"]["owned_by"] == "z-ai"
    # Details from the second models endpoint are merged in
    assert cached["glm-5.3"]["details"]["input_modalities"] == ["text"]
    assert llm_glm.reasoning_efforts(cached["glm-5.3"]) == ["low", "high", "max"]
    assert "image" in cached["glm-5.3-flash"]["details"]["input_modalities"]
    # glm-5.2 is only known to the endpoint that lists IDs
    assert "details" not in cached["glm-5.2"]
    assert llm_glm.reasoning_efforts(cached["glm-5.2"]) is None


@pytest.mark.vcr
def test_models_and_refresh_commands():
    runner = CliRunner()
    result = runner.invoke(cli, ["glm", "refresh"])
    assert result.exit_code == 0, result.output
    assert "Added: zai-glm/glm-5.3\n" in result.output
    result = runner.invoke(cli, ["glm", "refresh"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("No changes, ")


@pytest.mark.vcr
def test_models_command():
    result = CliRunner().invoke(cli, ["glm", "models", "--json"])
    assert result.exit_code == 0, result.output
    # Pretty printed JSON
    assert result.output.startswith('[\n  {\n    "id": "glm-4.5",\n')
    models = {model["id"]: model for model in json.loads(result.output)}
    # Every model is listed, with details for the ones Z.ai describes
    assert len(models) > 4
    assert models["glm-5.3"]["details"]["input_modalities"] == ["text"]
    assert "details" not in models["glm-5.2"]


def test_models_command_text(fake_models):
    result = CliRunner().invoke(cli, ["glm", "models"])
    assert result.exit_code == 0, result.output
    assert result.output == (
        "zai-glm/glm-5-turbo\n"
        "zai-glm/glm-5.2\n"
        "zai-glm/glm-5.3\n"
        "  Flagship model\n"
        "  Input: text\n"
        "  Reasoning effort: low, high, max (default max)\n"
        "  Context window: 1,048,576 tokens\n"
        "zai-glm/glm-5.3-flash\n"
        "  Input: text, image\n"
        "zai-glm/glm-5.3-flashx\n"
    )


def test_models_command_uses_cache(fake_models):
    runner = CliRunner()
    for _ in range(2):
        result = runner.invoke(cli, ["glm", "models", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == MODELS
    assert len(fake_models) == 1


@pytest.mark.vcr
@pytest.mark.parametrize("model_id", ("zai-glm/glm-5.3", "zai-glm/glm-5.3-flash"))
def test_schema(model_id):
    model = llm.get_model(model_id)
    response = model.prompt(
        "Invent a dog",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        },
        reasoning_effort="low",
    )
    dog = json.loads(response.text())
    assert set(dog) == {"name", "age"}
    assert isinstance(dog["name"], str)
    assert isinstance(dog["age"], int)


# Tests that do not need the network


def test_models_registered(fake_models):
    models = {
        model.model_id: model
        for model in llm.get_models()
        if model.model_id.startswith("zai-glm/")
    }
    assert sorted(models) == ["zai-glm/" + model["id"] for model in MODELS]
    model = models["zai-glm/glm-5.3"]
    assert str(model) == "Z.ai: zai-glm/glm-5.3"
    assert model.model_name == "glm-5.3"
    assert model.api_base == "https://api.z.ai/api/v1"
    assert model.needs_key == "zai"
    assert model.key_env_var == "ZAI_API_KEY"
    assert model.supports_tools
    assert model.supports_schema
    assert llm.get_async_model("zai-glm/glm-5.3").model_name == "glm-5.3"


def test_vision(fake_models):
    def attachment_types(model_id):
        return llm.get_model(model_id).attachment_types

    # Declared by the models endpoint
    assert "image/png" in attachment_types("zai-glm/glm-5.3-flash")
    assert not attachment_types("zai-glm/glm-5.3")
    # Not described by the endpoint, so assumed to support everything
    assert "image/png" in attachment_types("zai-glm/glm-5.3-flashx")
    assert "image/png" in attachment_types("zai-glm/glm-5.2")


def test_options(fake_models):
    def fields(model_id):
        return llm.get_model(model_id).Options.model_fields

    description = fields("zai-glm/glm-5.3")["reasoning_effort"].description
    assert description == "Reasoning effort, one of: low, high, max - default is max"
    # Unknown reasoning support still offers the option
    assert "reasoning_effort" in fields("zai-glm/glm-5.2")
    # A model that reports no reasoning levels does not
    assert "reasoning_effort" not in fields("zai-glm/glm-5-turbo")
    assert set(fields("zai-glm/glm-5-turbo")) == {
        "temperature",
        "top_p",
        "max_tokens",
        "json_object",
    }
    options = llm.get_model("zai-glm/glm-5.3").Options
    with pytest.raises(ValueError):
        options(temperature=1.5)
    with pytest.raises(ValueError):
        options(reasoning_effort="enormous")


def test_request_shape(fake_models):
    model = llm.get_model("zai-glm/glm-5.3")
    prompt = llm.Prompt(
        "hi",
        model=model,
        system="Be brief",
        options=model.Options(reasoning_effort="high", max_tokens=100, temperature=0.5),
    )
    _, instructions = model._build_responses_input(prompt)
    assert model._finalize_responses_kwargs(prompt, True, instructions) == {
        "max_output_tokens": 100,
        "temperature": 0.5,
        "reasoning": {"effort": "high"},
        "instructions": "Be brief",
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }


def test_without_key_models_come_from_cache(fake_models, monkeypatch, user_path):
    def glm_models():
        return [
            model.model_id
            for model in llm.get_models()
            if model.model_id.startswith("zai-glm/")
        ]

    monkeypatch.delenv("ZAI_API_KEY")
    # No key and no cache: no models, and no attempt to fetch them
    assert glm_models() == []
    assert fake_models == []
    # "llm glm refresh --key" fetches the list using the key it is given
    result = CliRunner().invoke(cli, ["glm", "refresh", "--key", "other-key"])
    assert result.exit_code == 0, result.output
    assert fake_models == ["other-key"]
    assert glm_models() == ["zai-glm/" + model["id"] for model in MODELS]
    # That keeps working once the cache is stale, since it cannot be refreshed
    path = user_path / "glm_models.json"
    old = time.time() - 2 * 3600
    os.utime(path, (old, old))
    assert glm_models() == ["zai-glm/" + model["id"] for model in MODELS]
    assert fake_models == ["other-key"]


def test_model_list_cache(fake_models, user_path):
    path = user_path / "glm_models.json"
    assert llm_glm.get_glm_models() == MODELS
    assert len(fake_models) == 1
    assert json.loads(path.read_text()) == MODELS
    # A fresh cache means no second fetch
    assert llm_glm.get_glm_models() == MODELS
    assert len(fake_models) == 1
    # An expired one is fetched again
    old = time.time() - 2 * 3600
    os.utime(path, (old, old))
    assert llm_glm.get_glm_models() == MODELS
    assert len(fake_models) == 2
    # As is skip_cache=True
    assert llm_glm.get_glm_models(skip_cache=True) == MODELS
    assert len(fake_models) == 3


def test_stale_cache_used_if_fetch_fails(monkeypatch, user_path):
    def fetch_models(key):
        raise httpx2.ConnectError("offline")

    monkeypatch.setattr(llm_glm, "fetch_models", fetch_models)
    # No cache at all: no models, but no error either
    assert llm_glm.get_glm_models() == []
    path = user_path / "glm_models.json"
    path.write_text(json.dumps(MODELS))
    old = time.time() - 2 * 3600
    os.utime(path, (old, old))
    assert llm_glm.get_glm_models() == MODELS
    assert llm.get_model("zai-glm/glm-5.3").model_name == "glm-5.3"
    # An explicit refresh reports the failure
    result = CliRunner().invoke(cli, ["glm", "refresh"])
    assert result.exit_code == 1
    assert "Could not fetch models" in result.output


def test_commands_need_a_key(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY")
    for command in ("refresh", "models"):
        result = CliRunner().invoke(cli, ["glm", command])
        assert result.exit_code == 1
        assert "llm keys set zai" in result.output


def test_schema_support(fake_models):
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    model = llm.get_model("zai-glm/glm-5.3")
    assert model.supports_schema
    # The schema is sent to the API, not described in the prompt
    prompt = llm.Prompt("Invent a dog", model=model, schema=schema, system="Be brief")
    kwargs = model._finalize_responses_kwargs(prompt, True, "Be brief")
    assert kwargs["text"] == {
        "format": {
            "type": "json_schema",
            "name": "output",
            "schema": schema,
            "strict": False,
        }
    }
    assert kwargs["instructions"] == "Be brief"
    # Models that are known to ignore schemas do not offer them
    assert not llm.get_model("zai-glm/glm-5.2").supports_schema
    assert not llm.get_model("zai-glm/glm-5-turbo").supports_schema
    # Models that have not been tested are assumed to support them
    assert llm.get_model("zai-glm/glm-5.3-flashx").supports_schema
    result = CliRunner().invoke(
        cli,
        ["-m", "zai-glm/glm-5.2", "Invent a dog", "--schema", "name, age int", "--no-log"],
    )
    assert result.exit_code == 1
    assert "does not support schemas" in result.output


class FakeResponses:
    def __init__(self, events):
        self.events = events

    def create(self, **kwargs):
        return iter(self.events)


class FakeAsyncResponses(FakeResponses):
    async def create(self, **kwargs):
        async def stream():
            for event in self.events:
                yield event

        return stream()


def failed_events():
    "The two events Z.ai streamed, with a 200 status, for a model I could not use"
    message = "You do not have permission to access glm-5.4[2026112401265]"
    return [
        SimpleNamespace(
            type="response.failed",
            response=SimpleNamespace(
                error=SimpleNamespace(code="permission_denied", message=message),
                output=[],
                status="failed",
            ),
        ),
        SimpleNamespace(type="error", code="permission_denied", message=message),
    ]


def use_fake_client(monkeypatch, responses):
    monkeypatch.setattr(
        llm_glm.Responses,
        "get_client",
        lambda self, key, async_=False: SimpleNamespace(responses=responses),
    )
    monkeypatch.setattr(
        llm_glm.AsyncResponses,
        "get_client",
        lambda self, key, async_=False: SimpleNamespace(responses=responses),
    )


def test_streamed_failure_is_an_error(fake_models, monkeypatch):
    use_fake_client(monkeypatch, FakeResponses(failed_events()))
    with pytest.raises(llm.ModelError) as ex:
        llm.get_model("zai-glm/glm-5.3").prompt("hello").text()
    assert "You do not have permission to access glm-5.4" in str(ex.value)


@pytest.mark.asyncio
async def test_async_streamed_failure_is_an_error(fake_models, monkeypatch):
    use_fake_client(monkeypatch, FakeAsyncResponses(failed_events()))
    with pytest.raises(llm.ModelError) as ex:
        await llm.get_async_model("zai-glm/glm-5.3").prompt("hello").text()
    assert "You do not have permission to access glm-5.4" in str(ex.value)


def test_streamed_error_event_alone_is_an_error(fake_models, monkeypatch):
    use_fake_client(monkeypatch, FakeResponses(failed_events()[1:]))
    with pytest.raises(llm.ModelError) as ex:
        llm.get_model("zai-glm/glm-5.3").prompt("hello").text()
    assert "You do not have permission to access glm-5.4" in str(ex.value)


def test_streamed_failure_in_cli(fake_models, monkeypatch):
    use_fake_client(monkeypatch, FakeResponses(failed_events()))
    result = CliRunner().invoke(cli, ["-m", "zai-glm/glm-5.3", "hello", "--no-log"])
    assert result.exit_code == 1
    assert "You do not have permission to access glm-5.4" in result.output


def test_incomplete_stream_records_usage(fake_models, monkeypatch):
    from openai.types.responses import Response

    # construct() skips validation, as the OpenAI client does for API data
    final = Response.construct(
        **{
            "id": "resp_1",
            "created_at": 1795454007,
            "model": "glm-5.3-flash",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {
                "input_tokens": 19,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 20,
                "output_tokens_details": {"reasoning_tokens": 20},
                "total_tokens": 39,
            },
        }
    )
    events = [SimpleNamespace(type="response.incomplete", response=final)]
    use_fake_client(monkeypatch, FakeResponses(events))
    response = llm.get_model("zai-glm/glm-5.3-flash").prompt("Count", max_tokens=20)
    assert response.text() == ""
    assert response.usage().input == 19
    assert response.usage().output == 20
    assert response.response_json["status"] == "incomplete"
