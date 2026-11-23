import json
import time
from functools import cache
from types import SimpleNamespace

import click
import httpx2
import llm
from llm.default_plugins.openai_models import (
    AsyncResponses,
    ReasoningEffortEnum,
    Responses,
)
from pydantic import Field, create_model

API_BASE = "https://api.z.ai/api/v1"
# Lists every model ID the key can see, but with no details. Needs a key.
MODEL_IDS_URL = "https://api.z.ai/api/paas/v4/models"
# Describes a handful of models in detail: modalities, reasoning levels.
MODEL_DETAILS_URL = "https://api.z.ai/api/v1/models"
CACHE_TIMEOUT = 3600

# Z.ai does not say which models can produce output matching a JSON schema.
# These ones were tested and ignore the schema. Other models, including any
# that are released in the future, are assumed to support schemas.
MODELS_WITHOUT_SCHEMAS = {
    "glm-4.5",
    "glm-4.5-air",
    "glm-4.6",
    "glm-4.7",
    "glm-5",
    "glm-5-turbo",
    "glm-5.1",
    "glm-5.2",
}

def get_key():
    return llm.get_key("", "zai", "ZAI_API_KEY")


def cache_path():
    return llm.user_dir() / "zai_models.json"


def fetch_models(key):
    "Fetch the list of models from Z.ai, combining both models endpoints"
    response = httpx2.get(
        MODEL_IDS_URL,
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()
    models = {model["id"]: model for model in response.json()["data"]}
    try:
        details_response = httpx2.get(
            MODEL_DETAILS_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
            follow_redirects=True,
        )
        details_response.raise_for_status()
        details = details_response.json()["models"]
    except (httpx2.HTTPError, KeyError, ValueError):
        # The details are a bonus - the list of IDs is enough to work with
        details = []
    for detail in details:
        model = models.setdefault(detail["slug"], {"id": detail["slug"]})
        model["details"] = detail
    return sorted(models.values(), key=lambda model: model["id"])


def get_zai_models(key=None, skip_cache=False):
    """
    Return the list of models, using a copy cached for up to an hour.

    Falls back to a stale cached copy if Z.ai cannot be reached, and to an
    empty list if there is no cached copy either.
    """
    path = cache_path()
    cached = None
    if path.is_file():
        try:
            cached = json.loads(path.read_text())
        except ValueError:
            cached = None
        if (
            cached is not None
            and not skip_cache
            and time.time() - path.stat().st_mtime < CACHE_TIMEOUT
        ):
            return cached
    key = key or get_key()
    if not key:
        return cached or []
    try:
        models = fetch_models(key)
    except (httpx2.HTTPError, KeyError, ValueError):
        if skip_cache:
            raise
        return cached or []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(models, indent=2))
    return models


def supports_vision(model):
    # Z.ai only describes some models. Assume the others support everything.
    details = model.get("details") or {}
    if "input_modalities" in details:
        return "image" in details["input_modalities"]
    return True


def reasoning_efforts(model):
    "List of supported efforts, or None if Z.ai does not say"
    details = model.get("details") or {}
    if "supported_reasoning_levels" not in details:
        return None
    return [level["effort"] for level in details["supported_reasoning_levels"]]


def _effort_description(efforts, default):
    if efforts:
        description = "Reasoning effort, one of: {}".format(", ".join(efforts))
        if default:
            description += " - default is {}".format(default)
        return description
    return (
        "Reasoning effort: none, minimal, low, medium, high, xhigh or max. "
        "Models differ in which of these they accept"
    )


@cache
def build_options(efforts=None, default_effort=None):
    """
    Options class for a model. ``efforts`` is a tuple of the reasoning efforts
    the model supports, () if it has no control over reasoning, or None if
    Z.ai does not say.
    """
    fields = {}
    if efforts is None or efforts:
        fields["reasoning_effort"] = (
            ReasoningEffortEnum | None,
            Field(
                description=_effort_description(efforts, default_effort),
                default=None,
            ),
        )
    return create_model("Options", __base__=ZaiOptions, **fields)


class ZaiOptions(llm.Options):
    temperature: float | None = Field(
        description=(
            "Sampling temperature, between 0.0 and 1.0. Higher values make the "
            "output more random, lower values make it more focused."
        ),
        ge=0,
        le=1,
        default=None,
    )
    top_p: float | None = Field(
        description=(
            "Nucleus sampling: only consider the tokens making up this much of "
            "the probability mass, between 0.01 and 1.0."
        ),
        ge=0.01,
        le=1,
        default=None,
    )
    max_tokens: int | None = Field(
        description=(
            "Maximum number of tokens to generate, including reasoning tokens."
        ),
        ge=1,
        default=None,
    )
    json_object: bool | None = Field(
        description="Output a valid JSON object {...}. Prompt must mention JSON.",
        default=None,
    )


def _error_message(error):
    if isinstance(error, dict):
        return error.get("message") or json.dumps(error)
    return getattr(error, "message", None) or str(error)


def _check_event(event):
    """
    Z.ai reports problems such as a missing permission or an exhausted
    balance inside a stream that has a 200 status code. LLM ignores those
    events, which would otherwise result in an empty response.
    """
    event_type = getattr(event, "type", None)
    if event_type == "response.failed":
        error = getattr(event.response, "error", None)
        raise llm.ModelError(_error_message(error) if error else "Response failed")
    if event_type == "error":
        raise llm.ModelError(_error_message(event))
    if event_type == "response.incomplete":
        # Sent in place of response.completed when max_tokens is reached.
        # Passing it off as that event lets LLM record the usage.
        return SimpleNamespace(type="response.completed", response=event.response)
    return event


def _checked(stream):
    for event in stream:
        yield _check_event(event)


async def _achecked(stream):
    async for event in stream:
        yield _check_event(event)


class _CheckedResponses:
    "Stands in for client.responses, checking the events of streamed responses"

    def __init__(self, responses, async_):
        self._responses = responses
        self._async = async_

    def create(self, **kwargs):
        if not kwargs.get("stream"):
            return self._responses.create(**kwargs)
        if self._async:
            return self._acreate(**kwargs)
        return _checked(self._responses.create(**kwargs))

    async def _acreate(self, **kwargs):
        return _achecked(await self._responses.create(**kwargs))


class _SharedZai:
    needs_key = "zai"
    key_env_var = "ZAI_API_KEY"

    def __init__(self, model_definition):
        model_name = model_definition["id"]
        efforts = reasoning_efforts(model_definition)
        super().__init__(
            model_id="zai/{}".format(model_name),
            model_name=model_name,
            api_base=API_BASE,
            vision=supports_vision(model_definition),
            reasoning=efforts is None or bool(efforts),
            # Z.ai returns reasoning text without a summary being requested
            reasoning_summary=False,
            supports_schema=model_name not in MODELS_WITHOUT_SCHEMAS,
            supports_tools=True,
        )
        self.Options = build_options(
            None if efforts is None else tuple(efforts),
            (model_definition.get("details") or {}).get("default_reasoning_level"),
        )

    def __str__(self):
        return "Z.ai: {}".format(self.model_id)

    def get_client(self, key, *, async_=False):
        client = super().get_client(key, async_=async_)
        return SimpleNamespace(responses=_CheckedResponses(client.responses, async_))


class ZaiResponses(_SharedZai, Responses):
    pass


class ZaiAsyncResponses(_SharedZai, AsyncResponses):
    pass


@llm.hookimpl
def register_models(register):
    # Without a key this is the cached list, or nothing if there is no cache
    for model_definition in get_zai_models():
        register(
            ZaiResponses(model_definition),
            ZaiAsyncResponses(model_definition),
        )


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def zai():
        "Commands relating to the llm-z-ai plugin"

    def require_key(key):
        key = llm.get_key(key or "", "zai", "ZAI_API_KEY")
        if not key:
            raise click.ClickException(
                "No key found - set one with 'llm keys set zai' "
                "or the ZAI_API_KEY environment variable"
            )
        return key

    @zai.command()
    @click.option("--key", help="Z.ai API key")
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    def models(key, json_):
        "List available Z.ai models"
        try:
            definitions = get_zai_models(key=require_key(key))
        except (httpx2.HTTPError, KeyError, ValueError) as ex:
            raise click.ClickException("Could not fetch models: {}".format(ex))
        if json_:
            click.echo(json.dumps(definitions, indent=2))
            return
        for model in definitions:
            click.echo("zai/{}".format(model["id"]))
            details = model.get("details")
            if not details:
                continue
            if details.get("description"):
                click.echo("  {}".format(details["description"]))
            if details.get("input_modalities"):
                click.echo("  Input: {}".format(", ".join(details["input_modalities"])))
            efforts = reasoning_efforts(model)
            if efforts:
                line = "  Reasoning effort: {}".format(", ".join(efforts))
                if details.get("default_reasoning_level"):
                    line += " (default {})".format(details["default_reasoning_level"])
                click.echo(line)
            if details.get("context_window"):
                click.echo(
                    "  Context window: {:,} tokens".format(details["context_window"])
                )

    @zai.command()
    @click.option("--key", help="Z.ai API key")
    def refresh(key):
        "Refresh the cached list of Z.ai models"
        path = cache_path()
        before = set()
        if path.is_file():
            try:
                before = {model["id"] for model in json.loads(path.read_text())}
            except ValueError:
                pass
        try:
            definitions = get_zai_models(key=require_key(key), skip_cache=True)
        except (httpx2.HTTPError, KeyError, ValueError) as ex:
            raise click.ClickException("Could not fetch models: {}".format(ex))
        after = {model["id"] for model in definitions}
        for model_id in sorted(after - before):
            click.echo("Added: zai/{}".format(model_id))
        for model_id in sorted(before - after):
            click.echo("Removed: zai/{}".format(model_id))
        if after == before:
            click.echo("No changes, {} models".format(len(after)))
