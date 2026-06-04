"""Call language models and return parsed JSON.

This module hides the differences between OpenAI's API and Amazon Bedrock so
the rest of the pipeline can just say "call this model and give me a dict".
The model registry below is the single source of truth for which model keys
are supported, which provider they belong to, and which exact model ID is
called at the wire.

For each call we:
1. Look up the provider in MODEL_REGISTRY.
2. Hand off to a provider-specific function (_call_openai or _call_bedrock).
3. Wrap the call in a small exponential-backoff retry loop for transient
   failures (rate limits, timeouts, etc).
4. Return the parsed JSON response as a Python dict.

JSON shape is one of four fixed schemas indexed by `schema_name`:
"questions", "requirements", "passage", "summary".
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

# Load .env from the project root so OPENAI_API_KEY, AWS_BEARER_TOKEN_BEDROCK,
# and AWS_REGION are available to the SDK clients below.
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
#
# Add a new model here to make it callable. Anything not in this dict will be
# rejected by call_model() with a clear error.

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "gpt-4.1": {
        "provider": "openai",
        "model_id": "gpt-4.1-2025-04-14",
    },
    "llama-4-maverick": {
        # Bedrock on-demand requires the cross-region inference profile ID
        # for this model, not the bare regional model ID.
        "provider": "bedrock",
        "model_id": "us.meta.llama4-maverick-17b-instruct-v1:0",
    },
}


# ---------------------------------------------------------------------------
# Generation parameters (constant across models, per the project spec)
# ---------------------------------------------------------------------------

TEMPERATURE = 1.0
TOP_P = 1.0
# Generous output cap, shared across providers. LLaMA 4 Maverick on Bedrock
# enforces a per-call hard limit of 8,192 output tokens (Bedrock returns a
# ValidationException above that); GPT-4.1 accepts up to 32,768 but 8,192
# is still ~6,000 words — far more than any of our prompts request.
# Truncation matters because Structured Outputs returns a half-emitted JSON
# string that json.loads can't repair, so we prefer a high ceiling and
# detect truncation explicitly (see finish_reason check in _call_openai).
MAX_OUTPUT_TOKENS = 8192

MAX_RETRIES = 3  # i.e. 1 initial attempt + up to 3 retries = 4 attempts total


# ---------------------------------------------------------------------------
# JSON schemas — used directly by OpenAI Structured Outputs, and embedded by
# reference in the Bedrock prompts so the model knows the expected shape.
# ---------------------------------------------------------------------------

def _list_schema(field_name: str) -> dict:
    """Build a schema for `{"<field_name>": ["...", "..."]}`."""
    return {
        "name": f"{field_name}_list",
        "schema": {
            "type": "object",
            "properties": {
                field_name: {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [field_name],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _string_schema(field_name: str) -> dict:
    """Build a schema for `{"<field_name>": "..."}`."""
    return {
        "name": f"{field_name}_object",
        "schema": {
            "type": "object",
            "properties": {
                field_name: {"type": "string"},
            },
            "required": [field_name],
            "additionalProperties": False,
        },
        "strict": True,
    }


JSON_SCHEMAS: dict[str, dict] = {
    "questions": _list_schema("questions"),
    "requirements": _list_schema("requirements"),
    "passage": _string_schema("passage"),
    "summary": _string_schema("summary"),
}


# ---------------------------------------------------------------------------
# Client construction — lazy so importing this module doesn't require keys.
# ---------------------------------------------------------------------------

_openai_client: Any = None
_bedrock_client: Any = None


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is None:
        # Imported lazily so test environments without the package can still
        # import this module to mock the call layer.
        from openai import OpenAI

        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def _get_bedrock_client() -> Any:
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        from botocore.config import Config

        # boto3 picks up AWS_BEARER_TOKEN_BEDROCK and AWS_REGION from the
        # environment automatically.
        # Adaptive retry mode is throttling-aware (jittered backoff that
        # widens when Bedrock pushes back on TPM). The Powers novel is ~158K
        # tokens per call, so we routinely brush against Bedrock's per-account
        # TPM limit; without adaptive retry, sustained throttling kills the
        # run after a few seconds of fixed backoff.
        region = os.environ.get("AWS_REGION", "us-west-2")
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
        )
    return _bedrock_client


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def call_model(
    model_key: str,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
) -> dict:
    """Call the named model and return the parsed JSON response.

    Args:
        model_key: One of the keys in MODEL_REGISTRY.
        system_prompt: Short system-level instruction (e.g. "respond in JSON").
        user_prompt: The fully rendered user-facing prompt.
        schema_name: One of "questions", "requirements", "passage", "summary".

    Returns:
        Parsed JSON as a dict.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model key {model_key!r}; "
            f"choose from {sorted(MODEL_REGISTRY)}"
        )
    if schema_name not in JSON_SCHEMAS:
        raise ValueError(
            f"unknown schema {schema_name!r}; "
            f"choose from {sorted(JSON_SCHEMAS)}"
        )

    entry = MODEL_REGISTRY[model_key]
    provider = entry["provider"]
    model_id = entry["model_id"]

    if provider == "openai":
        return _call_openai(model_id, system_prompt, user_prompt, schema_name)
    elif provider == "bedrock":
        return _call_bedrock(model_id, system_prompt, user_prompt, schema_name)
    else:
        raise ValueError(f"unknown provider {provider!r}")


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _with_retries(
    fn: Callable[[], Any],
    is_transient: Callable[[BaseException], bool],
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call fn(); retry on transient errors with exponential backoff.

    On attempt N (0-indexed) the sleep before the next attempt is 2**N seconds,
    so a sequence of three failures sleeps 1s, 2s, 4s before raising.

    `sleep` is parameterized so tests can pass a no-op or assert call counts.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except BaseException as e:  # noqa: BLE001 — re-raised below
            # Don't retry on programmer errors or anything we don't recognize
            # as transient.
            if not is_transient(e) or attempt == MAX_RETRIES:
                raise
            sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

def _is_openai_transient(exc: BaseException) -> bool:
    """Identify OpenAI errors worth retrying.

    We match on the class name rather than imported types so this module
    works even if the openai package isn't installed in a test environment.
    """
    name = type(exc).__name__
    return name in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    }


def _call_openai(
    model_id: str, system_prompt: str, user_prompt: str, schema_name: str
) -> dict:
    schema = JSON_SCHEMAS[schema_name]

    def attempt() -> dict:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_schema", "json_schema": schema},
        )
        choice = response.choices[0]
        # OpenAI sets finish_reason="length" when max_completion_tokens is hit.
        # Structured Outputs returns a truncated half-string in that case, so
        # json.loads will fail. Raise a clearer error before that happens.
        if choice.finish_reason == "length":
            raise RuntimeError(
                f"OpenAI response truncated at MAX_OUTPUT_TOKENS={MAX_OUTPUT_TOKENS} "
                f"for schema {schema_name!r}. Increase the cap or tighten the prompt."
            )
        # Structured Outputs guarantees valid JSON when not truncated.
        return json.loads(choice.message.content)

    return _with_retries(attempt, _is_openai_transient)


# ---------------------------------------------------------------------------
# Bedrock provider (Converse API)
# ---------------------------------------------------------------------------

def _is_bedrock_transient(exc: BaseException) -> bool:
    """Identify Bedrock errors worth retrying.

    The Converse API raises botocore ClientErrors; we look at the error code.
    Connection errors are also retried.
    """
    name = type(exc).__name__
    if name in {"ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError"}:
        return True
    # ClientError carries a response dict with an Error.Code field.
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "")
        return code in {
            "ThrottlingException",
            "ServiceUnavailableException",
            "ModelTimeoutException",
            "InternalServerException",
        }
    return False


# A relaxed regex used to pull the first JSON object out of any extra text the
# model might wrap around its response (markdown fences, leading "Here is...",
# etc.). Used only as a fallback before raising.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


# Bedrock models occasionally return the bare *value* (a string for `passage`
# and `summary`, a list for `questions` and `requirements`) instead of the
# expected `{"<schema_name>": value}` object. We coerce those into the expected
# shape rather than retrying — the content is correct, only the wrapper is
# missing. For every schema the expected key name equals schema_name itself.

def _ensure_bedrock_shape(parsed: object, schema_name: str) -> dict:
    """Coerce a parsed Bedrock JSON value into the expected single-key dict."""
    if isinstance(parsed, dict) and schema_name in parsed:
        return parsed
    # Bare value matching the expected payload type: wrap it.
    if schema_name in {"passage", "summary"} and isinstance(parsed, str):
        return {schema_name: parsed}
    if schema_name in {"questions", "requirements"} and isinstance(parsed, list):
        return {schema_name: parsed}
    raise ValueError(
        f"Bedrock response did not match expected shape for {schema_name!r}: "
        f"got {type(parsed).__name__}"
    )


def _extract_json(text: str) -> dict:
    """Try to parse `text` as JSON, falling back to extracting a JSON object.

    `strict=False` tells json to accept literal control characters (raw
    newlines, tabs) inside string values. Some Bedrock models (LLaMA in
    particular) emit JSON with embedded newlines in long string fields,
    which strict mode rejects.
    """
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        # Strip common markdown code-fence wrappers.
        stripped = text.strip()
        if stripped.startswith("```"):
            # Drop the first fence line and any trailing fence.
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines)
        # Last resort: regex-extract the first {...} block.
        match = _JSON_OBJECT_RE.search(stripped)
        if not match:
            raise
        return json.loads(match.group(0), strict=False)


def _call_bedrock(
    model_id: str, system_prompt: str, user_prompt: str, schema_name: str
) -> dict:
    # Embed the expected JSON schema in the prompt so the model knows the shape.
    schema = JSON_SCHEMAS[schema_name]["schema"]
    schema_hint = (
        "\n\nReturn JSON matching this schema exactly:\n"
        + json.dumps(schema)
    )

    def one_call(stricter: bool) -> str:
        instruction = system_prompt + schema_hint
        if stricter:
            instruction += (
                "\nReturn ONLY the JSON object — no markdown, no prose, no code fences."
            )
        client = _get_bedrock_client()
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            system=[{"text": instruction}],
            inferenceConfig={
                "temperature": TEMPERATURE,
                "topP": TOP_P,
                "maxTokens": MAX_OUTPUT_TOKENS,
            },
        )
        return response["output"]["message"]["content"][0]["text"]

    def attempt() -> dict:
        # First try with the normal system prompt.
        text = one_call(stricter=False)
        try:
            parsed = _extract_json(text)
            return _ensure_bedrock_shape(parsed, schema_name)
        except (json.JSONDecodeError, ValueError):
            pass

        # One self-correcting retry with a stricter instruction.
        text = one_call(stricter=True)
        try:
            parsed = _extract_json(text)
            return _ensure_bedrock_shape(parsed, schema_name)
        except (json.JSONDecodeError, ValueError):
            pass

        # Last-resort fallback for single-string schemas: some Bedrock models
        # (notably LLaMA on long inputs) ignore the JSON wrapper and emit just
        # the passage / summary content. Accept the raw text rather than
        # crashing the run. List schemas (questions, requirements) cannot be
        # rescued this way and still propagate the JSON failure.
        if schema_name in {"passage", "summary"}:
            return {schema_name: text.strip()}
        raise json.JSONDecodeError(
            f"Bedrock returned unparseable JSON for schema {schema_name!r}",
            text, 0,
        )

    return _with_retries(attempt, _is_bedrock_transient)
