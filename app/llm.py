"""Thin wrapper around the Anthropic SDK: plain text, and JSON constrained by a schema.

Every call names its `task` ("recap", "judge", "diagnose", ...). The real client ignores it;
the fake in `fakes.py` uses it to decide what to answer.
"""
from __future__ import annotations

import json
import logging

import anthropic

log = logging.getLogger("mnemo.llm")


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, api_key: str, workspace_id: str = "", max_retries: int = 3, timeout: float = 60.0):
        # A key that isn't scoped to a workspace must name one on every request.
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        # The SDK retries 408/409/429/5xx and connection errors with backoff.
        self.client = anthropic.Anthropic(api_key=api_key or None, max_retries=max_retries, timeout=timeout,
                                          default_headers=headers)

    def _create(self, model: str, system: str, prompt: str, max_tokens: int, effort: str, **extra):
        params = dict(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        output_config = {"effort": effort, **extra.pop("output_config", {})}
        if model.startswith(("claude-sonnet-5-5", "claude-opus-5")):
            # Server-side fallback re-runs a request a safety classifier declines on another model.
            extra["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            extra["extra_body"] = {"fallbacks": "default"}
        try:
            resp = self.client.messages.create(**params, output_config=output_config, **extra)
        except anthropic.APIStatusError as e:
            raise LLMError(f"{model}: {e.status_code} {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"{model}: cannot reach the Anthropic API") from e
        if resp.stop_reason == "refusal":
            raise LLMError(f"{model} declined the request")
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not text:
            raise LLMError(f"{model} returned no text (stop reason {resp.stop_reason})")
        return text

    def text(self, task: str, system: str, prompt: str, model: str, max_tokens: int = 4000,
             effort: str = "low") -> str:
        return self._create(model, system, prompt, max_tokens, effort)

    def json(self, task: str, system: str, prompt: str, schema: dict, model: str, max_tokens: int = 4000,
             effort: str = "low") -> dict:
        text = self._create(model, system, prompt, max_tokens, effort,
                            output_config={"format": {"type": "json_schema", "schema": schema}})
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"{model} returned invalid JSON for {task}") from e


def obj(properties: dict, required: list[str] | None = None) -> dict:
    """A strict JSON-schema object (all properties required unless listed otherwise)."""
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": False}
