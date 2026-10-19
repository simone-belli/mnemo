"""The Anthropic wrapper's request shape, checked against a stub client (no network)."""
from types import SimpleNamespace

import pytest

from app.llm import LLM, LLMError, obj


class StubMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=self.text)])


def make(text, stop_reason="end_turn"):
    llm = LLM("sk-test")
    llm.client = SimpleNamespace(messages=StubMessages(text, stop_reason))
    return llm, llm.client.messages


def test_text_on_recap_model_uses_server_side_fallback():
    llm, msgs = make("A recap.")
    assert llm.text("recap", "sys", "prompt", "claude-sonnet-5-5") == "A recap."
    kw = msgs.calls[0]
    assert kw["model"] == "claude-sonnet-5-5" and kw["system"] == "sys"
    assert kw["messages"] == [{"role": "user", "content": "prompt"}]
    assert kw["output_config"] == {"effort": "low"}
    assert kw["extra_body"] == {"fallbacks": "default"}
    assert kw["extra_headers"]["anthropic-beta"] == "server-side-fallback-2026-07-01"


def test_json_on_fast_model_uses_schema_and_no_fallback():
    llm, msgs = make('{"a": 1}')
    schema = obj({"a": {"type": "integer"}})
    assert llm.json("judge", "sys", "prompt", schema, "claude-haiku-5-5") == {"a": 1}
    kw = msgs.calls[0]
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": schema}
    assert "extra_body" not in kw and "extra_headers" not in kw
    assert schema["additionalProperties"] is False and schema["required"] == ["a"]


def test_refusal_and_bad_json_raise():
    llm, _ = make("nope", stop_reason="refusal")
    with pytest.raises(LLMError):
        llm.text("recap", "s", "p", "claude-sonnet-5-5")
    llm, _ = make("not json")
    with pytest.raises(LLMError):
        llm.json("judge", "s", "p", obj({}), "claude-haiku-5-5")
