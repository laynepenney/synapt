"""TDD spec: ModelClient.chat() additive OSS usage_out seam.

Design decision (2026-07-16): the underlying vLLM/MLX calls already compute
real token counts internally and simply discard them at the point ``chat()``
extracts only the completion text. Extend ``chat()`` additively — an optional
``usage_out: dict | None = None`` param each backend populates with
``{"tokens_in": int|None, "tokens_out": int|None, "cached_tokens": int|None}``
when the underlying call reports them. Non-breaking: existing callers that never
pass ``usage_out`` see zero behavior change.

``cached_tokens`` is structurally ``None`` for both backends in this contract.
Neither backend reports a cache count on the runtime objects read here. Never
estimate it. ``None`` is the honest value.
"""

from __future__ import annotations

import inspect

import pytest

from synapt._models.base import ModelClient, Message


def test_base_chat_signature_accepts_optional_usage_out():
    sig = inspect.signature(ModelClient.chat)
    assert "usage_out" in sig.parameters
    assert sig.parameters["usage_out"].default is None


def test_vllm_chat_populates_usage_out_from_the_real_request_output(monkeypatch):
    pytest.importorskip("vllm")  # Modal-cloud-only dependency, not installed in every dev venv
    from synapt._models.vllm_client import VLLMClient

    class FakeCompletionOutput:
        text = "hello world"
        token_ids = [1, 2, 3, 4]  # 4 real output tokens

    class FakeRequestOutput:
        prompt_token_ids = [10, 11, 12]  # 3 real input tokens
        outputs = [FakeCompletionOutput()]

    class FakeEngine:
        def chat(self, chat_messages, params):
            return [FakeRequestOutput()]

    client = VLLMClient.__new__(VLLMClient)  # bypass real engine init
    client.max_tokens = 512
    client._model_name = None
    monkeypatch.setattr(client, "_get_engine", lambda model: FakeEngine())

    usage_out: dict = {}
    text = client.chat(
        model="qwen3.5-4b",
        messages=[Message(role="user", content="hi")],
        usage_out=usage_out,
    )

    assert text == "hello world"
    assert usage_out["tokens_in"] == 3
    assert usage_out["tokens_out"] == 4
    assert usage_out["cached_tokens"] is None  # structurally null, no cache tier locally


def test_vllm_chat_with_no_usage_out_is_unchanged(monkeypatch):
    """The default (non-metering) call path must behave exactly as before —
    zero side effects, no new required argument."""
    pytest.importorskip("vllm")  # Modal-cloud-only dependency, not installed in every dev venv
    from synapt._models.vllm_client import VLLMClient

    class FakeCompletionOutput:
        text = "hello world"
        token_ids = [1, 2, 3]

    class FakeRequestOutput:
        prompt_token_ids = [10, 11]
        outputs = [FakeCompletionOutput()]

    class FakeEngine:
        def chat(self, chat_messages, params):
            return [FakeRequestOutput()]

    client = VLLMClient.__new__(VLLMClient)
    client.max_tokens = 512
    client._model_name = None
    monkeypatch.setattr(client, "_get_engine", lambda model: FakeEngine())

    text = client.chat(model="qwen3.5-4b", messages=[Message(role="user", content="hi")])
    assert text == "hello world"


def test_mlx_chat_populates_usage_out_from_the_real_stream_response(monkeypatch):
    from synapt._models.mlx_client import MLXClient

    class FakeResponse:
        text = "hello"
        prompt_tokens = 22
        generation_tokens = 9

    def fake_stream_generate(model_obj, tokenizer, prompt, **kwargs):
        yield FakeResponse()

    import importlib

    # mlx_lm/__init__.py does `from .generate import generate`, which overwrites the
    # `generate` ATTRIBUTE on the mlx_lm package object with the function — `import
    # mlx_lm.generate as x` resolves via that (shadowed) attribute access and silently
    # binds the function, not the module. importlib.import_module reads sys.modules
    # directly, bypassing the shadowed attribute.
    mlx_generate_module = importlib.import_module("mlx_lm.generate")

    client = MLXClient.__new__(MLXClient)
    client.options = type(
        "Opts", (), {"top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 256},
    )()
    monkeypatch.setattr(client, "_load", lambda model, adapter_path=None: (object(), object()))
    monkeypatch.setattr(client, "_format_messages", lambda messages, tokenizer: "prompt")
    monkeypatch.setattr(mlx_generate_module, "stream_generate", fake_stream_generate)

    usage_out: dict = {}
    text = client.chat(
        model="mlx-community/Ministral-3-3B-Instruct-2512-4bit",
        messages=[Message(role="user", content="hi")],
        usage_out=usage_out,
    )

    assert text == "hello"
    assert usage_out["tokens_in"] == 22
    assert usage_out["tokens_out"] == 9
    assert usage_out["cached_tokens"] is None  # structurally null, no cache tier locally


def test_mlx_chat_with_no_usage_out_is_unchanged(monkeypatch):
    from synapt._models.mlx_client import MLXClient

    class FakeResponse:
        text = "hello"
        prompt_tokens = 22
        generation_tokens = 9

    def fake_stream_generate(model_obj, tokenizer, prompt, **kwargs):
        yield FakeResponse()

    import importlib

    mlx_generate_module = importlib.import_module("mlx_lm.generate")
    client = MLXClient.__new__(MLXClient)
    client.options = type(
        "Opts", (), {"top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 256},
    )()
    monkeypatch.setattr(client, "_load", lambda model, adapter_path=None: (object(), object()))
    monkeypatch.setattr(client, "_format_messages", lambda messages, tokenizer: "prompt")
    monkeypatch.setattr(mlx_generate_module, "stream_generate", fake_stream_generate)

    assert client.chat(
        model="mlx-community/Ministral-3-3B-Instruct-2512-4bit",
        messages=[Message(role="user", content="hi")],
    ) == "hello"
