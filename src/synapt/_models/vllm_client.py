"""vLLM inference client for decoder-only models.

Fast GPU inference for models like Ministral-8B using vLLM's
offline LLM engine. No server process needed.

Requires: pip install vllm
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .base import Message, ModelClient

logger = logging.getLogger(__name__)

_ENGINE_CACHE: dict[str, object] = {}


class VLLMClient(ModelClient):
    """Decoder-only model client via vLLM offline inference."""

    def __init__(self, max_tokens: int = 800, model_name: str | None = None):
        self.max_tokens = max_tokens
        self._model_name = model_name

    def _get_engine(self, model: str):
        if model in _ENGINE_CACHE:
            return _ENGINE_CACHE[model]

        from vllm import LLM

        engine = LLM(model=model, max_model_len=4096, trust_remote_code=True)
        _ENGINE_CACHE[model] = engine
        return engine

    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.2,
        usage_out: Optional[dict] = None,
        **kwargs,
    ) -> str:
        from vllm import SamplingParams

        max_tokens = kwargs.get("max_tokens") or self.max_tokens
        engine = self._get_engine(self._model_name or model)

        # Use vLLM's chat API with the tokenizer's chat template
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]

        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = engine.chat(chat_messages, params)
        if outputs and outputs[0].outputs:
            if usage_out is not None:
                usage_out.update(usage_from_request_output(outputs[0]))
            return outputs[0].outputs[0].text.strip()
        return ""


def usage_from_request_output(request_output) -> dict:
    """Token counts off a vLLM ``RequestOutput``, as a PURE function.

    Split out deliberately.  Both end-to-end vLLM tests are ``importorskip``'d,
    and vllm is a runtime we do not install locally, so on every machine we work
    on those tests are SKIPS -- and a skip is not a green.  ``vllm_client``
    itself imports fine without vllm (every vllm import is call-time), so this
    helper can be executed directly against the same fake objects with no
    runtime present.  That is the only way this arithmetic is ever RUN here
    rather than merely typechecked.

    ``cached_tokens`` is structurally ``None``: vLLM reports no cache tier on
    the objects read here, and a guessed number in a metering field is worse
    than a missing one.
    """
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    completion = (getattr(request_output, "outputs", None) or [None])[0]
    completion_token_ids = getattr(completion, "token_ids", None)
    return {
        "tokens_in": len(prompt_token_ids) if prompt_token_ids is not None else None,
        "tokens_out": (
            len(completion_token_ids) if completion_token_ids is not None else None
        ),
        "cached_tokens": None,
    }
