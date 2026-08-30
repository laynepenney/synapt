from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Message:
    role: str
    content: str


class ModelClient:
    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.2,
        usage_out: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Run a completion and return its text.

        ``usage_out`` is an ADDITIVE output parameter.  When a caller passes a
        dict, a backend that can report token counts populates it with
        ``tokens_in`` / ``tokens_out`` / ``cached_tokens``.  Callers that omit
        it get the same returned TEXT as before.  That is the accurate claim,
        and it is narrower than "no behaviour change": on vLLM the counts were
        already in hand and were simply being discarded, but on MLX obtaining
        them required driving ``stream_generate`` directly instead of calling
        ``generate``.  The text is identical because ``generate`` is that same
        accumulation loop, and the returned value is unchanged -- but the call
        path did move, and a reader should not be told otherwise.

        ``cached_tokens`` is structurally ``None`` for both self-hosted
        backends: neither runtime reports a cache count on the objects read
        here.  Never estimate it -- ``None`` is the honest value, and a
        guessed number in a metering field is worse than a missing one.
        """
        raise NotImplementedError
