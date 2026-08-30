from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .base import Message, ModelClient


@dataclass
class MLXOptions:
    max_tokens: int = 256
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0


class MLXClient(ModelClient):
    _BASE_CACHE: Dict[str, Tuple[object, object]] = {}  # model_id → (base_model, tokenizer)
    _CURRENT_ADAPTER: Dict[str, str] = {}  # model_id → currently loaded adapter_path
    _FUSED_CACHE: Dict[Tuple[str, Tuple[str, ...]], Tuple[object, object]] = {}  # (model_id, adapters) → (fused_model, tokenizer)

    def __init__(self, options: Optional[MLXOptions] = None):
        self.options = options or MLXOptions()

    def _format_messages(self, messages: List[Message], tokenizer: object) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            chat_messages = [
                {"role": m.role, "content": m.content} for m in messages if m.content
            ]
            if chat_messages:
                return tokenizer.apply_chat_template(
                    chat_messages,
                    add_generation_prompt=True,
                )
        # Fallback: simple transcript
        lines = []
        for msg in messages:
            role = msg.role.capitalize()
            lines.append(f"{role}: {msg.content}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _load(self, model: str, adapter_path: Optional[str] = None) -> Tuple[object, object]:
        from mlx_lm.utils import load as mlx_load
        from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

        # 1. Load or retrieve cached base model
        if model not in self._BASE_CACHE:
            model_obj, tokenizer = mlx_load(
                model, tokenizer_config={"trust_remote_code": True}
            )
            self._BASE_CACHE[model] = (model_obj, tokenizer)
            self._CURRENT_ADAPTER[model] = ""

        model_obj, tokenizer = self._BASE_CACHE[model]
        current = self._CURRENT_ADAPTER.get(model, "")

        # 2. Swap adapter if needed
        if (adapter_path or "") != current:
            if current:
                remove_lora_layers(model_obj)
            if adapter_path:
                load_adapters(model_obj, adapter_path)
                model_obj.eval()
            self._CURRENT_ADAPTER[model] = adapter_path or ""

        return model_obj, tokenizer

    def _load_fused(
        self, model: str, fuse_adapters: List[str]
    ) -> Tuple[object, object]:
        """Load base model and progressively fuse adapters in-memory.

        Each adapter is loaded, materialized, and fused into the base weights
        before the next adapter is applied — matching the stacked LoRA training
        pattern. Results are cached by (model, adapters) tuple so fusion only
        happens once per unique combination.
        """
        cache_key = (model, tuple(fuse_adapters))
        if cache_key in self._FUSED_CACHE:
            return self._FUSED_CACHE[cache_key]

        from mlx_lm.utils import load as mlx_load
        from mlx_lm.tuner.utils import load_adapters
        import mlx.core as mx
        from mlx.utils import tree_unflatten

        model_obj, tokenizer = mlx_load(
            model, tokenizer_config={"trust_remote_code": True}
        )
        for adapter_path in fuse_adapters:
            load_adapters(model_obj, adapter_path)
            mx.eval(model_obj.parameters())
            fused_layers = [
                (name, module.fuse())
                for name, module in model_obj.named_modules()
                if hasattr(module, "fuse")
            ]
            model_obj.update_modules(tree_unflatten(fused_layers))

        self._FUSED_CACHE[cache_key] = (model_obj, tokenizer)
        return model_obj, tokenizer

    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.2,
        usage_out: Optional[dict] = None,
        adapter_path: Optional[str] = None,
        fuse_adapters: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Run a completion, optionally reporting the runtime's token counts.

        BEHAVIOUR CHANGE, stated rather than buried: this used to call
        ``mlx_lm.generate.generate``.  It now drives ``stream_generate``
        directly, because ``generate`` returns only the text and discards the
        counts we need.  The returned TEXT is unchanged, and that is provable
        rather than hopeful -- ``generate`` is, in full, for the
        ``verbose=False`` path we use::

            text = ""
            for response in stream_generate(model, tokenizer, prompt, **kwargs):
                text += response.text
            return text

        which is exactly the loop below.  Callers that omit ``usage_out`` see
        identical output; what changed is that we call the streaming primitive
        ``generate`` was already calling on our behalf.

        ``response.text`` is the detokenizer's LAST SEGMENT -- a DELTA per
        yield -- so the text must be ACCUMULATED.  ``generation_tokens`` is
        cumulative (``n + 1``) and ``prompt_tokens`` is constant, so the counts
        come from the FINAL response.  A single-yield fixture cannot tell
        accumulate from take-first or take-last; the two-yield witness in
        ``tests/test_model_client_usage_out.py`` is what pins it.
        """
        import importlib

        from mlx_lm.sample_utils import make_sampler

        # mlx_lm/__init__.py does ``from .generate import generate``, which
        # overwrites the ``generate`` ATTRIBUTE on the package object with the
        # function -- so ``import mlx_lm.generate as x`` silently binds the
        # FUNCTION, not the module, and ``x.stream_generate`` raises
        # AttributeError.  importlib reads sys.modules directly and is the only
        # form that reliably yields the module.
        mlx_generate = importlib.import_module("mlx_lm.generate")

        if fuse_adapters:
            model_obj, tokenizer = self._load_fused(model, fuse_adapters)
        else:
            model_obj, tokenizer = self._load(model, adapter_path=adapter_path)
        prompt = self._format_messages(messages, tokenizer)
        sampler = make_sampler(
            temp=temperature,
            top_p=self.options.top_p,
            top_k=self.options.top_k,
            min_p=self.options.min_p,
        )
        text = ""
        final = None
        for response in mlx_generate.stream_generate(
            model_obj,
            tokenizer,
            prompt,
            max_tokens=kwargs.get("max_tokens", self.options.max_tokens),
            sampler=sampler,
        ):
            text += response.text
            final = response

        if usage_out is not None:
            usage_out["tokens_in"] = getattr(final, "prompt_tokens", None)
            usage_out["tokens_out"] = getattr(final, "generation_tokens", None)
            # Structurally null: mlx_lm reports no cache tier on the response.
            usage_out["cached_tokens"] = None

        return text
