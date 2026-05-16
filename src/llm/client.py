"""Anthropic Claude API wrapper.

- Tenacity-based retry on transient failures (429, 5xx)
- Cost computation against pinned pricing in pipeline.yaml
- Single entry point: LLMClient.call(...)
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anthropic
from anthropic import APIError, RateLimitError

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..utils.config import Config
from ..utils.logging import get_logger
from .budget import BudgetTracker

log = get_logger(__name__)

Role = Literal["user", "assistant"]


@dataclass
class CacheableBlock:
    """Content block optionally marked for prompt caching."""
    text: str
    cache: bool = True


@dataclass
class Block:
    """Plain content block (text or image)."""
    type: Literal["text", "image"]
    text: str | None = None
    image_path: Path | None = None
    image_media_type: str | None = None


@dataclass
class Message:
    role: Role
    blocks: list[Block | CacheableBlock] = field(default_factory=list)


@dataclass
class LLMResponse:
    text: str
    raw: Any
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    usd: float


class LLMClient:
    """High-level Anthropic Claude client used by all pipeline stages."""

    def __init__(self, cfg: Config, budget: BudgetTracker):
        self.cfg = cfg
        self.budget = budget
        api_key = cfg.env.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env or environment")
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0)

    def list_models(self) -> list[str]:
        """Return available model IDs — useful for debugging."""
        try:
            return [m.id for m in self._client.models.list()]
        except Exception as e:
            return [f"(error listing models: {e})"]

    def _price_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_write_tokens: int,
        cache_read_tokens: int,
    ) -> float:
        p = self.cfg.pipeline["pricing"].get(model)
        if not p:
            return 0.0
        return (
            input_tokens / 1_000_000 * p["input"]
            + output_tokens / 1_000_000 * p["output"]
            + cache_write_tokens / 1_000_000 * p.get("cache_write", 0)
            + cache_read_tokens / 1_000_000 * p.get("cache_read", 0)
        )

    def _estimate_input_tokens(self, messages: list[Message], system: str | None) -> int:
        n_chars = len(system or "")
        n_images = 0
        for m in messages:
            for b in m.blocks:
                if isinstance(b, CacheableBlock):
                    n_chars += len(b.text)
                elif b.type == "text":
                    n_chars += len(b.text or "")
                else:
                    n_images += 1
        return max(1, int(n_chars / 3.5)) + n_images * 1700

    @retry(
        retry=retry_if_exception_type((Exception,)),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def _create(self, **kwargs: Any) -> Any:
        return self._client.messages.create(**kwargs)

    def _create_streaming(self, **kwargs: Any) -> Any:
        """Use streaming for long requests (required by Anthropic SDK for >10 min calls)."""
        text_chunks = []
        input_tokens = 0
        output_tokens = 0
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                text_chunks.append(text)
            final = stream.get_final_message()
            input_tokens = final.usage.input_tokens
            output_tokens = final.usage.output_tokens
        # Return a simple namespace that mimics the non-streaming response
        class _FakeUsage:
            pass
        usage = _FakeUsage()
        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens
        usage.cache_creation_input_tokens = 0
        usage.cache_read_input_tokens = 0
        class _FakeBlock:
            type = "text"
            text = "".join(text_chunks)
        class _FakeResp:
            pass
        resp = _FakeResp()
        resp.content = [_FakeBlock()]
        resp.usage = usage
        return resp

    def call(
        self,
        *,
        stage: str,
        model: str,
        messages: list[Message],
        system: list[Block | CacheableBlock] | str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        use_extended_cache: bool = False,
    ) -> LLMResponse:
        """Make a Claude API call. Records cost. Enforces budget pre-flight."""

        # Build system string
        sys_text = ""
        if isinstance(system, str):
            sys_text = system
        elif isinstance(system, list):
            sys_text = "".join(
                b.text for b in system
                if hasattr(b, "text") and b.text
            )

        # Pre-flight budget check
        est_in = self._estimate_input_tokens(messages, sys_text)
        est_cost = self._price_call(model, est_in, max_tokens, 0, 0)
        self.budget.check(est_cost, stage)

        # Build Anthropic message list
        anthropic_messages = []
        for m in messages:
            content: list[dict] = []
            for b in m.blocks:
                if isinstance(b, CacheableBlock):
                    content.append({"type": "text", "text": b.text})
                elif b.type == "text":
                    content.append({"type": "text", "text": b.text or ""})
                elif b.type == "image":
                    assert b.image_path is not None
                    img_b64 = base64.standard_b64encode(
                        b.image_path.read_bytes()
                    ).decode("utf-8")
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": b.image_media_type or "image/jpeg",
                            "data": img_b64,
                        },
                    })
            role = "user" if m.role == "user" else "assistant"
            anthropic_messages.append({"role": role, "content": content})

        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=anthropic_messages,
        )
        if sys_text:
            kwargs["system"] = sys_text
        # Opus 4.7 does not support temperature — skip it for that model
        if "opus-4-7" not in model:
            kwargs["temperature"] = temperature

        t0 = time.time()
        # Use streaming for large output requests (Anthropic SDK requirement for >10min calls)
        if max_tokens > 16000:
            resp = self._create_streaming(**kwargs)
        else:
            resp = self._create(**kwargs)
        elapsed = time.time() - t0

        text = ""
        for block in resp.content:
            if block.type == "text":
                text = block.text
                break

        usage = resp.usage
        in_tok = usage.input_tokens
        out_tok = usage.output_tokens
        cwrite = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cread = getattr(usage, "cache_read_input_tokens", 0) or 0
        usd = self._price_call(model, in_tok, out_tok, cwrite, cread)

        self.budget.record(
            stage=stage,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_write_tokens=cwrite,
            cache_read_tokens=cread,
            usd=usd,
            timestamp=time.time(),
        )

        log.info(
            "LLM %s [%s] in=%d out=%d cw=%d cr=%d $%.4f %.1fs",
            stage, model, in_tok, out_tok, cwrite, cread, usd, elapsed,
        )

        return LLMResponse(
            text=text,
            raw=resp,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_write_tokens=cwrite,
            cache_read_tokens=cread,
            usd=usd,
        )


# ----- Convenience constructors -----

def user_text(text: str) -> Message:
    return Message(role="user", blocks=[Block(type="text", text=text)])


def user_with_image(text: str, image_path: Path, media_type: str = "image/jpeg") -> Message:
    return Message(
        role="user",
        blocks=[
            Block(type="image", image_path=image_path, image_media_type=media_type),
            Block(type="text", text=text),
        ],
    )


def cached_system(*parts: str | tuple[str, bool]) -> list[CacheableBlock]:
    out: list[CacheableBlock] = []
    for p in parts:
        if isinstance(p, tuple):
            text, cache = p
            out.append(CacheableBlock(text=text, cache=cache))
        else:
            out.append(CacheableBlock(text=p, cache=True))
    return out
