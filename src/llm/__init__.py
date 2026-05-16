"""Anthropic API client wrappers with caching, retry, cost tracking.

`LLMClient` and the message dataclasses are imported lazily so that consumers
which only need budget tracking (or only need to inspect cost logs) don't
require the anthropic SDK to be installed.
"""
from .budget import BudgetExceeded, BudgetTracker

__all__ = [
    "LLMClient",
    "Message",
    "Block",
    "CacheableBlock",
    "BudgetTracker",
    "BudgetExceeded",
]


def __getattr__(name: str):
    if name in {"LLMClient", "Message", "Block", "CacheableBlock"}:
        from . import client as _client

        return getattr(_client, name)
    raise AttributeError(f"module 'src.llm' has no attribute {name!r}")
