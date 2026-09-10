"""Thin Anthropic wrapper. Every call degrades to None rather than raising, so the
agent always has a deterministic path to fall back on."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .config import settings

log = logging.getLogger("causal.llm")

_client = None


def _get_client():
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            return None
        try:
            from anthropic import Anthropic

            _client = Anthropic(api_key=settings.anthropic_api_key)
        except Exception as exc:  # pragma: no cover - import/auth issues
            log.warning("Could not construct Anthropic client: %s", exc)
            return None
    return _client


def available() -> bool:
    return settings.llm_enabled and _get_client() is not None


def complete(
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.2,
) -> str | None:
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.messages.create(
            model=model or settings.anthropic_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")
    except Exception as exc:
        log.warning("LLM call failed, falling back to deterministic path: %s", exc)
        return None


def complete_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.2,
) -> dict[str, Any] | None:
    """Ask for JSON and parse it, tolerating fenced code blocks and surrounding prose."""
    text = complete(
        system + "\n\nRespond with a single JSON object and nothing else.",
        user,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not text:
        return None
    return parse_json(text)


def parse_json(text: str) -> dict[str, Any] | None:
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    log.warning("Could not parse JSON from LLM response")
    return None
