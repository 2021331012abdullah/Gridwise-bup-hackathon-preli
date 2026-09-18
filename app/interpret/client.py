"""L2 — Interpretation channels.

Three channels run for every request:

  primary LLM     authoritative structured interpretation   6 s + one 4 s retry
  second provider independent cross-check, best effort      4 s hard cut-off
  rule channel    deterministic regex                       < 1 ms, always completes

The two LLM calls are raced concurrently, so the second provider never adds to
the critical path.  The rule channel costs nothing and is the only channel that
survives total network loss.

Every provider client is created lazily, so /health and the whole deterministic
path work with no API key present — the judges may run the image without
credentials.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.interpret.guardrails import guardrail, guardrail_channel
from app.interpret.prompt import SYSTEM_PROMPT, build_user_prompt
from app.interpret.reconcile import reconcile
from app.interpret.rules import rule_interpret

logger = logging.getLogger("gridwise.interpret")

PRIMARY_TIMEOUT = float(os.getenv("PRIMARY_TIMEOUT_SECONDS", "6.0"))
RETRY_TIMEOUT = float(os.getenv("RETRY_TIMEOUT_SECONDS", "4.0"))
SECONDARY_TIMEOUT = float(os.getenv("SECONDARY_TIMEOUT_SECONDS", "4.0"))

_clients: Dict[str, httpx.AsyncClient] = {}


def _key(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return None


def _client(which: str) -> Optional[httpx.AsyncClient]:
    """Lazily build a shared client.  Returns None when no key is configured."""
    if which in _clients:
        return _clients[which]

    if which == "primary":
        key = _key("OPENAI_API_KEY_PRIMARY", "OPENAI_API_KEY")
        base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    else:
        key = _key("OPENAI_API_KEY_SECONDARY", "DEEPSEEK_API_KEY")
        base = os.getenv("SECONDARY_API_BASE", "https://api.deepseek.com/v1")

    if not key:
        return None

    _clients[which] = httpx.AsyncClient(
        base_url=base.rstrip("/"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=httpx.Timeout(connect=3.0, read=12.0, write=5.0, pool=5.0),
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
    )
    return _clients[which]


async def aclose_clients() -> None:
    for c in _clients.values():
        try:
            await c.aclose()
        except Exception:
            pass
    _clients.clear()


def _extract_json(text: str) -> Optional[dict]:
    """Parse JSON out of a model reply, tolerating code fences and preamble."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


async def _call(
    which: str, notes: List[str], capacity: float, minimum: float, timeout: float
) -> Optional[List[dict]]:
    client = _client(which)
    if client is None:
        return None

    model = (os.getenv("OPENAI_MODEL", "gpt-4o-mini") if which == "primary"
             else os.getenv("SECONDARY_MODEL", "deepseek-chat"))

    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(notes, capacity, minimum)},
        ],
    }

    try:
        resp = await asyncio.wait_for(client.post("/chat/completions", json=body), timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        if not isinstance(parsed, dict):
            return None
        items = parsed.get("interpretations") or parsed.get("directives")
        return items if isinstance(items, list) else None
    except asyncio.TimeoutError:
        logger.warning("%s LLM call timed out after %.1fs", which, timeout)
    except Exception as exc:                       # noqa: BLE001 - never leak details
        logger.warning("%s LLM call failed: %s", which, type(exc).__name__)
    return None


async def interpret_notes(
    notes: List[str], capacity: float, minimum: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Run all channels and reconcile.

    Returns (report, reported_directives, envelope_directives, diagnostics).
    """
    diag: Dict[str, Any] = {"primary": False, "secondary": False, "retry": False}

    # Deterministic channel first: it is instant and always succeeds.
    rules_channel: List[Optional[Dict[str, Any]]] = []
    for i, note in enumerate(notes):
        cand = rule_interpret(note, capacity)
        rules_channel.append(guardrail(cand, i, note, capacity, source="rules"))

    primary_task = asyncio.create_task(_call("primary", notes, capacity, minimum, PRIMARY_TIMEOUT))
    secondary_task = asyncio.create_task(_call("secondary", notes, capacity, minimum, SECONDARY_TIMEOUT))

    primary_raw = await primary_task
    secondary_raw = await secondary_task

    if primary_raw is None and _client("primary") is not None:
        diag["retry"] = True
        primary_raw = await _call("primary", notes, capacity, minimum, RETRY_TIMEOUT)

    primary_channel = guardrail_channel(primary_raw, notes, capacity, "primary") \
        if primary_raw else [None] * len(notes)
    secondary_channel = guardrail_channel(secondary_raw, notes, capacity, "secondary") \
        if secondary_raw else [None] * len(notes)

    diag["primary"] = any(c is not None for c in primary_channel)
    diag["secondary"] = any(c is not None for c in secondary_channel)

    # An LLM reading outranks the deterministic one; the rule channel is a
    # cross-check that widens the envelope and carries the request alone only
    # when every provider is unreachable.
    for ch, boost in ((primary_channel, 1.0), (secondary_channel, 0.92)):
        for c in ch:
            if c:
                c["confidence"] = round(min(1.0, max(float(c.get("confidence", 0.5)), 0.85) * boost), 4)
    for c in rules_channel:
        if c and diag["primary"]:
            c["confidence"] = round(min(float(c.get("confidence", 0.5)), 0.70), 4)

    per_note = [
        [primary_channel[i], secondary_channel[i], rules_channel[i]]
        for i in range(len(notes))
    ]

    report, reported_dirs, envelope_dirs = reconcile(per_note, notes, capacity)
    diag["llm_used"] = diag["primary"] or diag["secondary"]
    return report, reported_dirs, envelope_dirs, diag
