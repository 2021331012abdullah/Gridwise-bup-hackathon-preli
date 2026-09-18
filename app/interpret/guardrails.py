"""L3 — Guardrails and repair.

Runs on every channel's output independently.  LLM output is untrusted
structured data until it passes through here (section 08 of the Problem
Statement).

Order: enum -> windows/hours -> unit resolution -> range clamp ->
applies coherence -> evidence check -> exact output shape.

Two principles decide the defaults:
  * A missing number is NEVER replaced by a value that relaxes the directive.
    Defaulting a solar factor to 1.0 means "no reduction at all", which lets
    the plan consume solar that does not exist and invalidates the case.
  * A missing number is NEVER replaced by 0 either, because reporting a cap of
    0 is certainly wrong and makes the LP infeasible, so the directive is lost
    twice over.  It is left as None, and the reconciler sources it from another
    channel or the note text.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.interpret.rules import (
    extract_grid_cap_kwh,
    extract_reserve_kwh,
    extract_solar_factor,
)
from app.interpret.timeparse import extract_window

logger = logging.getLogger("gridwise.guardrails")

VALID_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


def clean_hours(raw: Any) -> List[int]:
    """Unique integers 0..23, ascending.  Anything else is discarded."""
    if not isinstance(raw, (list, tuple, set)):
        return []
    out = set()
    for h in raw:
        if isinstance(h, bool):
            continue
        try:
            v = int(h)
        except (TypeError, ValueError):
            continue
        if 0 <= v <= 23:
            out.add(v)
    return sorted(out)


def expand_windows(windows: Any) -> List[int]:
    """Expand [{start_hour, end_hour_exclusive}] into hour indices.

    s == e is the single hour named, not a full day: a model that emits
    {"start_hour": 15, "end_hour_exclusive": 15} for "at 3 PM" must not silently
    produce an all-day directive.  s > e wraps past midnight.  Every branch is
    clamped to 0..23 because these hours are echoed verbatim in the response.
    """
    if not isinstance(windows, (list, tuple)):
        return []
    hours = set()
    for w in windows:
        if not isinstance(w, dict):
            continue
        try:
            s = int(w.get("start_hour", 0))
            e = int(w.get("end_hour_exclusive", w.get("end_hour", s + 1)))
        except (TypeError, ValueError):
            continue
        if s > 24 or e > 24 or s < 0 or e < 0:
            continue
        if s == e:
            span = [s]
        elif s < e:
            span = list(range(s, e))
        else:
            span = list(range(s, 24)) + list(range(0, e))
        hours.update(h for h in span if 0 <= h <= 23)
    return sorted(hours)


def _resolve_factor(candidate: Dict[str, Any], adj: Dict[str, Any]) -> Optional[float]:
    """Turn any of the accepted solar encodings into the fraction remaining."""
    solar = candidate.get("solar")
    if isinstance(solar, dict) and solar.get("value") is not None:
        try:
            val = float(solar["value"])
        except (TypeError, ValueError):
            val = None
        if val is not None:
            mode = str(solar.get("mode", "remaining_fraction"))
            if mode == "reduction_percent":
                return max(0.0, min(1.0, (100.0 - val) / 100.0))
            if mode == "remaining_percent":
                return max(0.0, min(1.0, val / 100.0))
            # remaining_fraction, or anything unrecognised
            if val > 1.0:
                return max(0.0, min(1.0, val / 100.0))
            return max(0.0, min(1.0, val))

    for source in (adj, candidate):
        if source.get("factor") is not None:
            try:
                val = float(source["factor"])
            except (TypeError, ValueError):
                continue
            # A bare 25 is obviously a percentage, not a factor.
            if val > 1.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
    return None


def _resolve_reserve(candidate: Dict[str, Any], adj: Dict[str, Any], capacity: float) -> Optional[float]:
    reserve = candidate.get("reserve")
    if isinstance(reserve, dict) and reserve.get("value") is not None:
        try:
            val = float(reserve["value"])
        except (TypeError, ValueError):
            val = None
        if val is not None:
            mode = str(reserve.get("mode", "kwh"))
            if mode == "percent_of_capacity":
                return max(0.0, min(capacity, val / 100.0 * capacity))
            if mode == "fraction_of_capacity":
                return max(0.0, min(capacity, val * capacity))
            return max(0.0, min(capacity, val))

    for source in (adj, candidate):
        if source.get("minimum_energy_kwh") is not None:
            try:
                return max(0.0, min(capacity, float(source["minimum_energy_kwh"])))
            except (TypeError, ValueError):
                continue
    return None


def _resolve_grid_cap(candidate: Dict[str, Any], adj: Dict[str, Any]) -> Optional[float]:
    for key in ("max_grid_kwh", "grid_cap_kwh"):
        for source in (adj, candidate):
            if source.get(key) is not None:
                try:
                    v = float(source[key])
                except (TypeError, ValueError):
                    continue
                if v >= 0 and v == v and v not in (float("inf"), float("-inf")):
                    return v
    return None


def _noop(note_index: int, explanation: str, confidence: float) -> Dict[str, Any]:
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": explanation,
        "confidence": confidence,
    }


def guardrail(
    candidate: Dict[str, Any],
    note_index: int,
    note_text: str,
    capacity: float,
    source: str = "llm",
) -> Dict[str, Any]:
    """Normalise and validate one candidate interpretation of one note."""
    if not isinstance(candidate, dict):
        return _noop(note_index, "Unreadable interpretation.", 0.1)

    dtype = str(candidate.get("directive_type") or candidate.get("type") or "no_op").strip().lower()
    confidence = float(candidate.get("confidence", 0.5) or 0.5)

    if dtype not in VALID_TYPES:
        # Never invent an unsupported type; fall back to the deterministic read.
        logger.warning("note %d: unsupported directive_type %r from %s", note_index, dtype, source)
        from app.interpret.rules import classify
        dtype, rule_conf = classify(note_text)
        confidence = min(confidence, rule_conf) * 0.6

    if dtype == "no_op":
        return _noop(
            note_index,
            str(candidate.get("explanation") or "This note does not affect today's 24-hour energy schedule."),
            confidence,
        )

    adj = candidate.get("structured_adjustment")
    if not isinstance(adj, dict):
        adj = {}

    # ── Hours ─────────────────────────────────────────────────────────────
    hours: List[int] = []
    if candidate.get("windows"):
        hours = expand_windows(candidate["windows"])
    if not hours:
        hours = clean_hours(adj.get("hours"))
    if not hours:
        hours = clean_hours(candidate.get("hours"))
    if not hours:
        # Deterministic recovery straight from the note.
        hours = extract_window(note_text)
        if hours:
            confidence *= 0.9
    if not hours:
        logger.warning("note %d: %s with no resolvable hours", note_index, dtype)
        confidence *= 0.25

    out_adj: Dict[str, Any] = {"hours": hours}

    # ── Numeric fields, with deterministic recovery from the note ─────────
    if dtype == "solar_reduction":
        factor = _resolve_factor(candidate, adj)
        if factor is None:
            factor = extract_solar_factor(note_text)
            confidence *= 0.7
        if factor is None:
            # No level stated anywhere.  0.5 is the midpoint; it under-uses
            # solar rather than over-using it, which keeps the plan valid.
            factor = 0.5
            confidence *= 0.4
            logger.warning("note %d: solar_reduction with no factor, assuming 0.5", note_index)
        out_adj["factor"] = round(max(0.0, min(1.0, factor)), 4)

    elif dtype == "minimum_battery_reserve":
        reserve = _resolve_reserve(candidate, adj, capacity)
        if reserve is None:
            reserve = extract_reserve_kwh(note_text, capacity)
            confidence *= 0.7
        if reserve is None:
            logger.warning("note %d: reserve with no value", note_index)
            return _noop(
                note_index,
                "Battery reserve mentioned but no level could be determined.",
                confidence * 0.2,
            )
        out_adj["minimum_energy_kwh"] = round(max(0.0, min(capacity, reserve)), 4)

    elif dtype == "max_grid_window":
        cap_val = _resolve_grid_cap(candidate, adj)
        if cap_val is None:
            cap_val = extract_grid_cap_kwh(note_text)
            confidence *= 0.7
        if cap_val is None:
            logger.warning("note %d: max_grid_window with no cap value", note_index)
            return _noop(
                note_index,
                "Grid limit mentioned but no numeric cap could be determined.",
                confidence * 0.2,
            )
        out_adj["max_grid_kwh"] = round(max(0.0, cap_val), 4)

    # ── Evidence check: cheap hallucination detector ──────────────────────
    evidence = candidate.get("evidence")
    if isinstance(evidence, str) and evidence.strip():
        ev = evidence.lower().strip()
        note_l = note_text.lower()
        if ev not in note_l:
            ev_tokens = set(re.findall(r"\w+", ev))
            note_tokens = set(re.findall(r"\w+", note_l))
            overlap = len(ev_tokens & note_tokens) / max(len(ev_tokens), 1)
            if overlap < 0.6:
                confidence *= 0.5
                logger.info("note %d: evidence span not found in note", note_index)

    return {
        "note_index": note_index,
        "applies": True,
        "directive_type": dtype,
        "structured_adjustment": out_adj,
        "explanation": str(candidate.get("explanation") or f"Interpreted as {dtype}."),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
    }


def guardrail_channel(
    entries: Any,
    notes: List[str],
    capacity: float,
    source: str = "llm",
) -> List[Optional[Dict[str, Any]]]:
    """Normalise a whole channel.  Returns one slot per note, None where the
    channel said nothing about that note (as opposed to saying no_op)."""
    out: List[Optional[Dict[str, Any]]] = [None] * len(notes)
    if not isinstance(entries, list):
        return out

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("note_index", -1))
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(notes)) or out[idx] is not None:
            continue  # out of range, or a duplicate: keep the first
        out[idx] = guardrail(entry, idx, notes[idx], capacity, source)

    return out
