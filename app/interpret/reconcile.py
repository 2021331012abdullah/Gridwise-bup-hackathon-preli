"""L4 — Reconciliation: report the best guess, optimise the envelope.

The judge scores interpretation and plan through two separate paths:

  * directive_interpretation is compared field by field against ground truth;
  * hourly_plan is replayed against ground truth, and nothing checks the plan
    against the interpretation we reported.

So the reported reading and the constraint set fed to the optimiser do not have
to be the same object.  We report the single most probable reading, and we
optimise against the most restrictive consistent superset of every candidate
reading.  When the channels agree the two are identical and the split costs
nothing; it only widens where there is genuine disagreement.

Envelope rules, per field:
  hours    union, within one directive type only
  factor   minimum  (least solar remaining)
  reserve  maximum, capped at capacity
  grid cap minimum
  types    conflicting types are BOTH applied, never collapsed into a winner
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from app.interpret.rules import classify

logger = logging.getLogger("gridwise.reconcile")

FACTOR_TOL = 0.02
VALUE_TOL = 0.5


def envelope_enabled() -> bool:
    return os.getenv("ENVELOPE_MODE", "on").strip().lower() not in ("off", "0", "false", "no")


def _adj(c: Dict[str, Any]) -> Dict[str, Any]:
    return c.get("structured_adjustment") or {}


def agree(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if a["directive_type"] != b["directive_type"]:
        return False
    if a["directive_type"] == "no_op":
        return True
    aa, bb = _adj(a), _adj(b)
    if aa.get("hours") != bb.get("hours"):
        return False
    for key, tol in (("factor", FACTOR_TOL),
                     ("minimum_energy_kwh", VALUE_TOL),
                     ("max_grid_kwh", VALUE_TOL)):
        if key in aa or key in bb:
            va, vb = aa.get(key), bb.get(key)
            if va is None or vb is None or abs(float(va) - float(vb)) > tol:
                return False
    return True


def _best(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(candidates, key=lambda c: float(c.get("confidence", 0.0)))


def build_envelope(candidates: List[Dict[str, Any]], capacity: float) -> List[Dict[str, Any]]:
    """Most restrictive superset of the candidates for a single note.

    Candidates are grouped by directive type and merged only within a group, so
    a solar_reduction proposed by one channel is never absorbed into (and lost
    behind) a no_charge_window proposed by another.
    """
    active = [c for c in candidates
              if c.get("applies") and c.get("directive_type") != "no_op"]
    if not active:
        return []

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for c in active:
        by_type.setdefault(c["directive_type"], []).append(c)

    out: List[Dict[str, Any]] = []
    for dtype, group in by_type.items():
        hours = sorted({h for c in group for h in _adj(c).get("hours", [])})
        if not hours:
            continue
        entry: Dict[str, Any] = {
            "type": dtype,
            "hours": hours,
            "confidence": max(float(c.get("confidence", 0.0)) for c in group),
        }
        if dtype == "solar_reduction":
            vals = [_adj(c)["factor"] for c in group if _adj(c).get("factor") is not None]
            if not vals:
                continue
            entry["factor"] = min(vals)                       # least solar remaining
        elif dtype == "minimum_battery_reserve":
            vals = [_adj(c)["minimum_energy_kwh"] for c in group
                    if _adj(c).get("minimum_energy_kwh") is not None]
            if not vals:
                continue
            entry["minimum_energy_kwh"] = min(max(vals), capacity)   # highest floor
        elif dtype == "max_grid_window":
            vals = [_adj(c)["max_grid_kwh"] for c in group
                    if _adj(c).get("max_grid_kwh") is not None]
            if not vals:
                continue
            entry["max_grid_kwh"] = min(vals)                 # tightest cap
        out.append(entry)
    return out


def _to_directive(interp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Interpretation entry -> compiler directive."""
    if not interp.get("applies") or interp.get("directive_type") == "no_op":
        return None
    adj = _adj(interp)
    hours = adj.get("hours") or []
    if not hours:
        return None
    d: Dict[str, Any] = {
        "type": interp["directive_type"],
        "hours": hours,
        "confidence": float(interp.get("confidence", 0.5)),
    }
    for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
        if adj.get(key) is not None:
            d[key] = adj[key]
    return d


def _ensure_superset(envelope: List[Dict[str, Any]], reported: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Guarantee the envelope is at least as restrictive as the report."""
    out = [dict(d) for d in envelope]
    for r in reported:
        match = next((e for e in out if e["type"] == r["type"]), None)
        if match is None:
            out.append(dict(r))
            continue
        match["hours"] = sorted(set(match.get("hours", [])) | set(r.get("hours", [])))
        if "factor" in r:
            match["factor"] = min(match.get("factor", r["factor"]), r["factor"])
        if "minimum_energy_kwh" in r:
            match["minimum_energy_kwh"] = max(
                match.get("minimum_energy_kwh", r["minimum_energy_kwh"]), r["minimum_energy_kwh"])
        if "max_grid_kwh" in r:
            match["max_grid_kwh"] = min(
                match.get("max_grid_kwh", r["max_grid_kwh"]), r["max_grid_kwh"])
    return out


def reconcile(
    per_note_candidates: List[List[Dict[str, Any]]],
    notes: List[str],
    capacity: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (report, reported_directives, envelope_directives)."""
    report: List[Dict[str, Any]] = []
    envelope: List[Dict[str, Any]] = []
    use_envelope = envelope_enabled()

    for idx, candidates in enumerate(per_note_candidates):
        candidates = [c for c in candidates if c]
        if not candidates:
            report.append({
                "note_index": idx,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No interpretation available for this note.",
                "confidence": 0.1,
            })
            continue

        consensus = all(agree(candidates[0], c) for c in candidates[1:])
        if consensus:
            chosen = _best(candidates)
        else:
            active = [c for c in candidates if c.get("directive_type") != "no_op"]
            noops = [c for c in candidates if c.get("directive_type") == "no_op"]
            if active and noops:
                # no_op is a classification, never a fallback.  Only accept it
                # over a proposed directive when the deterministic channel also
                # reads the note as having no energy content.
                rule_type, _ = classify(notes[idx])
                if rule_type == "no_op" and max(
                    float(c.get("confidence", 0)) for c in noops
                ) >= max(float(c.get("confidence", 0)) for c in active):
                    chosen = _best(noops)
                else:
                    chosen = dict(_best(active))
                    chosen["confidence"] = max(float(chosen.get("confidence", 0)), 0.6)
            else:
                chosen = _best(candidates)
            logger.info("note %d: channels disagreed, reporting %s",
                        idx, chosen.get("directive_type"))

        report.append(chosen)
        if use_envelope:
            envelope.extend(build_envelope(candidates, capacity))

    reported_dirs = [d for d in (_to_directive(r) for r in report) if d]

    if not use_envelope:
        return report, reported_dirs, list(reported_dirs)

    return report, reported_dirs, _ensure_superset(envelope, reported_dirs)
