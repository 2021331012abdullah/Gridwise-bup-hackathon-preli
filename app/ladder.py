"""L7 — Relaxation ladder.

Fires only when the LP is infeasible.  Organiser scoring scenarios are promised
feasible, but robustness probes are not, and the safety envelope can
over-constrain.

Rungs:
  1. envelope  (most restrictive superset)
  2. reported  (best guess only)
  3. drop directives by ascending confidence, one at a time
  4. drop by type priority, hardest-to-satisfy first
  5. zero directives
  6. explicit idle plan  (valid for ANY input, since the battery never moves)

Every drop is recorded and logged; nothing is dropped silently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.compile import compile_constraints, normalise_directive
from app.optimize import optimise

logger = logging.getLogger("gridwise.ladder")

DROP_ORDER = [
    "max_grid_window",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "solar_reduction",
]


def _type_of(d: Dict[str, Any]) -> str:
    return d.get("directive_type") or d.get("type") or ""


def _live(directives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only entries that actually carry a constraint."""
    return [d for d in directives if normalise_directive(d) is not None]


def solve_with_ladder(
    scenario: Dict[str, Any],
    reported: List[Dict[str, Any]],
    envelope: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Return (raw_plan, directives_actually_applied, dropped_reasons).

    raw_plan is never None: the last rung is an idle plan that is valid for any
    input because the battery stays put, so SoC is constant and end-of-day
    neutrality holds by construction.
    """
    env = _live(envelope)
    rep = _live(reported)

    # Rung 1 + 2
    for label, live in (("envelope", env), ("reported", rep)):
        raw, _ = optimise(compile_constraints(scenario, live))
        if raw is not None:
            return raw, live, ([] if label == "envelope" else ["envelope_extras"])

    # Rung 3 — shed least-trusted directives first, but stop before emptying the
    # set; the empty set is rung 5, and draining it here made rungs 4 and 5 dead.
    by_conf = sorted(rep, key=lambda d: float(d.get("confidence", 1.0)))
    live = list(by_conf)
    dropped: List[str] = []
    while len(live) > 1:
        live = live[1:]
        dropped.append("low_confidence")
        raw, _ = optimise(compile_constraints(scenario, live))
        if raw is not None:
            logger.warning("ladder: dropped %s", dropped)
            return raw, live, dropped

    # Rung 4 — drop whole types, hardest first.
    live = list(rep)
    dropped = []
    for t in DROP_ORDER:
        if not any(_type_of(d) == t for d in live):
            continue
        live = [d for d in live if _type_of(d) != t]
        dropped.append(t)
        raw, _ = optimise(compile_constraints(scenario, live))
        if raw is not None:
            logger.warning("ladder: dropped types %s", dropped)
            return raw, live, dropped

    # Rung 5 — zero directives.
    raw, _ = optimise(compile_constraints(scenario, []))
    if raw is not None:
        logger.warning("ladder: dropped ALL directives")
        return raw, [], dropped + ["ALL"]

    # Rung 6 — only reachable for a schema-valid impossibility such as
    # initial_energy_kwh > capacity_kwh.
    logger.error("ladder: zero-directive LP infeasible, using idle plan")
    return _idle_plan(scenario), [], dropped + ["ALL", "idle_fallback"]


def _idle_plan(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Battery untouched, solar used up to demand, grid covers the rest."""
    hours_raw = scenario["hours"]
    if hours_raw and not isinstance(hours_raw[0], dict):
        hours_raw = [
            {"hour": h.hour, "demand_kwh": h.demand_kwh, "solar_kwh": h.solar_kwh}
            for h in hours_raw
        ]
    hours = sorted(hours_raw, key=lambda x: x["hour"])

    bat = scenario["battery"]
    e0 = float(bat["initial_energy_kwh"] if isinstance(bat, dict) else bat.initial_energy_kwh)

    return [
        {
            "hour": idx,
            "solar_used_kwh": min(float(h["solar_kwh"]), float(h["demand_kwh"])),
            "battery_action": "idle",
            "battery_kwh": 0.0,
            "battery_energy_after_kwh": e0,
        }
        for idx, h in enumerate(hours)
    ]
