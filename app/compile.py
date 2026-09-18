"""L5 — Constraint compiler.

Directives in, six flat 24-element arrays out.  Knows nothing about LLMs or
natural language.

Accepts both directive shapes used in this codebase:
  * public schema   {"directive_type": ..., "applies": ..., "structured_adjustment": {...}}
  * internal schema {"type": ..., "hours": [...], "factor": ...}

Overlap policy per array:
  eff_solar  solar[h] * product(factors)   (product <= min <= any single factor,
                                            so it can never cause solar overuse)
  minres     max(base_floor, all directive reserves)
  chg_cap    0 if any no_charge_window covers h
  dis_cap    0 if any no_discharge_window covers h
  grid_cap   min(all max_grid_kwh at h) else +inf
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

H = 24


def combine_solar_factors(current: float, factor: float) -> float:
    """Isolated overlap policy for stacked solar_reduction directives.

    The product of factors in [0, 1] is <= any single factor, so it is the most
    conservative reading and can never cause solar overuse.  Swap to min() here
    (one line) if the organisers clarify a different rule.
    """
    return current * factor


def _num(obj: Any, key: str, default: float = 0.0) -> float:
    """Read a float from a dict or an attribute, tolerating either shape."""
    if isinstance(obj, dict):
        val = obj.get(key, default)
    else:
        val = getattr(obj, key, default)
    return float(val if val is not None else default)


def normalise_directive(d: Any) -> Dict[str, Any] | None:
    """Reduce any accepted directive shape to {type, hours, <numeric field>}.

    Returns None for entries that carry no constraint (no_op, applies=False,
    empty hours, or a missing required numeric value).  Returning None rather
    than a half-built directive keeps a broken entry from silently becoming an
    unconstrained one inside the LP.
    """
    if not isinstance(d, dict):
        return None
    if d.get("applies") is False:
        return None

    dtype = d.get("directive_type") or d.get("type")
    if not dtype or dtype == "no_op":
        return None

    adj = d.get("structured_adjustment")
    if not isinstance(adj, dict):
        adj = d

    raw_hours = adj.get("hours", [])
    if not isinstance(raw_hours, (list, tuple)):
        return None

    hours = sorted({int(h) for h in raw_hours if isinstance(h, (int, float)) and 0 <= int(h) <= 23})
    if not hours:
        return None

    out: Dict[str, Any] = {"type": dtype, "hours": hours}

    if dtype == "solar_reduction":
        if adj.get("factor") is None:
            return None
        out["factor"] = max(0.0, min(1.0, float(adj["factor"])))
    elif dtype == "minimum_battery_reserve":
        if adj.get("minimum_energy_kwh") is None:
            return None
        out["minimum_energy_kwh"] = max(0.0, float(adj["minimum_energy_kwh"]))
    elif dtype == "max_grid_window":
        if adj.get("max_grid_kwh") is None:
            return None
        out["max_grid_kwh"] = max(0.0, float(adj["max_grid_kwh"]))
    elif dtype not in ("no_charge_window", "no_discharge_window"):
        return None  # unsupported type never reaches the optimiser

    return out


def compile_constraints(scenario: Dict[str, Any], directives: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compile a scenario plus directives into the flat arrays the LP needs."""
    hours_raw = scenario["hours"]
    if hours_raw and not isinstance(hours_raw[0], dict):
        hours_raw = [
            {
                "hour": h.hour,
                "demand_kwh": h.demand_kwh,
                "solar_kwh": h.solar_kwh,
                "tariff_bdt_per_kwh": h.tariff_bdt_per_kwh,
            }
            for h in hours_raw
        ]
    hours = sorted(hours_raw, key=lambda x: x["hour"])

    bat = scenario["battery"]
    cap = _num(bat, "capacity_kwh")
    e0 = _num(bat, "initial_energy_kwh")
    base_min = _num(bat, "minimum_energy_kwh")
    maxc = _num(bat, "max_charge_kwh_per_hour")
    maxd = _num(bat, "max_discharge_kwh_per_hour")

    # Relax the BASE floor only, and only when the input itself is impossible
    # (initial < minimum makes hour 23 unsatisfiable because E[23] == initial).
    # This must NEVER touch a directive reserve, whose whole purpose is to push
    # the battery up to a level it does not start at.
    base_floor = min(base_min, e0)

    # Clamping e0 into [0, capacity] keeps the zero-directive LP feasible even
    # for a schema-valid impossibility such as initial_energy > capacity.
    e0_eff = max(0.0, min(e0, cap))

    eff = [float(h["solar_kwh"]) for h in hours]
    minres = [base_floor] * H
    chg = [maxc] * H
    dis = [maxd] * H
    gcap: List[float] = [math.inf] * H

    for raw in directives:
        d = normalise_directive(raw)
        if d is None:
            continue
        t = d["type"]
        if t == "solar_reduction":
            for hr in d["hours"]:
                eff[hr] = combine_solar_factors(eff[hr], d["factor"])
        elif t == "minimum_battery_reserve":
            for hr in d["hours"]:
                minres[hr] = max(minres[hr], d["minimum_energy_kwh"])
        elif t == "no_charge_window":
            for hr in d["hours"]:
                chg[hr] = 0.0
        elif t == "no_discharge_window":
            for hr in d["hours"]:
                dis[hr] = 0.0
        elif t == "max_grid_window":
            for hr in d["hours"]:
                gcap[hr] = min(gcap[hr], d["max_grid_kwh"])

    return {
        "hours": hours,
        "capacity": cap,
        "e0": e0,
        "e0_eff": e0_eff,
        "demand": [float(h["demand_kwh"]) for h in hours],
        "tariff": [float(h["tariff_bdt_per_kwh"]) for h in hours],
        "eff_solar": eff,
        "minres": minres,
        "chg_cap": chg,
        "dis_cap": dis,
        "grid_cap": gcap,
    }
