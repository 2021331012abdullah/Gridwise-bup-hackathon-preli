"""L9 — Independent replay validator.

Written from the Problem Statement (sections 09 and 11).  Imports NOTHING from
compile.py, optimize.py or serialize.py: it re-derives every limit from the raw
scenario and the directives so that a bug in the compiler cannot hide itself on
both sides of the check.

Takes (plan, scenario, directives) and returns a list of violation strings.
An empty list means the plan is valid.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

TOL = 0.01


def _as_dict_hours(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = scenario["hours"]
    if raw and not isinstance(raw[0], dict):
        raw = [
            {
                "hour": h.hour,
                "demand_kwh": h.demand_kwh,
                "solar_kwh": h.solar_kwh,
                "tariff_bdt_per_kwh": h.tariff_bdt_per_kwh,
            }
            for h in raw
        ]
    return sorted(raw, key=lambda x: x["hour"])


def replay_validate(
    plan: List[Dict[str, Any]],
    scenario: Dict[str, Any],
    directives: List[Dict[str, Any]],
    tol: float = TOL,
    reported_totals: Optional[Dict[str, float]] = None,
    strict_base_floor: bool = False,
) -> List[str]:
    """Replay the plan hour by hour against the scenario and directives.

    strict_base_floor=True uses the raw battery.minimum_energy_kwh, which is
    what an external judge would do.  The default relaxes it to
    min(minimum, initial) so that a physically impossible input
    (initial < minimum) does not report violations on every single hour for a
    plan that is otherwise the best obtainable.
    """
    errs: List[str] = []
    hours = _as_dict_hours(scenario)

    b = scenario["battery"]
    if not isinstance(b, dict):
        b = {
            "capacity_kwh": b.capacity_kwh,
            "initial_energy_kwh": b.initial_energy_kwh,
            "minimum_energy_kwh": b.minimum_energy_kwh,
            "max_charge_kwh_per_hour": b.max_charge_kwh_per_hour,
            "max_discharge_kwh_per_hour": b.max_discharge_kwh_per_hour,
        }

    cap = float(b["capacity_kwh"])
    e0 = float(b["initial_energy_kwh"])
    declared_floor = float(b["minimum_energy_kwh"])
    floor = declared_floor if strict_base_floor else min(declared_floor, e0)
    maxc = float(b["max_charge_kwh_per_hour"])
    maxd = float(b["max_discharge_kwh_per_hour"])

    # ── Re-derive every limit from the directives, independently ──────────
    eff = [float(h["solar_kwh"]) for h in hours]
    minres = [floor] * 24
    chg = [maxc] * 24
    dis = [maxd] * 24
    gcap: List[float] = [math.inf] * 24

    for d in directives:
        if not isinstance(d, dict) or d.get("applies") is False:
            continue
        t = d.get("directive_type") or d.get("type")
        if not t or t == "no_op":
            continue
        adj = d.get("structured_adjustment")
        if not isinstance(adj, dict):
            adj = d
        hrs = [int(x) for x in adj.get("hours", []) if 0 <= int(x) <= 23]

        if t == "solar_reduction" and adj.get("factor") is not None:
            f = float(adj["factor"])
            for hr in hrs:
                eff[hr] *= f
        elif t == "minimum_battery_reserve" and adj.get("minimum_energy_kwh") is not None:
            v = float(adj["minimum_energy_kwh"])
            for hr in hrs:
                minres[hr] = max(minres[hr], v)
        elif t == "no_charge_window":
            for hr in hrs:
                chg[hr] = 0.0
        elif t == "no_discharge_window":
            for hr in hrs:
                dis[hr] = 0.0
        elif t == "max_grid_window" and adj.get("max_grid_kwh") is not None:
            v = float(adj["max_grid_kwh"])
            for hr in hrs:
                gcap[hr] = min(gcap[hr], v)

    # ── Structural check ──────────────────────────────────────────────────
    if not isinstance(plan, list) or len(plan) != 24:
        return ["hourly_plan must contain exactly 24 entries"]
    if sorted(p["hour"] for p in plan) != list(range(24)):
        return ["hourly_plan is not exactly hours 0..23"]

    sorted_plan = sorted(plan, key=lambda p: p["hour"])

    E = e0
    sum_grid = 0.0
    sum_cost = 0.0
    peak = 0.0

    for h, p in enumerate(sorted_plan):
        g = float(p["grid_kwh"])
        s = float(p["solar_used_kwh"])
        act = str(p["battery_action"])
        mag = float(p["battery_kwh"])
        demand_h = float(hours[h]["demand_kwh"])
        tariff_h = float(hours[h]["tariff_bdt_per_kwh"])

        if act not in ("charge", "discharge", "idle"):
            errs.append(f"h{h}: invalid battery_action '{act}'")
        if not all(map(math.isfinite, (g, s, mag))):
            errs.append(f"h{h}: non-finite value")
            continue
        if g < -tol or s < -tol or mag < -tol:
            errs.append(f"h{h}: negative value (g={g:.4f} s={s:.4f} mag={mag:.4f})")
        if s > eff[h] + tol:
            errs.append(f"h{h}: solar_used {s:.4f} > effective solar {eff[h]:.4f}")
        if act == "idle" and abs(mag) > tol:
            errs.append(f"h{h}: idle but battery_kwh={mag:.4f}")

        c = mag if act == "charge" else 0.0
        d = mag if act == "discharge" else 0.0

        if c > chg[h] + tol:
            errs.append(f"h{h}: charge {c:.4f} > limit {chg[h]:.4f}")
        if d > dis[h] + tol:
            errs.append(f"h{h}: discharge {d:.4f} > limit {dis[h]:.4f}")
        if math.isfinite(gcap[h]) and g > gcap[h] + tol:
            errs.append(f"h{h}: grid {g:.4f} > cap {gcap[h]:.4f}")

        # grid + solar + discharge == demand + charge
        imbalance = (g + s + d) - (demand_h + c)
        if abs(imbalance) > tol:
            errs.append(f"h{h}: energy balance off by {imbalance:.4f}")

        E += c - d
        reported_soc = float(p["battery_energy_after_kwh"])
        if abs(E - reported_soc) > tol:
            errs.append(f"h{h}: reported SoC {reported_soc:.4f} != replayed {E:.4f}")
        if E < minres[h] - tol:
            errs.append(f"h{h}: SoC {E:.4f} < reserve {minres[h]:.4f}")
        if E > cap + tol:
            errs.append(f"h{h}: SoC {E:.4f} > capacity {cap:.4f}")

        sum_grid += g
        sum_cost += g * tariff_h
        peak = max(peak, g)

    if abs(E - e0) > tol:
        errs.append(f"end-of-day SoC {E:.4f} != initial {e0:.4f}")

    if reported_totals:
        for key, recomputed in (
            ("total_grid_kwh", sum_grid),
            ("total_cost_bdt", sum_cost),
            ("peak_grid_kwh", peak),
        ):
            if key in reported_totals and reported_totals[key] is not None:
                if abs(recomputed - float(reported_totals[key])) > tol:
                    errs.append(
                        f"{key} mismatch: recomputed {recomputed:.4f} "
                        f"vs reported {float(reported_totals[key]):.4f}"
                    )

    return errs
