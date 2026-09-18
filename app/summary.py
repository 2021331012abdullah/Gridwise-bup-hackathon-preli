"""Deterministic plan_summary.

Generated from the reported directives and the serialised totals.  It carries no
marks of its own and must never cost an LLM call.
"""

from __future__ import annotations

from typing import Any, Dict, List

LABELS = {
    "solar_reduction": "reduced solar availability",
    "minimum_battery_reserve": "a raised battery reserve floor",
    "no_charge_window": "a battery charging block",
    "no_discharge_window": "a battery discharging block",
    "max_grid_window": "a grid import cap",
}


def _fmt_hours(hours: List[int]) -> str:
    if not hours:
        return "no hours"
    if len(hours) == 24:
        return "all 24 hours"
    if hours == list(range(hours[0], hours[-1] + 1)):
        return f"hours {hours[0]}-{hours[-1]}" if len(hours) > 1 else f"hour {hours[0]}"
    return "hours " + ", ".join(str(h) for h in hours)


def generate_summary(report: List[Dict[str, Any]], totals: Dict[str, Any]) -> str:
    grid = float(totals.get("total_grid_kwh", 0.0))
    cost = float(totals.get("total_cost_bdt", 0.0))
    peak = float(totals.get("peak_grid_kwh", 0.0))

    applied = [r for r in report
               if r.get("applies") and r.get("directive_type") != "no_op"]

    parts: List[str] = []
    if not applied:
        parts.append("No operator directive constrains today's schedule, so the "
                     "battery is scheduled purely to shift load into cheaper hours.")
    else:
        clauses = []
        for r in applied:
            adj = r.get("structured_adjustment") or {}
            hours = adj.get("hours") or []
            label = LABELS.get(r["directive_type"], r["directive_type"])
            detail = ""
            if r["directive_type"] == "solar_reduction" and adj.get("factor") is not None:
                detail = f" to {adj['factor'] * 100:.0f}% of forecast"
            elif r["directive_type"] == "minimum_battery_reserve" and adj.get("minimum_energy_kwh") is not None:
                detail = f" of {adj['minimum_energy_kwh']:.0f} kWh"
            elif r["directive_type"] == "max_grid_window" and adj.get("max_grid_kwh") is not None:
                detail = f" of {adj['max_grid_kwh']:.0f} kWh"
            clauses.append(f"{label}{detail} across {_fmt_hours(hours)}")
        parts.append(
            f"Applied {len(applied)} operator directive(s): " + "; ".join(clauses) + "."
        )
        parts.append("The battery charges in low-tariff hours and discharges into "
                     "peak hours within those limits, ending the day at its starting level.")

    parts.append(
        f"Total grid import {grid:.2f} kWh at {cost:.2f} BDT, peaking at {peak:.2f} kWh."
    )
    return " ".join(parts)
