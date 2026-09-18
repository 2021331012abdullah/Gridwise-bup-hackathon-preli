"""L6 — Two-phase LP optimiser (SciPy / HiGHS).

Phase 1  minimise sum(tariff[h] * g[h])                       -> C*
Phase 2  minimise sum(c[h] + d[h])  subject to cost <= C* + slack

Phase 1 gives the exact optimum with no epsilon distortion.  Phase 2 picks,
among all equally-optimal schedules, the one that moves the least energy.  At
its optimum at most one of charge/discharge is non-zero in any hour (round-trip
efficiency is 1 in this spec, so simultaneous charge and discharge cancels
exactly and is never profitable), which makes battery_action read straight off
the sign and makes the output deterministic across repeated requests.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

H = 24


def _matrices(C: Dict[str, Any]):
    """Variables (96): g[0:24] | s[24:48] | c[48:72] | d[72:96]."""
    n = 4 * H
    e0 = float(C.get("e0_eff", C["e0"]))
    cap = float(C["capacity"])

    bounds: List[Tuple[float, Optional[float]]] = []
    for h in range(H):  # grid import
        gc = C["grid_cap"][h]
        bounds.append((0.0, float(gc) if math.isfinite(gc) else None))
    for h in range(H):  # solar used
        bounds.append((0.0, max(0.0, float(C["eff_solar"][h]))))
    for h in range(H):  # battery charge
        bounds.append((0.0, max(0.0, float(C["chg_cap"][h]))))
    for h in range(H):  # battery discharge
        bounds.append((0.0, max(0.0, float(C["dis_cap"][h]))))

    A_eq: List[np.ndarray] = []
    b_eq: List[float] = []

    # Hourly energy balance: g[h] + s[h] + d[h] - c[h] = demand[h]
    for h in range(H):
        r = np.zeros(n)
        r[h] = 1.0
        r[H + h] = 1.0
        r[3 * H + h] = 1.0
        r[2 * H + h] = -1.0
        A_eq.append(r)
        b_eq.append(float(C["demand"][h]))

    # End-of-day neutrality: sum(c) - sum(d) = 0
    r = np.zeros(n)
    for h in range(H):
        r[2 * H + h] = 1.0
        r[3 * H + h] = -1.0
    A_eq.append(r)
    b_eq.append(0.0)

    # Rolling state of charge, both directions.
    A_ub: List[np.ndarray] = []
    b_ub: List[float] = []
    for h in range(H):
        r = np.zeros(n)
        for k in range(h + 1):
            r[2 * H + k] = 1.0
            r[3 * H + k] = -1.0
        A_ub.append(r.copy())
        b_ub.append(cap - e0)                      # E[h] <= capacity
        A_ub.append(-r)
        b_ub.append(e0 - float(C["minres"][h]))    # E[h] >= minres[h]

    return n, bounds, np.array(A_eq), np.array(b_eq), A_ub, b_ub


def optimise(
    C: Dict[str, Any], polish: bool = True, slack: float = 1e-4
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[float]]:
    """Return (raw per-hour records, optimal cost) or (None, None) if infeasible."""
    try:
        n, bounds, A_eq, b_eq, A_ub, b_ub = _matrices(C)
    except Exception:
        return None, None

    cost = np.zeros(n)
    for h in range(H):
        cost[h] = float(C["tariff"][h])

    A_ub_arr = np.array(A_ub)
    b_ub_arr = np.array(b_ub)

    try:
        p1 = linprog(cost, A_ub=A_ub_arr, b_ub=b_ub_arr, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
    except Exception:
        return None, None

    if not p1.success or p1.x is None:
        return None, None

    best = p1
    c_star = float(p1.fun)

    if polish:
        thr = np.zeros(n)
        for h in range(H):
            thr[2 * H + h] = 1.0
            thr[3 * H + h] = 1.0
        try:
            p2 = linprog(
                thr,
                A_ub=np.vstack([A_ub_arr, cost.reshape(1, -1)]),
                b_ub=np.append(b_ub_arr, c_star + slack),
                A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
            )
            # Never let the polish pass break a valid answer.
            if p2.success and p2.x is not None:
                best = p2
        except Exception:
            pass

    x = best.x
    e0 = float(C.get("e0_eff", C["e0"]))
    raw: List[Dict[str, Any]] = []
    E = e0

    for h in range(H):
        c_val = max(0.0, float(x[2 * H + h]))
        d_val = max(0.0, float(x[3 * H + h]))
        net = c_val - d_val
        if abs(net) < 1e-7:
            net = 0.0
        E += net
        if net > 0:
            action, mag = "charge", net
        elif net < 0:
            action, mag = "discharge", -net
        else:
            action, mag = "idle", 0.0
        raw.append({
            "hour": h,
            "solar_used_kwh": max(0.0, float(x[H + h])),
            "battery_action": action,
            "battery_kwh": mag,
            "battery_energy_after_kwh": E,
        })

    return raw, c_star
