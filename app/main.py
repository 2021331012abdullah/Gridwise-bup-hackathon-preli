"""GridWise — FastAPI service.

Pipeline: L1 schema -> L2 interpretation -> L3 guardrails -> L4 reconcile ->
L5 compile -> L6 two-phase LP -> L7 relaxation ladder -> L8 serialise ->
L9 independent replay -> response.

Two rules govern the error paths:
  * A valid request must never return 5xx.  Every failure inside the pipeline —
    timeout, provider outage, solver trouble, a bad interpretation entry — falls
    back to a schedule that is still valid and still returned with HTTP 200.
  * Nothing that leaves this process contains a stack trace, a request body or
    anything resembling a credential.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.cache import interpretation_cache
from app.compile import compile_constraints
from app.interpret.client import aclose_clients, interpret_notes
from app.ladder import solve_with_ladder, _idle_plan
from app.optimize import optimise
from app.schemas import OptimizeResponse, ScenarioIn, SemanticRequestError
from app.serialize import serialise
from app.summary import generate_summary
from app.validate import replay_validate

# ─── Logging with credential redaction ───────────────────────────────────

_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9_\-\.]{8,}|"
    r"(?i:api[_-]?key)\s*[=:]\s*\S+)"
)


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if _SECRET_RE.search(msg):
            record.msg = _SECRET_RE.sub("[REDACTED]", msg)
            record.args = ()
        return True


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
for _h in logging.getLogger().handlers:
    _h.addFilter(RedactingFilter())
logger = logging.getLogger("gridwise")

DEADLINE_S = float(os.getenv("REQUEST_DEADLINE_SECONDS", "24.0"))

_stats: Dict[str, int] = {
    "requests": 0, "cache_hits": 0, "llm_used": 0,
    "relaxations": 0, "validator_rejects": 0, "deadline_fallbacks": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await aclose_clients()


app = FastAPI(title="GridWise Campus Energy Optimization", version="1.0.0", lifespan=lifespan)


# ─── Error handlers ──────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    """400 for structural problems, 422 for well-formed but impossible ones."""
    for err in exc.errors():
        cause = err.get("ctx", {}).get("error")
        if isinstance(cause, SemanticRequestError):
            return JSONResponse(status_code=422, content={"error": "semantic_error"})
        if "must cover each of" in str(err.get("msg", "")):
            return JSONResponse(status_code=422, content={"error": "semantic_error"})
    return JSONResponse(status_code=400, content={"error": "invalid_request"})


@app.exception_handler(ValidationError)
async def on_model_error(request: Request, exc: ValidationError):
    return JSONResponse(status_code=400, content={"error": "invalid_request"})


@app.exception_handler(Exception)
async def on_unhandled(request: Request, exc: Exception):
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal_error"})


# ─── Endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    """Static readiness probe: no LLM, no solver, no I/O, no credentials."""
    return {"status": "ok"}


@app.get("/debug/stats")
async def debug_stats() -> Dict[str, Any]:
    """Diagnostics only; never part of the scored response."""
    return dict(_stats)


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(scenario: ScenarioIn):
    _stats["requests"] += 1
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(_process(scenario), timeout=DEADLINE_S)
    except asyncio.TimeoutError:
        _stats["deadline_fallbacks"] += 1
        logger.error("scenario=%s exceeded %.1fs deadline, serving fallback plan",
                     scenario.scenario_id, DEADLINE_S)
        result = _fallback_response(scenario, "Deadline exceeded; served an unconstrained plan.")
    except Exception:
        logger.exception("scenario=%s pipeline failure, serving fallback plan",
                         scenario.scenario_id)
        result = _fallback_response(scenario, "Recovered from an internal error.")

    logger.info("scenario=%s elapsed=%.3fs", scenario.scenario_id,
                time.perf_counter() - started)
    return result


# ─── Pipeline ────────────────────────────────────────────────────────────

async def _process(scenario: ScenarioIn) -> Dict[str, Any]:
    notes = list(scenario.operator_notes)
    bat = scenario.battery
    capacity = float(bat.capacity_kwh)
    minimum = float(bat.minimum_energy_kwh)
    scenario_dict = scenario.model_dump()

    # L2-L4, cached on the notes plus the two figures the prompt mentions.
    key = interpretation_cache.key(notes, capacity, minimum)
    cached = interpretation_cache.get(key)
    if cached is not None:
        _stats["cache_hits"] += 1
        report, reported_dirs, envelope_dirs = cached
    else:
        report, reported_dirs, envelope_dirs, diag = await interpret_notes(
            notes, capacity, minimum
        )
        if diag.get("llm_used"):
            _stats["llm_used"] += 1
        interpretation_cache.put(key, (report, reported_dirs, envelope_dirs))

    # L5-L7
    raw, applied, dropped = solve_with_ladder(scenario_dict, reported_dirs, envelope_dirs)
    if dropped:
        _stats["relaxations"] += 1
        logger.warning("scenario=%s relaxation: %s", scenario.scenario_id, dropped)

    # L8
    serialised = serialise(raw, scenario_dict, applied)

    # L9 — replay against what was actually applied, then against the reported
    # reading, so an envelope that drifted from the report cannot slip through.
    violations = replay_validate(
        serialised["hourly_plan"], scenario_dict, applied,
        reported_totals=serialised,
    )
    if violations:
        _stats["validator_rejects"] += 1
        logger.error("scenario=%s replay violations: %s",
                     scenario.scenario_id, violations[:5])
        safe = compile_constraints(scenario_dict, [])
        raw_safe, _ = optimise(safe)
        if raw_safe is None:
            raw_safe = _idle_plan(scenario_dict)
        serialised = serialise(raw_safe, scenario_dict, [])
        applied = []

    return _build_response(scenario, report, serialised)


def _build_response(
    scenario: ScenarioIn, report: List[Dict[str, Any]], serialised: Dict[str, Any]
) -> Dict[str, Any]:
    """Assemble the exact response schema, dropping internal fields.

    Coherence is forced here rather than asserted: applies is true if and only if
    the type is not no_op, and structured_adjustment is null if and only if it
    is.  A malformed entry becomes a no_op instead of failing the request.
    """
    entries: List[Dict[str, Any]] = []
    for i in range(len(scenario.operator_notes)):
        src = next((r for r in report if r.get("note_index") == i), None) or {}
        dtype = src.get("directive_type", "no_op")
        adj = src.get("structured_adjustment")

        if dtype == "no_op" or not isinstance(adj, dict) or not adj.get("hours"):
            entries.append({
                "note_index": i,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": str(src.get("explanation")
                                   or "This note does not affect today's 24-hour energy schedule."),
            })
            continue

        clean: Dict[str, Any] = {"hours": sorted({int(h) for h in adj["hours"] if 0 <= int(h) <= 23})}
        if dtype == "solar_reduction":
            clean["factor"] = float(adj.get("factor", 0.5))
        elif dtype == "minimum_battery_reserve":
            clean["minimum_energy_kwh"] = float(adj.get("minimum_energy_kwh", 0.0))
        elif dtype == "max_grid_window":
            clean["max_grid_kwh"] = float(adj.get("max_grid_kwh", 0.0))

        entries.append({
            "note_index": i,
            "applies": True,
            "directive_type": dtype,
            "structured_adjustment": clean,
            "explanation": str(src.get("explanation") or f"Interpreted as {dtype}."),
        })

    return {
        "scenario_id": scenario.scenario_id,
        "directive_interpretation": entries,
        "hourly_plan": serialised["hourly_plan"],
        "total_grid_kwh": serialised["total_grid_kwh"],
        "total_cost_bdt": serialised["total_cost_bdt"],
        "peak_grid_kwh": serialised["peak_grid_kwh"],
        "plan_summary": generate_summary(
            [{"applies": e["applies"], "directive_type": e["directive_type"],
              "structured_adjustment": e["structured_adjustment"]} for e in entries],
            serialised,
        ),
    }


def _fallback_response(scenario: ScenarioIn, reason: str) -> Dict[str, Any]:
    """Last resort: an unconstrained but always-valid schedule, still HTTP 200."""
    scenario_dict = scenario.model_dump()
    try:
        C = compile_constraints(scenario_dict, [])
        raw, _ = optimise(C)
        if raw is None:
            raw = _idle_plan(scenario_dict)
        serialised = serialise(raw, scenario_dict, [])
    except Exception:
        logger.exception("fallback solver failed, emitting idle plan")
        serialised = serialise(_idle_plan(scenario_dict), scenario_dict, [])

    report = [
        {"note_index": i, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None,
         "explanation": "Interpretation unavailable for this request."}
        for i in range(len(scenario.operator_notes))
    ]
    response = _build_response(scenario, report, serialised)
    response["plan_summary"] = f"{reason} " + response["plan_summary"]
    return response
