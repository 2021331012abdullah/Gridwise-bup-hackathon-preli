"""L1 — Request and response schemas.

Field names and types follow sections 07 and 10 of the Problem Statement
exactly.

Status-code split (section 6.1):
  400  malformed JSON or structurally invalid request — wrong types, missing
       fields, wrong array lengths, empty or too many notes
  422  well-formed but semantically invalid — duplicate hours, hours that do
       not cover 0..23

The distinction is carried by SemanticRequestError so the handler does not have
to guess from an error message.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SemanticRequestError(ValueError):
    """Well-formed request that cannot describe a real scenario -> HTTP 422."""


class HourIn(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    demand_kwh: float = Field(..., ge=0.0)
    solar_kwh: float = Field(..., ge=0.0)
    tariff_bdt_per_kwh: float

    @field_validator("hour", mode="before")
    @classmethod
    def _int_hour(cls, v: Any) -> Any:
        # bool is a subclass of int; a JSON true must not become hour 1.
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
            raise ValueError("hour must be an integer between 0 and 23")
        return int(v)

    @field_validator("demand_kwh", "solar_kwh", "tariff_bdt_per_kwh", mode="before")
    @classmethod
    def _finite(cls, v: Any) -> Any:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("must be a number")
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("must be finite")
        return v


class BatteryIn(BaseModel):
    capacity_kwh: float = Field(..., gt=0.0)
    initial_energy_kwh: float = Field(..., ge=0.0)
    minimum_energy_kwh: float = Field(..., ge=0.0)
    max_charge_kwh_per_hour: float = Field(..., ge=0.0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0.0)


class ScenarioIn(BaseModel):
    scenario_id: str
    operator_notes: List[str]
    hours: List[HourIn]
    battery: BatteryIn

    @field_validator("operator_notes")
    @classmethod
    def _notes(cls, v: List[str]) -> List[str]:
        if not 1 <= len(v) <= 3:
            raise ValueError("operator_notes must contain between 1 and 3 entries")
        for i, note in enumerate(v):
            if not isinstance(note, str) or not note.strip():
                raise ValueError(f"operator_notes[{i}] must be a non-empty string")
        return v

    @field_validator("hours")
    @classmethod
    def _hours(cls, v: List[HourIn]) -> List[HourIn]:
        if len(v) != 24:
            # Wrong shape -> structurally invalid -> 400.
            raise ValueError(f"hours must contain exactly 24 entries, got {len(v)}")
        seen = [h.hour for h in v]
        if len(set(seen)) != 24 or set(seen) != set(range(24)):
            # Right shape, impossible content -> 422.
            raise SemanticRequestError("hours must cover each of 0..23 exactly once")
        return sorted(v, key=lambda h: h.hour)


class DirectiveInterpretationOut(BaseModel):
    note_index: int
    applies: bool
    directive_type: Literal[
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    ]
    structured_adjustment: Optional[Dict[str, Any]] = None
    explanation: str


class HourlyPlanItemOut(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    grid_kwh: float = Field(..., ge=0.0)
    solar_used_kwh: float = Field(..., ge=0.0)
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float = Field(..., ge=0.0)
    battery_energy_after_kwh: float = Field(..., ge=0.0)


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretationOut]
    hourly_plan: List[HourlyPlanItemOut]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
