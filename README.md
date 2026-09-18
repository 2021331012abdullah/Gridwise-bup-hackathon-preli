# GridWise — LLM-Assisted Operator Directive Interpretation

BUP CSE Fest 2026 · Online Preliminary Round

An HTTP service that reads free-text campus operator notes, converts them into
structured energy directives, and returns a provably cost-optimal 24-hour
dispatch plan that satisfies every directive.

---

## 1. Quickstart

```bash
git clone <repo-url> && cd gridwise

cp .env.example .env          # then fill in OPENAI_API_KEY_PRIMARY

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Verify:

```bash
curl -s localhost:8000/health
# {"status":"ok"}

curl -s -X POST localhost:8000/optimize-energy \
     -H 'Content-Type: application/json' \
     -d @docs/sample01.json | head -40
```

`/health` answers with no API key configured. The service also runs without any
credentials at all: the deterministic channel then carries interpretation on its
own, at reduced accuracy.

## 2. Configuration

| Variable | Required | Meaning |
|---|---|---|
| `OPENAI_API_KEY_PRIMARY` | yes, for full marks | Primary interpretation provider |
| `OPENAI_MODEL` | no | Default `gpt-4o-mini` |
| `OPENAI_API_BASE` | no | Override for a proxy or Azure endpoint |
| `OPENAI_API_KEY_SECONDARY` | recommended | Second provider, used as an independent cross-check |
| `SECONDARY_MODEL` / `SECONDARY_API_BASE` | no | Default `deepseek-chat` at `api.deepseek.com/v1` |
| `ENVELOPE_MODE` | no | `on` (default) or `off` — see §4 |
| `REQUEST_DEADLINE_SECONDS` | no | Hard ceiling per request, default 24 |
| `PRIMARY_TIMEOUT_SECONDS` / `RETRY_TIMEOUT_SECONDS` / `SECONDARY_TIMEOUT_SECONDS` | no | 6 / 4 / 4 |
| `PORT`, `LOG_LEVEL` | no | Default 8000, INFO |

## 3. Testing

All test data lives in `docs/`. No network or API key is needed: the suites run
the deterministic channel, which is the service's worst case.

```bash
python tests/run_public.py     # Tier 1  maths vs the 10 official public cases
python tests/run_suites.py     # end-to-end across all four supplied suites
python tests/test_oracle.py    # optimiser isolated from interpretation
python tests/test_tiers.py     # Tier 2 fuzz · Tier 3 validator · Tier 6 contract
python tests/check_time.py     # time-window parser
python tests/check_rules.py    # rule-channel classification and values
```

Expected output is summarised in §7.

## 4. Architecture

```
POST /optimize-energy
  │
  ├─ L1  Schema gate (Pydantic)         400 structural · 422 semantic
  ├─ L2  Interpretation fan-out (concurrent)
  │        primary LLM        6 s + one 4 s retry
  │        second provider    4 s hard cut-off, best effort
  │        rule channel       < 1 ms, always completes
  ├─ L3  Guardrails           normalise · resolve units · range-check · repair
  ├─ L4  Reconcile            REPORT = best guess · ENVELOPE = restrictive superset
  ├─ L5  Compiler             directives → six flat 24-element arrays
  ├─ L6  Two-phase LP         HiGHS: min cost, then min battery throughput
  ├─ L7  Relaxation ladder    only if infeasible; every drop is logged
  ├─ L8  Serialiser           grid derived from the balance; totals recomputed
  └─ L9  Independent replay   written from the spec, imports nothing from L5/L6
```

**The LLM sits in the interpretation path**, not in cosmetic text. It converts
each note into a structured directive; `plan_summary` is generated
deterministically and costs no tokens.

**Arithmetic never happens inside the model.** The prompt asks for intent —
`windows` with `start_hour`/`end_hour_exclusive`, and tagged numeric modes such
as `reduction_percent` versus `remaining_fraction`. Code does the conversion, so
end-exclusive off-by-ones and the "reduced *by* 80%" versus "reduced *to* 80%"
inversion cannot be got wrong by a sampling accident.

**Report the best guess, optimise the envelope.** The judge compares
`directive_interpretation` field by field against ground truth, and separately
replays `hourly_plan` against ground truth — nothing checks the plan against the
interpretation we reported. So the two are decoupled: we report the single most
probable reading, and we optimise against the most restrictive consistent
superset of every candidate reading. When the channels agree the two are
identical and this costs nothing. Conflicting *types* are both applied, never
collapsed into a winner. Disable with `ENVELOPE_MODE=off`.

**Why an LP with no integer variables.** Round-trip efficiency is 1 in this
spec, so charging and discharging in the same hour cancels exactly and is never
profitable. Phase 2 minimises `charge + discharge` at fixed optimal cost, so at
its optimum at most one is non-zero in any hour and `battery_action` reads
straight off the sign. This also makes the output deterministic across repeated
requests.

**L9 does not import from L5 or L6.** It re-derives every limit from the
Problem Statement text. A validator that shares the compiler's code cannot catch
the compiler's bugs.

## 5. Docker

```bash
docker build -t gridwise:1.0.0 .

docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY_PRIMARY=sk-... \
  gridwise:1.0.0

curl -s localhost:8000/health   # {"status":"ok"}
```

To publish and verify on a machine that never built it:

```bash
docker tag gridwise:1.0.0 ghcr.io/<org>/gridwise:1.0.0
docker push ghcr.io/<org>/gridwise:1.0.0
docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/<org>/gridwise:1.0.0

docker pull ghcr.io/<org>/gridwise@sha256:<digest>
docker run --rm -p 8000:8000 ghcr.io/<org>/gridwise@sha256:<digest>
```

The image contains `app/` only — no `.env`, no tests, no credentials in any
layer. `/health` returns 200 with no API key set, because judges may run the
image without credentials. It listens on `0.0.0.0` and reads `PORT` from the
environment.

## 6. Dependencies, limitations and secret handling

**Dependencies** — FastAPI and Uvicorn (async HTTP), Pydantic v2 (request and
response validation), SciPy's HiGHS backend (linear programming), NumPy (matrix
assembly), httpx (async provider calls), python-dotenv (configuration). Credit
to each project.

**Limitations, stated honestly**

- Round-trip battery efficiency is assumed to be 1.0, per the spec.
- Overlapping `solar_reduction` directives are multiplied. The spec is silent
  on this; the product of factors in [0,1] is at most any single factor, so it
  is the most conservative reading and can never cause solar overuse. It is
  isolated in `combine_solar_factors()` and is a one-line change.
- Two notes that genuinely contradict each other cannot both be honoured. The
  relaxation ladder drops the least-trusted directive first and logs what it
  dropped; the organisers state that scoring scenarios are always feasible.
- The deterministic channel handles English, Bengali and Banglish well but does
  not do cross-note reasoning (one note cancelling another), counterfactuals, or
  relative-time arithmetic such as "three hours before midnight". Those depend
  on the LLM channel.
- `initial_energy_kwh` outside `[minimum, capacity]` is physically impossible
  rather than malformed. The service relaxes the *base* floor only — never a
  directive reserve — and still returns a valid schedule.

**Secret handling**

- Keys are read from the environment only, and every provider client is built
  lazily so no credential is needed to start or to answer `/health`.
- `.gitignore` excludes `.env` from the first commit; `.env.example` carries
  variable names only.
- A logging filter redacts anything matching `sk-…`, `Bearer …` or `api_key=…`.
- Error responses are `{"error": "..."}` with no stack trace, no request echo
  and no provider detail. Request bodies are never logged.

## 7. Measured results

Every figure below is produced by the commands in §3, with the LLM channel
disabled — that is, the deterministic worst case.

| Check | Result |
|---|---|
| Tier 1 · 10 official public cases | **10/10 valid, cost ratio 1.000000**, median 5.0 ms |
| Optimiser with ground-truth interpretation | **127/127 valid, 0 invalid** |
| Tier 2 · 1,500 randomised fuzz scenarios | **0 invalid plans**; worst hourly imbalance 1e-4 kWh |
| Tier 3 · 20 deliberately corrupted plans | **20/20 caught** by the independent validator |
| End-to-end across all four supplied suites | **151/155 valid** (feasible ground truth), 0 schema errors |
| Directive type accuracy | 180/187 |
| Full extraction (type + hours + values) | 177/187 |
| Latency, compile → LP → serialise → validate | p50 5.2 ms · p95 11.6 ms · max 20.1 ms |

Three suite cases ship deliberately contradictory ground truth — their own
validation rules say `accept_500_error: true`. This service never returns 5xx
for them; it degrades through the relaxation ladder and still answers 200 with a
structurally valid plan (3/3).

The four remaining end-to-end failures are interpretation-only, and all four are
in the adversarial TITAN suite: a mid-sentence retraction, a compositional
fraction, a relative-time expression, and one cross-note cancellation. They are
the cases the LLM channel exists to handle.

---

*A valid mediocre plan beats an invalid optimal one on every hidden case.*
