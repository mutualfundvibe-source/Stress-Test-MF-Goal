"""
Stress-Test Mutual Fund Goal API — FastAPI

What it does
------------
Runs Monte Carlo simulations (monthly) to estimate the probability of reaching a target corpus under multiple stress
scenarios:
- base: normal returns
- early_drawdown: crash in early months (one-off shock and/or depressed drift)
- sip_pause: pause SIP for chosen months
- high_inflation: evaluates success against an inflation-adjusted target

Inputs (JSON)
-------------
POST /stress-test
{
  "corpus": 2000000,             # existing corpus (₹)
  "sip": 30000,                  # monthly SIP (₹)
  "years": 7,                    # time horizon in years
  "target": 10000000,            # target corpus (₹, nominal)

  "sims": 10000,                 # Monte Carlo paths (500–100000)
  "mu_annual": 0.13,             # expected annual return (decimal)
  "sigma_annual": 0.20,          # annual volatility (decimal)
  "seed": 123,                   # optional RNG seed for repeatability

  "scenarios": {
    "base": { "enabled": true },

    "early_drawdown": {          # crash scenario in first K months
      "enabled": true,
      "months": 6,               # length of stress window
      "shock_pct": -0.20,        # one-time shock applied in month 1 (e.g., -20%)
      "depressed_mu": 0.02       # annualized drift during the stress window (after the shock)
    },

    "sip_pause": {
      "enabled": true,
      "months": [1,2,3]          # 1-based month numbers where SIP is skipped
    },

    "high_inflation": {
      "enabled": true,
      "inflation": 0.07          # annual inflation; success measured vs target * (1+infl)^{years}
    }
  }
}

Outputs (JSON)
--------------
- One result per scenario key: success_probability, final corpus percentiles, mean/median, target_used, notes

How to run locally
------------------
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install fastapi uvicorn[standard] pydantic numpy
uvicorn app_stress:app --reload --port 8000

Render deployment
-----------------
Build:  pip install -r requirements.txt
Start:  uvicorn app_stress:app --host 0.0.0.0 --port $PORT

Minimal requirements.txt
------------------------
fastapi\nuvicorn[standard]\npydantic\nnumpy

Example cURL
------------
# Base + stresses
curl -X POST http://localhost:8000/stress-test \
  -H "Content-Type: application/json" \
  -d '{
    "corpus": 2000000,
    "sip": 30000,
    "years": 7,
    "target": 10000000,
    "sims": 10000,
    "mu_annual": 0.13,
    "sigma_annual": 0.20,
    "seed": 42,
    "scenarios": {
      "base": {"enabled": true},
      "early_drawdown": {"enabled": true, "months": 6, "shock_pct": -0.2, "depressed_mu": 0.02},
      "sip_pause": {"enabled": true, "months": [1,2,3]},
      "high_inflation": {"enabled": true, "inflation": 0.07}
    }
  }'
"""

from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
from math import isnan

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, confloat

# ----------------------
# Models
# ----------------------
class EarlyDrawdownCfg(BaseModel):
    enabled: bool = True
    months: conint(ge=1, le=36) = 6
    shock_pct: confloat(lt=0, gt=-0.95) = -0.2
    depressed_mu: confloat(ge=-0.5, le=0.2) = 0.02

class SipPauseCfg(BaseModel):
    enabled: bool = True
    months: List[conint(ge=1)] = Field(default_factory=list, description="1-based months where SIP is skipped")

class HighInflationCfg(BaseModel):
    enabled: bool = True
    inflation: confloat(ge=0, le=0.2) = 0.07

class ScenariosCfg(BaseModel):
    base: Dict[str, bool] = {"enabled": True}
    early_drawdown: EarlyDrawdownCfg = EarlyDrawdownCfg()
    sip_pause: SipPauseCfg = SipPauseCfg()
    high_inflation: HighInflationCfg = HighInflationCfg()

class StressRequest(BaseModel):
    corpus: confloat(ge=0)
    sip: confloat(ge=0)
    years: confloat(gt=0, le=60)
    target: confloat(gt=0)

    sims: conint(ge=500, le=100_000) = 10_000
    mu_annual: confloat(gt=-1, lt=2) = 0.13
    sigma_annual: confloat(gt=0, lt=2) = 0.20
    seed: Optional[int] = None

    scenarios: ScenariosCfg = ScenariosCfg()

class Percentiles(BaseModel):
    p5: float; p25: float; p50: float; p75: float; p95: float

class ScenarioResult(BaseModel):
    scenario: str
    success_probability: float
    mean_final: float
    median_final: float
    percentiles: Percentiles
    months: int
    simulations: int
    target_used: float
    notes: str

class StressResponse(BaseModel):
    results: List[ScenarioResult]

# ----------------------
# Helpers
# ----------------------

def to_monthly(mu_annual: float, sigma_annual: float):
    mu_m = np.log1p(mu_annual) / 12.0     # log drift per month
    sigma_m = sigma_annual / np.sqrt(12.0)
    return mu_m, sigma_m


def simulate(req: StressRequest, *, variant: str) -> ScenarioResult:
    if req.seed is not None:
        np.random.seed(req.seed + hash(variant) % 10_000)

    months = int(round(req.years * 12))
    sims = int(req.sims)
    mu_m, sigma_m = to_monthly(req.mu_annual, req.sigma_annual)

    # random normals for full panel
    Z = np.random.normal(size=(sims, months))
    monthly_r = np.exp((mu_m - 0.5 * sigma_m**2) + sigma_m * Z) - 1.0  # simple returns

    # Start values
    V = np.full(sims, float(req.corpus), dtype=np.float64)

    # Scenario-specific adjustments
    target_used = float(req.target)
    notes: List[str] = []

    if variant == "early_drawdown" and req.scenarios.early_drawdown.enabled:
        ed = req.scenarios.early_drawdown
        m0 = min(months, int(ed.months))
        # One-off shock at month 1
        monthly_r[:, 0] = (1.0 + monthly_r[:, 0]) * (1.0 + float(ed.shock_pct)) - 1.0
        notes.append(f"Shock {ed.shock_pct*100:.0f}% in month 1; depressed mu for {m0} months")
        # Depressed drift for first m0 months
        mu_m_dep, sigma_m_dep = to_monthly(float(ed.depressed_mu), req.sigma_annual)
        Z2 = np.random.normal(size=(sims, m0))
        monthly_r[:, :m0] = np.exp((mu_m_dep - 0.5 * sigma_m_dep**2) + sigma_m_dep * Z2) - 1.0

    skip_months = set()
    if variant == "sip_pause" and req.scenarios.sip_pause.enabled:
        skip_months = {int(m) for m in req.scenarios.sip_pause.months if 1 <= int(m) <= months}
        if skip_months:
            notes.append(f"SIP paused in months: {sorted(skip_months)}")

    if variant == "high_inflation" and req.scenarios.high_inflation.enabled:
        infl = float(req.scenarios.high_inflation.inflation)
        target_used = float(req.target) * ((1.0 + infl) ** float(req.years))
        notes.append(f"Target inflated to ₹{target_used:,.0f} at {infl*100:.1f}% p.a.")

    # Monthly loop
    sip = float(req.sip)
    for m in range(months):
        if not (variant == "sip_pause" and (m+1) in skip_months):
            V += sip  # contribute at start of month
        V *= (1.0 + monthly_r[:, m])

    p = np.mean(V >= target_used)
    mean_final = float(np.mean(V))
    median_final = float(np.median(V))
    p5, p25, p50, p75, p95 = np.percentile(V, [5,25,50,75,95])

    return ScenarioResult(
        scenario=variant,
        success_probability=float(p),
        mean_final=float(mean_final),
        median_final=float(median_final),
        percentiles=Percentiles(p5=float(p5), p25=float(p25), p50=float(p50), p75=float(p75), p95=float(p95)),
        months=months,
        simulations=sims,
        target_used=float(target_used),
        notes=", ".join(notes) if notes else "",
    )

# ----------------------
# App
# ----------------------
app = FastAPI(title="Stress-Test MF Goal API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/stress-test", response_model=StressResponse)
def stress_test(req: StressRequest):
    results: List[ScenarioResult] = []
    cfg = req.scenarios

    # Base
    if cfg.base.get("enabled", True):
        results.append(simulate(req, variant="base"))

    # Early drawdown
    if cfg.early_drawdown.enabled:
        results.append(simulate(req, variant="early_drawdown"))

    # SIP pause
    if cfg.sip_pause.enabled:
        results.append(simulate(req, variant="sip_pause"))

    # High inflation
    if cfg.high_inflation.enabled:
        results.append(simulate(req, variant="high_inflation"))

    return StressResponse(results=results)
