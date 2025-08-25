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
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, confloat


# =========================
# Pydantic models
# =========================

class EarlyDrawdownCfg(BaseModel):
    enabled: bool = True
    months: conint(ge=1, le=36) = 6
    shock_pct: confloat(lt=0, gt=-0.95) = -0.20   # e.g. -20% in month 1
    depressed_mu: confloat(ge=-0.5, le=0.2) = 0.02  # annual drift during first `months`


class SipPauseCfg(BaseModel):
    enabled: bool = True
    months: List[conint(ge=1)] = Field(default_factory=list, description="1-based months where SIP is skipped")


class HighInflationCfg(BaseModel):
    enabled: bool = True
    inflation: confloat(ge=0, le=0.2) = 0.07  # 7% p.a. inflation


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


# =========================
# Helpers
# =========================

def to_monthly(mu_annual: float, sigma_annual: float):
    """Convert annual to monthly GBM log-params."""
    mu_m = np.log1p(mu_annual) / 12.0
    sigma_m = sigma_annual / np.sqrt(12.0)
    return mu_m, sigma_m


def gen_common_returns(seed: Optional[int], sims: int, months: int,
                       mu_annual: float, sigma_annual: float):
    """
    Generate ONE common panel of monthly returns for all scenarios.
    Returns:
      base_returns: (sims, months) simple returns for the base case
      Z:            (sims, months) underlying standard normal shocks (for aligned transformations)
      mu_m, sigma_m
    """
    if seed is not None:
        np.random.seed(seed)

    mu_m, sigma_m = to_monthly(mu_annual, sigma_annual)
    Z = np.random.normal(size=(sims, months))          # same shocks for every scenario
    base_returns = np.exp((mu_m - 0.5 * sigma_m**2) + sigma_m * Z) - 1.0
    return base_returns, Z, mu_m, sigma_m


def evolve_portfolio(returns: np.ndarray, sip: float, corpus0: float,
                     skip_months_1based: Optional[set] = None) -> np.ndarray:
    """
    Evolve portfolio with monthly SIP at start of month, then apply return.
    returns: (sims, months)
    skip_months_1based: set of months where SIP is skipped (1-based)
    """
    sims, months = returns.shape
    V = np.full(sims, float(corpus0), dtype=np.float64)
    skip = skip_months_1based or set()
    for m in range(months):
        if (m + 1) not in skip:
            V += sip
        V *= (1.0 + returns[:, m])
    return V


def summarize(V: np.ndarray, target_used: float, months: int, sims: int,
              scenario_name: str, notes: str) -> ScenarioResult:
    p = float(np.mean(V >= target_used))
    mean_final = float(np.mean(V))
    median_final = float(np.median(V))
    p5, p25, p50, p75, p95 = np.percentile(V, [5, 25, 50, 75, 95])

    return ScenarioResult(
        scenario=scenario_name,
        success_probability=p,
        mean_final=mean_final,
        median_final=median_final,
        percentiles=Percentiles(
            p5=float(p5), p25=float(p25), p50=float(p50), p75=float(p75), p95=float(p95)
        ),
        months=months,
        simulations=sims,
        target_used=float(target_used),
        notes=notes
    )


# =========================
# FastAPI app
# =========================

app = FastAPI(title="Stress-Test MF Goal API (Aligned Paths)", version="2.0.0")

# CORS for web embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten later to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/stress-test", response_model=StressResponse)
def stress_test(req: StressRequest):
    months = int(round(float(req.years) * 12))
    sims = int(req.sims)

    # 1) Generate ONE common set of random market shocks & base returns
    base_returns, Z, mu_m, sigma_m = gen_common_returns(
        req.seed, sims, months, float(req.mu_annual), float(req.sigma_annual)
    )

    results: List[ScenarioResult] = []

    # ---------- Base (uses common base returns unchanged) ----------
    if req.scenarios.base.get("enabled", True):
        V_base = evolve_portfolio(base_returns, float(req.sip), float(req.corpus))
        results.append(summarize(
            V_base, float(req.target), months, sims, "base",
            notes="Normal market ups & downs (expected ~13% p.a., with volatility)."
        ))

    # ---------- Early Drawdown (aligned) ----------
    if req.scenarios.early_drawdown.enabled:
        ed = req.scenarios.early_drawdown
        m0 = max(1, min(months, int(ed.months)))

        # Use SAME Z for first m0 months, but with depressed drift (and same sigma)
        mu_m_dep, sigma_m_dep = to_monthly(float(ed.depressed_mu), float(req.sigma_annual))
        # Recompute returns for first m0 months using SAME shocks Z[:, :m0]
        r_ed = base_returns.copy()
        r_ed[:, :m0] = np.exp((mu_m_dep - 0.5 * sigma_m_dep**2) + sigma_m_dep * Z[:, :m0]) - 1.0

        # Apply one-off shock in month 1 on top of depressed month-1 return
        r_ed[:, 0] = (1.0 + r_ed[:, 0]) * (1.0 + float(ed.shock_pct)) - 1.0

        V_ed = evolve_portfolio(r_ed, float(req.sip), float(req.corpus))
        results.append(summarize(
            V_ed, float(req.target), months, sims, "early_drawdown",
            notes=f"Early crash {ed.shock_pct*100:.0f}% in month 1, then {m0} months of weak returns."
        ))

    # ---------- SIP Pause (aligned) ----------
    if req.scenarios.sip_pause.enabled:
        sp = req.scenarios.sip_pause
        skip = {int(m) for m in sp.months if 1 <= int(m) <= months}
        V_sp = evolve_portfolio(base_returns, float(req.sip), float(req.corpus), skip_months_1based=skip)
        results.append(summarize(
            V_sp, float(req.target), months, sims, "sip_pause",
            notes=f"SIP skipped in months: {sorted(skip)}." if skip else "No SIP months skipped."
        ))

    # ---------- High Inflation (aligned) ----------
    if req.scenarios.high_inflation.enabled:
        hi = req.scenarios.high_inflation
        target_inflated = float(req.target) * ((1.0 + float(hi.inflation)) ** float(req.years))
        V_hi = evolve_portfolio(base_returns, float(req.sip), float(req.corpus))
        results.append(summarize(
            V_hi, float(target_inflated), months, sims, "high_inflation",
            notes=f"Target inflated to ₹{target_inflated:,.0f} at {hi.inflation*100:.1f}% p.a."
        ))

    return StressResponse(results=results)
