"""
api.py

High-performance RESTful API exposing the portfolio risk analytics engine
(finance_math.py) via FastAPI. Clients submit only tickers and weights;
this service automatically fetches 3 years of daily historical pricing
for portfolio assets from Yahoo Finance (yfinance) and the EGX 30
benchmark from egx30_benchmark.get_egx30() (a self-refreshing local
loader, since TradingView/tvDatafeed is permanently rate-limited),
inner-joins the two PRICE series on date (no forward-fill), differences
that single aligned price frame once into log returns (for covariance,
beta, and risk decomposition) and simple returns (for expected portfolio/
market return, since simple returns — unlike log returns — are additive
across assets), and runs the full risk pipeline, returning a single
structured JSON response.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import uvicorn
import yfinance as yf
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from egx30_benchmark import get_egx30
from finance_math import (
    calculate_asset_betas,
    calculate_covariance_matrix,
    calculate_historical_returns,
    calculate_market_volatility,
    calculate_portfolio_beta,
    calculate_portfolio_variance,
    calculate_portfolio_volatility,
    calculate_risk_adjusted_metrics,
    calculate_roys_safety_first,
    decompose_risk,
)

logger = logging.getLogger("portfolio_risk_api")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------
# Market data configuration
# --------------------------------------------------------------------------

EGX_SUFFIX = ".CA"
MARKET_TICKER = "EGX30"
LOOKBACK_PERIOD = "3y"
TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS = 60


def _format_egx_ticker(ticker: str) -> str:
    """Normalize a raw ticker to its EGX (Cairo) Yahoo Finance symbol.

    Index symbols (prefixed with '^') and tickers already carrying the
    '.CA' suffix are returned unchanged; everything else has '.CA' appended.

    Args:
        ticker: Raw ticker as submitted by the client (e.g. 'COMI').

    Returns:
        The Yahoo Finance-compatible symbol (e.g. 'COMI.CA').
    """
    normalized = ticker.strip().upper()
    if normalized.startswith("^") or normalized.endswith(EGX_SUFFIX):
        return normalized
    return f"{normalized}{EGX_SUFFIX}"


def _fetch_adjusted_close(ticker: str) -> pd.Series:
    """Fetch 3 years of daily 'Adj Close' history for a single ticker.

    Args:
        ticker: Yahoo Finance-compatible symbol to fetch.

    Returns:
        A Series of adjusted close prices indexed by date.

    Raises:
        ValueError: if yfinance returns no usable data for the ticker
            (invalid symbol, delisted, or a transient fetch failure).
    """
    try:
        history = yf.download(
            ticker,
            period=LOOKBACK_PERIOD,
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        # The user-facing ValueError below stays generic; this is the only
        # place the real exception type/message/status reaches a log.
        logger.exception("yf.download raised for %s: %s: %s", ticker, type(exc).__name__, exc)
        raise ValueError(f"Invalid ticker symbol or no data available for: {ticker}") from exc

    if history.empty or "Adj Close" not in history.columns.get_level_values(0):
        # No exception — distinct from the case above: this is Yahoo
        # returning a "successful" (HTTP 200) response with no usable rows,
        # e.g. a soft rate-limit/block rather than a thrown 429/403.
        logger.error(
            "yf.download returned no usable data for %s with no exception raised "
            "(empty=%s, columns=%s)",
            ticker,
            history.empty,
            list(history.columns),
        )
        raise ValueError(f"Invalid ticker symbol or no data available for: {ticker}")

    adj_close = history["Adj Close"]
    series = (adj_close.iloc[:, 0] if isinstance(adj_close, pd.DataFrame) else adj_close).dropna()

    if series.empty:
        logger.error(
            "yf.download returned rows for %s but 'Adj Close' was entirely NaN after dropna "
            "(no exception raised, columns=%s)",
            ticker,
            list(history.columns),
        )
        raise ValueError(f"Invalid ticker symbol or no data available for: {ticker}")

    series.name = ticker
    return series


def _fetch_price_history(tickers: List[str]) -> Dict[str, pd.Series]:
    """Fetch adjusted-close history for multiple tickers concurrently.

    Args:
        tickers: Yahoo Finance-compatible symbols to fetch in parallel.

    Returns:
        Dict mapping each ticker to its adjusted-close price Series.

    Raises:
        ValueError: propagated from `_fetch_adjusted_close` if any ticker
            has no usable data.
    """
    results: Dict[str, pd.Series] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(tickers))) as executor:
        future_to_ticker = {executor.submit(_fetch_adjusted_close, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            results[ticker] = future.result()
    return results


# --------------------------------------------------------------------------
# Pydantic Schemas
# --------------------------------------------------------------------------

class AssetWeight(BaseModel):
    """A portfolio constituent identified by ticker with its target weight."""

    ticker: str = Field(..., min_length=1, description="Asset ticker, e.g. 'COMI' (Cairo suffix optional).")
    weight: float = Field(
        ..., ge=0, description="Portfolio weight allocated to this asset (non-negative; EGX has no retail short selling)."
    )


class PortfolioRequest(BaseModel):
    """Full payload required to run the portfolio risk analytics pipeline.

    Historical prices are not supplied by the client — they are fetched
    automatically from Yahoo Finance for each ticker (EGX-suffixed), and
    the EGX 30 market benchmark is fetched separately via egx30_benchmark.
    """

    assets: List[AssetWeight] = Field(..., min_length=1, description="Portfolio constituents and weights.")
    risk_free_rate: float = Field(
        default=0.0,
        gt=-1,
        description="ANNUAL risk-free rate as a decimal (e.g. 0.18 for 18%). Converted internally to a "
        "daily-compounded rate via (1 + rf) ** (1/252) - 1.",
    )
    threshold_return: float = Field(
        default=0.0,
        gt=-1,
        description="ANNUAL minimum acceptable return as a decimal (e.g. 0.10 for 10%), used for Roy's "
        "Safety-First ratio. Converted internally to a daily-compounded rate via (1 + r) ** (1/252) - 1.",
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "PortfolioRequest":
        tickers = [a.ticker.strip().upper() for a in self.assets]
        if len(set(tickers)) != len(tickers):
            raise ValueError("Asset tickers must be unique.")

        weights = np.array([a.weight for a in self.assets], dtype=np.float64)
        if not np.isclose(weights.sum(), 1.0, atol=1e-4):
            raise ValueError(f"Asset weights must sum to 1.0 (got {weights.sum():.6f}).")

        return self


# ---- Response schemas ----

class RiskDecomposition(BaseModel):
    total_variance: float
    systematic_variance: float
    unsystematic_variance: float
    portfolio_beta: float
    pct_systematic: float


class RiskAdjustedMetrics(BaseModel):
    sharpe_ratio: float
    treynor_ratio: float
    jensens_alpha: float
    m2_measure: float


class PortfolioAnalysisResponse(BaseModel):
    tickers: List[str]
    weights: List[float]
    market_ticker: str
    period_start: str
    period_end: str
    observations: int
    stale_return_pct: float
    covariance_matrix: Dict[str, Dict[str, float]]
    asset_betas: Dict[str, float]
    portfolio_variance: float
    portfolio_volatility: float
    portfolio_beta: float
    market_volatility: float
    portfolio_expected_return: float
    market_expected_return: float
    risk_decomposition: RiskDecomposition
    risk_adjusted_metrics: RiskAdjustedMetrics
    roys_safety_first_ratio: float
    threshold_cleared: bool


# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------

app = FastAPI(
    title="Portfolio Risk Analytics API",
    description="Vectorized N-asset portfolio risk analytics engine exposed over REST, backed by live EGX market data.",
    version="2.0.0",
)

_default_allowed_origins = "http://localhost:3000"
_allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", _default_allowed_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Global exception handling
# --------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "InvalidComputationInput", "detail": str(exc)},
    )


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "MissingDataKey", "detail": f"Missing expected key: {exc}"},
    )


@app.exception_handler(TypeError)
async def type_error_handler(request: Request, exc: TypeError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "InvalidDataType", "detail": str(exc)},
    )


@app.exception_handler(np.linalg.LinAlgError)
async def linalg_error_handler(request: Request, exc: np.linalg.LinAlgError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "SingularMatrixError", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception while processing request.")
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "detail": "An unexpected error occurred while computing risk metrics."},
    )


# --------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------

@app.post("/api/v1/analyze-portfolio", response_model=PortfolioAnalysisResponse)
def analyze_portfolio(request: PortfolioRequest) -> PortfolioAnalysisResponse:
    """Run the full risk analytics pipeline against a submitted portfolio.

    Pipeline: ticker/weight list -> automated 3y asset price fetch (yfinance)
    -> prices inner-joined against the EGX30 benchmark (egx30_benchmark) on
    date, no forward-fill -> that single aligned price frame differenced
    once into log returns (covariance, beta, risk decomposition, market
    volatility) and simple returns (expected portfolio/market return) ->
    portfolio variance/volatility -> asset & portfolio beta -> systematic/
    unsystematic risk decomposition -> risk-adjusted performance metrics
    (Sharpe, Treynor, Jensen's Alpha, M^2) -> Roy's Safety-First ratio.
    """
    original_tickers = [a.ticker for a in request.assets]
    weights = np.array([a.weight for a in request.assets], dtype=np.float64)

    formatted_map = {ticker: _format_egx_ticker(ticker) for ticker in original_tickers}

    # _format_egx_ticker maps e.g. both "COMI" and "COMI.CA" to the same
    # Yahoo symbol; the request-level uniqueness check only sees raw
    # tickers, so a collision here would otherwise fetch one price series
    # into two identical columns and silently double-count the position.
    reverse_map: Dict[str, List[str]] = {}
    for orig, formatted in formatted_map.items():
        reverse_map.setdefault(formatted, []).append(orig)
    collisions = {formatted: origs for formatted, origs in reverse_map.items() if len(origs) > 1}
    if collisions:
        detail = "; ".join(f"{origs} both resolve to '{formatted}'" for formatted, origs in collisions.items())
        raise ValueError(f"Duplicate tickers after EGX symbol normalization: {detail}")

    fetch_tickers = list(formatted_map.values())

    fetched_series = _fetch_price_history(fetch_tickers)
    benchmark_series = get_egx30()

    price_data = {orig: fetched_series[formatted_map[orig]] for orig in original_tickers}
    asset_price_df = pd.DataFrame(price_data)

    # Align at the PRICE level, once: every asset AND the benchmark must
    # have a recorded trade on a date for it to survive. No forward-fill —
    # that would inject artificial zero returns on stale-quote days,
    # deflating volatility and pulling beta toward zero.
    price_df = asset_price_df.join(benchmark_series, how="inner").dropna()

    # Log returns are time-additive (consistent with the x252/sqrt(252)
    # annualization below) and drive every risk statistic. Simple returns
    # are used only for expected return, since portfolio return is the
    # weighted SUM of simple returns across assets — log returns are not
    # additive across assets (Jensen's inequality). Both are differenced
    # from this one aligned price frame, so every series shares one grid.
    log_returns_df = calculate_historical_returns(price_df, method="log")
    simple_returns_df = calculate_historical_returns(price_df, method="simple")

    n_obs = log_returns_df.shape[0]
    if n_obs < MIN_OBSERVATIONS:
        date_range = (
            f"{log_returns_df.index.min().date()} to {log_returns_df.index.max().date()}"
            if n_obs > 0
            else "no overlapping dates"
        )
        raise ValueError(
            f"Insufficient return history to compute robust risk metrics: found {n_obs} observations "
            f"(minimum {MIN_OBSERVATIONS} required), spanning {date_range}."
        )

    asset_log_returns = log_returns_df[original_tickers]

    cov_matrix = calculate_covariance_matrix(asset_log_returns)
    portfolio_variance = calculate_portfolio_variance(weights, cov_matrix)
    portfolio_volatility = calculate_portfolio_volatility(portfolio_variance)

    asset_betas = calculate_asset_betas(log_returns_df, MARKET_TICKER)
    portfolio_beta = calculate_portfolio_beta(weights, asset_betas.loc[original_tickers])

    market_volatility = calculate_market_volatility(log_returns_df, MARKET_TICKER)

    risk_decomp = decompose_risk(weights, log_returns_df, MARKET_TICKER, asset_order=original_tickers)

    asset_simple_returns = simple_returns_df[original_tickers]
    portfolio_expected_return = float(asset_simple_returns.mean().to_numpy() @ weights)
    market_expected_return = float(simple_returns_df[MARKET_TICKER].mean())

    # Data-quality proxy: a return cell that is exactly zero is almost
    # always a stale/halted quote, not a genuine flat day — these deflate
    # volatility and pull beta toward zero if left unflagged.
    stale_mask = asset_log_returns.abs().to_numpy() < 1e-12
    stale_return_pct = float(stale_mask.sum() / stale_mask.size) if stale_mask.size > 0 else 0.0

    # Rf/threshold arrive as ANNUAL rates; convert to daily-compounded
    # equivalents (not naive division, which mis-reannualizes materially
    # at Egypt's 18-27% rates). Field constraints (gt=-1) already guarantee
    # the root below is defined.
    rf_daily = (1.0 + request.risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    threshold_daily = (1.0 + request.threshold_return) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

    risk_adjusted = calculate_risk_adjusted_metrics(
        portfolio_return=portfolio_expected_return,
        market_return=market_expected_return,
        risk_free_rate=rf_daily,
        portfolio_volatility=portfolio_volatility,
        portfolio_beta=portfolio_beta,
        market_volatility=market_volatility,
    )

    sf_ratio = calculate_roys_safety_first(
        portfolio_return=portfolio_expected_return,
        threshold_return=threshold_daily,
        portfolio_volatility=portfolio_volatility,
    )

    # Annualize daily figures (252 trading days/year) at the API boundary so
    # finance_math's core functions stay period-agnostic. Returns/variance
    # scale linearly with time; volatility-based ratios scale by sqrt(252)
    # since their denominator (volatility) scales by sqrt(252) while their
    # numerator (a return) scales by 252.
    annualized_portfolio_return = portfolio_expected_return * TRADING_DAYS_PER_YEAR
    annualized_market_return = market_expected_return * TRADING_DAYS_PER_YEAR
    annualized_portfolio_variance = portfolio_variance * TRADING_DAYS_PER_YEAR
    annualized_portfolio_volatility = portfolio_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    annualized_market_volatility = market_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    annualized_sf_ratio = sf_ratio * np.sqrt(TRADING_DAYS_PER_YEAR)
    annualized_threshold_return = threshold_daily * TRADING_DAYS_PER_YEAR

    risk_adjusted["sharpe_ratio"] *= np.sqrt(TRADING_DAYS_PER_YEAR)
    risk_adjusted["treynor_ratio"] *= TRADING_DAYS_PER_YEAR
    risk_adjusted["jensens_alpha"] *= TRADING_DAYS_PER_YEAR
    risk_adjusted["m2_measure"] *= TRADING_DAYS_PER_YEAR

    risk_decomp["total_variance"] *= TRADING_DAYS_PER_YEAR
    risk_decomp["systematic_variance"] *= TRADING_DAYS_PER_YEAR
    risk_decomp["unsystematic_variance"] *= TRADING_DAYS_PER_YEAR

    threshold_cleared = annualized_portfolio_return >= annualized_threshold_return

    return PortfolioAnalysisResponse(
        tickers=original_tickers,
        weights=[a.weight for a in request.assets],
        market_ticker=MARKET_TICKER,
        period_start=str(log_returns_df.index.min().date()),
        period_end=str(log_returns_df.index.max().date()),
        observations=int(n_obs),
        stale_return_pct=stale_return_pct,
        covariance_matrix=cov_matrix.to_dict(),
        asset_betas=asset_betas.loc[original_tickers].to_dict(),
        portfolio_variance=annualized_portfolio_variance,
        portfolio_volatility=annualized_portfolio_volatility,
        portfolio_beta=portfolio_beta,
        market_volatility=annualized_market_volatility,
        portfolio_expected_return=annualized_portfolio_return,
        market_expected_return=annualized_market_return,
        risk_decomposition=RiskDecomposition(**risk_decomp),
        risk_adjusted_metrics=RiskAdjustedMetrics(**risk_adjusted),
        roys_safety_first_ratio=annualized_sf_ratio,
        threshold_cleared=threshold_cleared,
    )


# --------------------------------------------------------------------------
# Server execution
# --------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
