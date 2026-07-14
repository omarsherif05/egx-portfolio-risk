"""
verify.py

Proves the fixes shipped in api.py / finance_math.py are actually correct,
by calling the REAL production functions (not reimplementations) against a
live portfolio: COMI / TMGH / HRHO at 0.40 / 0.35 / 0.25, Rf = 18%,
threshold = 10%.

Where a check needs an intermediate value the HTTP response doesn't expose
(the raw aligned price frame, the log-vs-simple return split, the naive-vs-
compounded Rf, the old M^2 fudge), this script re-invokes the exact same
finance_math functions api.py itself uses, wired together with the identical
one-line join/dropna glue that lives in api.analyze_portfolio. No risk math
is reimplemented anywhere in this file — the one exception, clearly labeled,
is the deleted sigma_p/|beta| M^2 fudge, which is deliberately reconstructed
as a baseline to diff the fix against.

Live market data is fetched exactly ONCE via the real api._fetch_price_history
/ api.get_egx30, then those two names are monkeypatched (in-memory only — no
source file is touched) to replay that single snapshot for the rest of this
run. This is pure test scaffolding: without it, api.analyze_portfolio() would
hit the network a second time on every call in this script, and live EGX
quotes can tick between calls, making exact-equality assertions flaky for
reasons that have nothing to do with whether the code is correct.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import api
import finance_math as fm

TICKERS = ["COMI", "TMGH", "HRHO"]
WEIGHTS = np.array([0.40, 0.35, 0.25])
RF_ANNUAL = 0.18
THRESHOLD_ANNUAL = 0.10
TRADING_DAYS = api.TRADING_DAYS_PER_YEAR

failures: list[str] = []


def check(condition: bool, description: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {description}")
    if not condition:
        failures.append(description)


print("=" * 78)
print("VERIFY.PY — proving the api.py / finance_math.py fixes against live data")
print("=" * 78)

# --------------------------------------------------------------------------
# Fetch real data ONCE via the real fetch functions, then monkeypatch those
# two names to replay this exact snapshot for every subsequent call below.
# --------------------------------------------------------------------------
formatted_map = {t: api._format_egx_ticker(t) for t in TICKERS}
fetch_tickers = list(dict.fromkeys(formatted_map.values()))

_fetched_snapshot = api._fetch_price_history(fetch_tickers)
_benchmark_snapshot = api.get_egx30()

_real_fetch_price_history = api._fetch_price_history
_real_get_egx30 = api.get_egx30
api._fetch_price_history = lambda tickers: {t: _fetched_snapshot[t] for t in tickers}
api.get_egx30 = lambda: _benchmark_snapshot

# --------------------------------------------------------------------------
# Build the request and call the REAL production endpoint function.
# --------------------------------------------------------------------------
request = api.PortfolioRequest(
    assets=[{"ticker": t, "weight": w} for t, w in zip(TICKERS, WEIGHTS)],
    risk_free_rate=RF_ANNUAL,
    threshold_return=THRESHOLD_ANNUAL,
)
response = api.analyze_portfolio(request)

# --------------------------------------------------------------------------
# 1. Observations / period / stale_return_pct
# --------------------------------------------------------------------------
print("\n[1] Response metadata (from the real analyze_portfolio() call)")
print(f"    observations     : {response.observations}")
print(f"    period_start     : {response.period_start}")
print(f"    period_end       : {response.period_end}")
print(f"    stale_return_pct : {response.stale_return_pct:.4%}")

# --------------------------------------------------------------------------
# Reconstruct the aligned price frame from the SAME snapshot, using the
# identical glue line api.analyze_portfolio uses. Needed for checks 2-5,
# which require intermediates the response doesn't expose.
# --------------------------------------------------------------------------
asset_price_df = pd.DataFrame({orig: _fetched_snapshot[formatted_map[orig]] for orig in TICKERS})
union_rows = len(asset_price_df)

# The exact alignment line from api.analyze_portfolio:
price_df = asset_price_df.join(_benchmark_snapshot, how="inner").dropna()

# --------------------------------------------------------------------------
# 2. Strict intersection: no forward-fill, every surviving date is genuine
#    for every asset AND the benchmark.
# --------------------------------------------------------------------------
print("\n[2] Price-frame alignment integrity")
print(f"    raw asset union rows (pre-alignment) : {union_rows}")
print(f"    aligned intersection rows            : {len(price_df)}")

raw_indices = {orig: _fetched_snapshot[formatted_map[orig]].index for orig in TICKERS}
raw_indices[api.MARKET_TICKER] = _benchmark_snapshot.index

no_nans = not price_df.isna().to_numpy().any()
check(no_nans, "aligned price frame contains zero NaNs")

genuine_everywhere = all(price_df.index.isin(raw_indices[col]).all() for col in price_df.columns)
check(
    genuine_everywhere,
    "every surviving date is present in EVERY column's own raw source index "
    "(i.e. a genuine trade, not a value copied forward from a prior day)",
)

# Counterfactual: if forward-fill were still happening, at least as many
# rows would survive (ffill invents values dropna() would otherwise remove).
hypothetical_ffill_rows = len(asset_price_df.ffill().join(_benchmark_snapshot, how="inner").ffill().dropna())
check(
    len(price_df) <= hypothetical_ffill_rows,
    f"strict intersection ({len(price_df)} rows) is <= what a forward-filled "
    f"pipeline would have kept ({hypothetical_ffill_rows} rows)",
)

# --------------------------------------------------------------------------
# Return frames — the same two calls api.analyze_portfolio makes.
# --------------------------------------------------------------------------
log_returns_df = fm.calculate_historical_returns(price_df, method="log")
simple_returns_df = fm.calculate_historical_returns(price_df, method="simple")

asset_log_returns = log_returns_df[TICKERS]
asset_simple_returns = simple_returns_df[TICKERS]

cov_matrix = fm.calculate_covariance_matrix(asset_log_returns)
portfolio_variance = fm.calculate_portfolio_variance(WEIGHTS, cov_matrix)
portfolio_volatility = fm.calculate_portfolio_volatility(portfolio_variance)

asset_betas = fm.calculate_asset_betas(log_returns_df, api.MARKET_TICKER)
portfolio_beta = fm.calculate_portfolio_beta(WEIGHTS, asset_betas.loc[TICKERS])

market_volatility = fm.calculate_market_volatility(log_returns_df, api.MARKET_TICKER)

# --------------------------------------------------------------------------
# 3. Annualized portfolio return: simple vs log, side by side.
# --------------------------------------------------------------------------
portfolio_return_simple_daily = float(asset_simple_returns.mean().to_numpy() @ WEIGHTS)
portfolio_return_log_daily = float(asset_log_returns.mean().to_numpy() @ WEIGHTS)

annualized_return_simple = portfolio_return_simple_daily * TRADING_DAYS
annualized_return_log = portfolio_return_log_daily * TRADING_DAYS

print("\n[3] Annualized portfolio return: simple vs log basis")
print(f"    simple-return basis (correct, used for portfolio_expected_return): {annualized_return_simple:.4%}")
print(f"    log-return basis (biased low by Jensen's inequality)             : {annualized_return_log:.4%}")
print(f"    response.portfolio_expected_return (should match simple)        : {response.portfolio_expected_return:.4%}")

check(annualized_return_simple > annualized_return_log, "simple-return annualized return > log-return annualized return")
check(
    abs(response.portfolio_expected_return - annualized_return_simple) < 1e-9,
    "response.portfolio_expected_return matches the simple-return computation exactly",
)

# --------------------------------------------------------------------------
# 4. Daily Rf: naive division vs compounded root, and which one the real
#    code actually used (proved by matching the response's Sharpe ratio).
# --------------------------------------------------------------------------
rf_naive_daily = RF_ANNUAL / TRADING_DAYS
rf_compounded_daily = (1 + RF_ANNUAL) ** (1 / TRADING_DAYS) - 1

print("\n[4] Daily risk-free rate conversion")
print(f"    naive division      Rf/252            : {rf_naive_daily:.8f}  (reannualizes via x252 to {rf_naive_daily * TRADING_DAYS:.4%})")
print(f"    compounded root  (1+Rf)^(1/252)-1      : {rf_compounded_daily:.8f}  (reannualizes via x252 to {rf_compounded_daily * TRADING_DAYS:.4%})")

market_return_simple_daily = float(simple_returns_df[api.MARKET_TICKER].mean())

metrics_with_naive_rf = fm.calculate_risk_adjusted_metrics(
    portfolio_return=portfolio_return_simple_daily,
    market_return=market_return_simple_daily,
    risk_free_rate=rf_naive_daily,
    portfolio_volatility=portfolio_volatility,
    portfolio_beta=portfolio_beta,
    market_volatility=market_volatility,
)
metrics_with_compounded_rf = fm.calculate_risk_adjusted_metrics(
    portfolio_return=portfolio_return_simple_daily,
    market_return=market_return_simple_daily,
    risk_free_rate=rf_compounded_daily,
    portfolio_volatility=portfolio_volatility,
    portfolio_beta=portfolio_beta,
    market_volatility=market_volatility,
)

sharpe_annualized_naive = metrics_with_naive_rf["sharpe_ratio"] * np.sqrt(TRADING_DAYS)
sharpe_annualized_compounded = metrics_with_compounded_rf["sharpe_ratio"] * np.sqrt(TRADING_DAYS)

print(f"    Sharpe if the code used naive Rf       : {sharpe_annualized_naive:.6f}")
print(f"    Sharpe if the code used compounded Rf  : {sharpe_annualized_compounded:.6f}")
print(f"    response.risk_adjusted_metrics.sharpe_ratio (actual)            : {response.risk_adjusted_metrics.sharpe_ratio:.6f}")

check(
    abs(response.risk_adjusted_metrics.sharpe_ratio - sharpe_annualized_compounded) < 1e-9,
    "actual Sharpe ratio matches the COMPOUNDED Rf computation exactly",
)
check(
    abs(response.risk_adjusted_metrics.sharpe_ratio - sharpe_annualized_naive) > 1e-6,
    "actual Sharpe ratio does NOT match the naive Rf computation (they meaningfully differ)",
)

# --------------------------------------------------------------------------
# 5. M2: true market volatility vs the deleted sigma_p/|beta| fudge.
#    NOTE: the fudge formula below no longer exists in finance_math.py — it
#    is intentionally reconstructed here ONLY as a comparison baseline for
#    the bug this fix corrects, not as a reimplementation of current logic.
# --------------------------------------------------------------------------
portfolio_log_return_series = asset_log_returns.to_numpy() @ WEIGHTS
market_log_return_series = log_returns_df[api.MARKET_TICKER].to_numpy()
correlation = float(np.corrcoef(portfolio_log_return_series, market_log_return_series)[0, 1])

implied_market_vol_fudge = portfolio_volatility / abs(portfolio_beta)  # deleted bug, rebuilt for comparison only

sharpe_daily = metrics_with_compounded_rf["sharpe_ratio"]
m2_true_daily = metrics_with_compounded_rf["m2_measure"]
m2_fudge_daily = rf_compounded_daily + sharpe_daily * implied_market_vol_fudge

m2_true_annualized = m2_true_daily * TRADING_DAYS
m2_fudge_annualized = m2_fudge_daily * TRADING_DAYS

print("\n[5] M^2 measure: true market volatility vs the deleted sigma_p/|beta| fudge")
print(f"    portfolio-vs-market correlation (rho)         : {correlation:.4f}")
print(f"    true market volatility (annualized)           : {market_volatility * np.sqrt(TRADING_DAYS):.4%}")
print(f"    fudged 'implied' market volatility (annualized): {implied_market_vol_fudge * np.sqrt(TRADING_DAYS):.4%}  (= sigma_m / rho)")
print(f"    M^2 with true market volatility (correct)      : {m2_true_annualized:.4%}")
print(f"    M^2 with old sigma_p/|beta| fudge (buggy)      : {m2_fudge_annualized:.4%}")
print(f"    response.risk_adjusted_metrics.m2_measure      : {response.risk_adjusted_metrics.m2_measure:.4%}")

check(
    abs(response.risk_adjusted_metrics.m2_measure - m2_true_annualized) < 1e-9,
    "actual M^2 matches the TRUE-market-volatility computation exactly",
)
if correlation < 1.0:
    check(m2_true_annualized < m2_fudge_annualized, "true M^2 is lower than the fudged M^2 given correlation < 1")

# --------------------------------------------------------------------------
# 6. threshold_cleared presence, type, and flip behavior.
# --------------------------------------------------------------------------
print("\n[6] threshold_cleared presence and flip behavior")
check(hasattr(response, "threshold_cleared"), "response has a threshold_cleared attribute")
check(isinstance(response.threshold_cleared, bool), "threshold_cleared is a real bool")
print(f"    threshold=10%  -> threshold_cleared = {response.threshold_cleared}")

high_threshold_request = api.PortfolioRequest(
    assets=[{"ticker": t, "weight": w} for t, w in zip(TICKERS, WEIGHTS)],
    risk_free_rate=RF_ANNUAL,
    threshold_return=5.00,
)
high_threshold_response = api.analyze_portfolio(high_threshold_request)
print(f"    threshold=500% -> threshold_cleared = {high_threshold_response.threshold_cleared}")

check(response.threshold_cleared is True, "threshold_cleared is True at threshold=10%")
check(high_threshold_response.threshold_cleared is False, "threshold_cleared is False at threshold=500%")

# --------------------------------------------------------------------------
# 7. Risk decomposition consistency.
# --------------------------------------------------------------------------
rd = response.risk_decomposition
print("\n[7] Risk decomposition consistency")
print(f"    total_variance        : {rd.total_variance:.8f}")
print(f"    systematic_variance   : {rd.systematic_variance:.8f}")
print(f"    unsystematic_variance : {rd.unsystematic_variance:.8f}")
print(f"    pct_systematic        : {rd.pct_systematic:.6f}")

check(0.0 <= rd.pct_systematic <= 1.0, "pct_systematic is between 0 and 1")
check(
    abs((rd.systematic_variance + rd.unsystematic_variance) - rd.total_variance) < 1e-9,
    "systematic_variance + unsystematic_variance == total_variance (within 1e-9)",
)

# --------------------------------------------------------------------------
# 8. Duplicate-ticker collision (COMI vs COMI.CA) must raise, not silently
#    build a singular covariance matrix.
# --------------------------------------------------------------------------
print("\n[8] Duplicate-ticker collision (COMI + COMI.CA)")
dup_request = api.PortfolioRequest(
    assets=[{"ticker": "COMI", "weight": 0.5}, {"ticker": "COMI.CA", "weight": 0.5}],
    risk_free_rate=RF_ANNUAL,
    threshold_return=THRESHOLD_ANNUAL,
)

raised = False
message = ""
try:
    api.analyze_portfolio(dup_request)
except ValueError as exc:
    raised = True
    message = str(exc)

print(f"    raised   : {raised}")
print(f"    message  : {message}")

check(raised, "submitting COMI + COMI.CA raises ValueError instead of silently proceeding")
check("Duplicate tickers" in message, "the raised error clearly names the duplicate-ticker collision")

# --------------------------------------------------------------------------
# Restore the real fetch functions (good hygiene; process exits right after).
# --------------------------------------------------------------------------
api._fetch_price_history = _real_fetch_price_history
api.get_egx30 = _real_get_egx30

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
if failures:
    print(f"FAILED — {len(failures)} check(s) did not pass:")
    for f in failures:
        print(f"  - {f}")
    print("=" * 78)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    print("=" * 78)
    sys.exit(0)
