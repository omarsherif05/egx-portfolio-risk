"""
finance_math.py

Production-grade portfolio risk analytics engine.

All statistical operations (returns, covariance, beta, risk decomposition)
are fully vectorized with NumPy/Pandas — no per-asset or per-period Python
loops are used for numerical computation, so the engine scales to an
arbitrary number of assets (N) and observations (T).
"""

from __future__ import annotations

from typing import Union, Sequence, Dict, List

import numpy as np
import pandas as pd

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _to_weight_array(weights: ArrayLike) -> np.ndarray:
    """Coerce arbitrary array-likes into a 1-D float64 NumPy weight vector.

    Raises:
        ValueError: if weights are empty, non-finite, or do not sum to ~1.0.
    """
    w = np.asarray(weights, dtype=np.float64).flatten()
    if w.size == 0:
        raise ValueError("Weights vector is empty.")
    if not np.all(np.isfinite(w)):
        raise ValueError("Weights contain NaN or infinite values.")
    if not np.isclose(w.sum(), 1.0, atol=1e-4):
        raise ValueError(f"Portfolio weights must sum to 1.0 (got {w.sum():.6f}).")
    return w


# --------------------------------------------------------------------------
# 1. Returns
# --------------------------------------------------------------------------

def calculate_historical_returns(price_df: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Calculate periodic asset returns from a price history.

    Mathematical definition:
        Simple (discrete) return:  R_t = (P_t / P_{t-1}) - 1
        Log (continuously compounded) return:  r_t = ln(P_t / P_{t-1})

    Args:
        price_df: DataFrame of shape (T+1, N) indexed by date/period, with one
            column per asset containing raw prices. Must be strictly positive.
        method: 'log' for continuously compounded returns, 'simple' (or
            'discrete') for simple percentage returns.

    Returns:
        DataFrame of shape (T, N) of periodic returns, first observation
        (which has no prior price to compare against) dropped.

    Raises:
        ValueError: on an unrecognized method, non-numeric data, or
            non-positive prices (which make log returns undefined).
    """
    if not isinstance(price_df, pd.DataFrame):
        raise TypeError("price_df must be a pandas DataFrame.")
    if price_df.empty:
        raise ValueError("price_df is empty.")

    prices = price_df.apply(pd.to_numeric, errors="coerce")
    if prices.isna().all(axis=None):
        raise ValueError("price_df contains no numeric data.")

    # Forward-fill isolated gaps (e.g. holidays/missing ticks) before diffing.
    prices = prices.ffill()

    if (prices.le(0)).any(axis=None):
        raise ValueError("Prices must be strictly positive to compute returns.")

    method = method.lower()
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method in ("simple", "discrete"):
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'log' or 'simple'.")

    return returns.dropna()


# --------------------------------------------------------------------------
# 2. Covariance
# --------------------------------------------------------------------------

def calculate_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the N x N sample covariance matrix of asset returns.

    Mathematical definition (unbiased / sample estimator, ddof = 1):
        Sigma_{i,j} = (1 / (T - 1)) * sum_t (r_{i,t} - mean(r_i)) * (r_{j,t} - mean(r_j))

    Args:
        returns_df: DataFrame of shape (T, N) of periodic asset returns.

    Returns:
        DataFrame of shape (N, N), indexed and columned by asset name,
        representing the sample covariance matrix.

    Raises:
        ValueError: if fewer than 2 observations are available (covariance
            is undefined with ddof=1 for T < 2).
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")
    clean = returns_df.dropna()
    if clean.shape[0] < 2:
        raise ValueError("At least 2 return observations are required to estimate covariance.")

    cov = clean.cov(ddof=1)  # vectorized pandas/NumPy covariance estimator
    return cov


# --------------------------------------------------------------------------
# 3 & 4. Portfolio variance / volatility
# --------------------------------------------------------------------------

def calculate_portfolio_variance(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """Compute total portfolio variance via vectorized matrix multiplication.

    Mathematical definition:
        sigma_p^2 = w^T * Sigma * w

    Args:
        weights: Length-N vector of portfolio weights (should sum to 1.0).
        cov_matrix: N x N covariance matrix (DataFrame or ndarray).

    Returns:
        Scalar portfolio variance.

    Raises:
        ValueError: if dimensions of weights and cov_matrix are incompatible.
    """
    w = _to_weight_array(weights)
    sigma = cov_matrix.to_numpy() if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix, dtype=np.float64)

    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("cov_matrix must be a square 2-D matrix.")
    if sigma.shape[0] != w.shape[0]:
        raise ValueError(
            f"Dimension mismatch: {w.shape[0]} weights vs {sigma.shape[0]}x{sigma.shape[1]} covariance matrix."
        )

    variance = float(w @ sigma @ w)
    if variance < 0:
        # Guards against numerically non-PSD covariance matrices.
        variance = max(variance, 0.0)
    return variance


def calculate_portfolio_volatility(portfolio_variance: float) -> float:
    """Return total portfolio risk (standard deviation).

    Mathematical definition:
        sigma_p = sqrt(sigma_p^2)

    Args:
        portfolio_variance: Non-negative portfolio variance.

    Returns:
        Portfolio volatility (standard deviation).

    Raises:
        ValueError: if portfolio_variance is negative.
    """
    if portfolio_variance < 0:
        raise ValueError("portfolio_variance cannot be negative.")
    return float(np.sqrt(portfolio_variance))


# --------------------------------------------------------------------------
# 5 & 6. Beta
# --------------------------------------------------------------------------

def calculate_asset_betas(returns_df: pd.DataFrame, market_col: str) -> pd.Series:
    """Calculate systematic risk (Beta) for every asset relative to a market benchmark.

    Mathematical definition:
        Beta_i = Cov(r_i, r_m) / Var(r_m)

    Args:
        returns_df: DataFrame of shape (T, N) including the market benchmark
            column among the assets.
        market_col: Column name in returns_df representing the market/benchmark.

    Returns:
        Series indexed by asset name (excluding the market column itself)
        with each asset's beta coefficient.

    Raises:
        KeyError: if market_col is not present in returns_df.
        ValueError: if market variance is zero (undefined beta).
    """
    if market_col not in returns_df.columns:
        raise KeyError(f"market_col '{market_col}' not found in returns_df columns.")

    clean = returns_df.dropna()
    market_var = clean[market_col].var(ddof=1)
    if market_var == 0 or not np.isfinite(market_var):
        raise ValueError("Market variance is zero or invalid; beta is undefined.")

    cov_matrix = clean.cov(ddof=1)
    asset_cols = [c for c in clean.columns if c != market_col]

    betas = cov_matrix.loc[asset_cols, market_col] / market_var
    betas.name = "beta"
    return betas


def calculate_portfolio_beta(weights: np.ndarray, asset_betas: pd.Series) -> float:
    """Calculate the weighted systematic risk (Beta) of the whole portfolio.

    Mathematical definition:
        Beta_p = sum_i (w_i * Beta_i)

    Args:
        weights: Length-N vector of portfolio weights (should sum to 1.0).
        asset_betas: Length-N Series of individual asset betas.

    Returns:
        Scalar portfolio beta.

    Raises:
        ValueError: if dimensions of weights and asset_betas are incompatible.
    """
    w = _to_weight_array(weights)
    betas = np.asarray(asset_betas, dtype=np.float64).flatten()

    if betas.shape[0] != w.shape[0]:
        raise ValueError(
            f"Dimension mismatch: {w.shape[0]} weights vs {betas.shape[0]} asset betas."
        )
    return float(w @ betas)


# --------------------------------------------------------------------------
# 7. Risk decomposition (Market / Single-Index Model)
# --------------------------------------------------------------------------

def decompose_risk(
    weights: np.ndarray,
    returns_df: pd.DataFrame,
    market_col: str,
    asset_order: List[str],
) -> Dict[str, float]:
    """Decompose total portfolio variance into systematic and unsystematic risk.

    Mathematical definition (Single-Index / Market Model):
        Total Variance:        sigma_p^2 = w^T * Sigma * w
        Systematic Variance:   sigma_sys^2 = Beta_p^2 * Var(r_m)
        Unsystematic Variance: sigma_unsys^2 = sigma_p^2 - sigma_sys^2  (residual risk)

    Args:
        weights: Length-N vector of portfolio weights, in the same order as
            `asset_order` (should sum to 1.0).
        returns_df: DataFrame of shape (T, N+1) including the market column.
        market_col: Column name representing the market/benchmark.
        asset_order: Explicit list of the N asset column names in returns_df,
            in the exact order `weights` corresponds to. This removes any
            dependency on DataFrame column-iteration order — without it, the
            weights vector and the beta/covariance vectors are only aligned
            by luck, and reordering columns silently produces a wrong beta.

    Returns:
        Dict with keys: 'total_variance', 'systematic_variance',
        'unsystematic_variance', 'portfolio_beta', and
        'pct_systematic' (fraction of total variance explained by the market).

    Raises:
        KeyError: if market_col is missing.
        ValueError: if len(asset_order) != len(weights), if any name in
            asset_order is missing from returns_df, or on non-positive
            market variance.
    """
    if market_col not in returns_df.columns:
        raise KeyError(f"market_col '{market_col}' not found in returns_df columns.")

    if len(asset_order) != len(weights):
        raise ValueError(
            f"asset_order length ({len(asset_order)}) must match weights length ({len(weights)})."
        )

    missing = [c for c in asset_order if c not in returns_df.columns]
    if missing:
        raise ValueError(f"asset_order names not found in returns_df columns: {missing}")

    clean = returns_df.dropna()

    asset_betas = calculate_asset_betas(clean, market_col)
    port_beta = calculate_portfolio_beta(weights, asset_betas.loc[asset_order])

    asset_cov = calculate_covariance_matrix(clean[asset_order])
    total_variance = calculate_portfolio_variance(weights, asset_cov)

    market_variance = float(clean[market_col].var(ddof=1))
    systematic_variance = (port_beta ** 2) * market_variance
    unsystematic_variance = max(total_variance - systematic_variance, 0.0)

    pct_systematic = (systematic_variance / total_variance) if total_variance > 0 else 0.0

    return {
        "total_variance": total_variance,
        "systematic_variance": systematic_variance,
        "unsystematic_variance": unsystematic_variance,
        "portfolio_beta": port_beta,
        "pct_systematic": pct_systematic,
    }


# --------------------------------------------------------------------------
# 7b. Market volatility
# --------------------------------------------------------------------------

def calculate_market_volatility(returns_df: pd.DataFrame, market_col: str) -> float:
    """Calculate the sample standard deviation of the market/benchmark returns.

    Mathematical definition (unbiased / sample estimator, ddof = 1):
        sigma_m = sqrt(Var(r_m))

    Args:
        returns_df: DataFrame of periodic returns including the market column.
        market_col: Column name representing the market/benchmark.

    Returns:
        Scalar market volatility (standard deviation).

    Raises:
        KeyError: if market_col is not present in returns_df.
        ValueError: if market variance is zero or non-finite.
    """
    if market_col not in returns_df.columns:
        raise KeyError(f"market_col '{market_col}' not found in returns_df columns.")

    market_returns = returns_df[market_col].dropna()
    market_variance = market_returns.var(ddof=1)
    if market_variance == 0 or not np.isfinite(market_variance):
        raise ValueError("Market variance is zero or invalid; market volatility is undefined.")

    return float(np.sqrt(market_variance))


# --------------------------------------------------------------------------
# 8. Risk-adjusted performance metrics
# --------------------------------------------------------------------------

def calculate_risk_adjusted_metrics(
    portfolio_return: float,
    market_return: float,
    risk_free_rate: float,
    portfolio_volatility: float,
    portfolio_beta: float,
    market_volatility: float,
) -> Dict[str, float]:
    """Calculate standard risk-adjusted performance metrics.

    Mathematical definitions:
        Sharpe Ratio   = (R_p - R_f) / sigma_p
        Treynor Ratio  = (R_p - R_f) / Beta_p
        Jensen's Alpha = (R_p - R_f) - Beta_p * (R_m - R_f)
        M^2 Measure    = R_f + Sharpe Ratio * sigma_m

    Args:
        portfolio_return: Realized/expected portfolio return (R_p).
        market_return: Realized/expected market return (R_m).
        risk_free_rate: Risk-free rate (R_f), same periodicity as the returns.
        portfolio_volatility: Portfolio standard deviation (sigma_p), must be > 0.
        portfolio_beta: Portfolio beta (Beta_p).
        market_volatility: Market/benchmark standard deviation (sigma_m), must
            be > 0. Derive this with `calculate_market_volatility` on the
            market's own return series. Do NOT approximate it as
            portfolio_volatility / abs(portfolio_beta) — since
            Beta_p = rho * sigma_p / sigma_m, that expression algebraically
            reduces to sigma_m / rho, silently dividing by the portfolio's
            correlation with the market and inflating sigma_m (and therefore
            M^2) whenever rho < 1.

    Returns:
        Dict with keys: 'sharpe_ratio', 'treynor_ratio', 'jensens_alpha',
        'm2_measure'.

    Raises:
        ValueError: if portfolio_volatility <= 0 or market_volatility <= 0.
    """
    if portfolio_volatility <= 0:
        raise ValueError("portfolio_volatility must be positive to compute Sharpe/M^2.")
    if market_volatility <= 0:
        raise ValueError("market_volatility must be positive to compute M^2.")

    excess_return = portfolio_return - risk_free_rate
    sharpe_ratio = excess_return / portfolio_volatility

    if portfolio_beta == 0:
        treynor_ratio = float("nan")
    else:
        treynor_ratio = excess_return / portfolio_beta

    jensens_alpha = excess_return - portfolio_beta * (market_return - risk_free_rate)

    m2_measure = risk_free_rate + sharpe_ratio * market_volatility

    return {
        "sharpe_ratio": sharpe_ratio,
        "treynor_ratio": treynor_ratio,
        "jensens_alpha": jensens_alpha,
        "m2_measure": m2_measure,
    }


# --------------------------------------------------------------------------
# 9. Roy's Safety-First criterion
# --------------------------------------------------------------------------

def calculate_roys_safety_first(
    portfolio_return: float, threshold_return: float, portfolio_volatility: float
) -> float:
    """Calculate Roy's Safety-First ratio (SFRatio).

    Mathematical definition:
        SFRatio = (R_p - R_L) / sigma_p

    Where R_L is the minimum acceptable ("threshold" / disaster level) return.
    A higher SFRatio implies a lower probability of the portfolio's return
    falling below the threshold (per Chebyshev/Roy's criterion).

    Args:
        portfolio_return: Expected portfolio return (R_p).
        threshold_return: Minimum acceptable return (R_L).
        portfolio_volatility: Portfolio standard deviation (sigma_p), must be > 0.

    Returns:
        Scalar SFRatio.

    Raises:
        ValueError: if portfolio_volatility <= 0.
    """
    if portfolio_volatility <= 0:
        raise ValueError("portfolio_volatility must be positive to compute SFRatio.")
    return float((portfolio_return - threshold_return) / portfolio_volatility)


# --------------------------------------------------------------------------
# Execution verification block
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("PORTFOLIO RISK ANALYTICS ENGINE — VERIFICATION RUN")
    print("=" * 70)

    # ---- Synthetic data generation: 3 assets + 1 market index, 100 days ----
    rng = np.random.default_rng(seed=42)
    n_days = 100
    dates = pd.bdate_range(start="2025-01-01", periods=n_days + 1)

    tickers = ["AssetA", "AssetB", "AssetC"]
    market_col = "MarketIndex"
    all_cols = tickers + [market_col]

    # Simulate daily log-return "shocks" then compound into a price path,
    # so implied returns are realistic and strictly positive.
    daily_vol = np.array([0.015, 0.020, 0.012, 0.010])  # per-asset + market
    daily_drift = np.array([0.0004, 0.0003, 0.0005, 0.0003])
    shocks = rng.normal(loc=daily_drift, scale=daily_vol, size=(n_days, 4))

    start_prices = np.array([100.0, 50.0, 200.0, 4000.0])
    log_price_paths = np.log(start_prices) + np.vstack([np.zeros(4), np.cumsum(shocks, axis=0)])
    price_array = np.exp(log_price_paths)

    price_df = pd.DataFrame(price_array, index=dates, columns=all_cols)

    print(f"\nGenerated synthetic price data: {price_df.shape[0]} rows x {price_df.shape[1]} columns")
    print(price_df.head(3))

    # ---- 1. Returns ----
    log_returns = calculate_historical_returns(price_df, method="log")
    simple_returns = calculate_historical_returns(price_df, method="simple")
    print(f"\n[1] Log returns computed: shape={log_returns.shape}")
    print(f"[1] Simple returns computed: shape={simple_returns.shape}")

    asset_returns = log_returns[tickers]
    market_returns = log_returns[market_col]

    # ---- 2. Covariance matrix ----
    cov_matrix = calculate_covariance_matrix(asset_returns)
    print("\n[2] Covariance matrix (assets only):")
    print(cov_matrix)

    # ---- Portfolio weights ----
    weights = np.array([0.4, 0.35, 0.25])
    print(f"\nPortfolio weights: {dict(zip(tickers, weights))}")

    # ---- 3 & 4. Portfolio variance / volatility ----
    port_variance = calculate_portfolio_variance(weights, cov_matrix)
    port_vol = calculate_portfolio_volatility(port_variance)
    print(f"\n[3] Portfolio variance (daily): {port_variance:.8f}")
    print(f"[4] Portfolio volatility (daily): {port_vol:.6f}")

    # ---- 5 & 6. Beta ----
    asset_betas = calculate_asset_betas(log_returns, market_col)
    print("\n[5] Asset betas vs MarketIndex:")
    print(asset_betas)

    port_beta = calculate_portfolio_beta(weights, asset_betas.loc[tickers])
    print(f"\n[6] Portfolio beta: {port_beta:.4f}")

    # ---- 7. Risk decomposition ----
    risk_decomp = decompose_risk(weights, log_returns, market_col, asset_order=tickers)
    print("\n[7] Risk decomposition:")
    for k, v in risk_decomp.items():
        print(f"    {k}: {v:.8f}" if isinstance(v, float) else f"    {k}: {v}")

    # ---- 7b. Market volatility ----
    market_vol = calculate_market_volatility(log_returns, market_col)
    print(f"\n[7b] Market volatility (daily): {market_vol:.6f}")

    # ---- 8. Risk-adjusted metrics ----
    annualization = 252
    portfolio_daily_return = float(asset_returns.mean() @ weights)
    market_daily_return = float(market_returns.mean())
    risk_free_daily = 0.02 / annualization  # 2% annual risk-free rate

    metrics = calculate_risk_adjusted_metrics(
        portfolio_return=portfolio_daily_return,
        market_return=market_daily_return,
        risk_free_rate=risk_free_daily,
        portfolio_volatility=port_vol,
        portfolio_beta=port_beta,
        market_volatility=market_vol,
    )
    print("\n[8] Risk-adjusted performance metrics (daily basis):")
    for k, v in metrics.items():
        print(f"    {k}: {v:.6f}")

    # ---- 9. Roy's Safety-First ratio ----
    threshold_return = 0.0  # capital preservation threshold
    sf_ratio = calculate_roys_safety_first(portfolio_daily_return, threshold_return, port_vol)
    print(f"\n[9] Roy's Safety-First Ratio (threshold={threshold_return}): {sf_ratio:.6f}")

    print("\n" + "=" * 70)
    print("ALL FUNCTIONS EXECUTED SUCCESSFULLY — NO EXCEPTIONS RAISED")
    print("=" * 70)
