"use client";

import { useId, useState } from "react";
import {
  Plus,
  Trash2,
  LoaderCircle,
  TriangleAlert,
  CircleCheckBig,
  CircleX,
} from "lucide-react";

type Asset = {
  id: string;
  ticker: string;
  weight: number;
};

type AnalysisResult = {
  observations: number;
  period_start: string;
  period_end: string;
  stale_return_pct: number;
  portfolio_variance: number;
  portfolio_volatility: number;
  portfolio_expected_return: number;
  market_volatility: number;
  risk_adjusted_metrics: {
    sharpe_ratio: number;
    treynor_ratio: number;
    jensens_alpha: number;
    m2_measure: number;
  };
  risk_decomposition: {
    portfolio_beta: number;
    pct_systematic: number;
  };
  roys_safety_first_ratio: number;
  threshold_cleared: boolean;
};

const API_URL = "http://127.0.0.1:8000/api/v1/analyze-portfolio";
const WEIGHT_TOLERANCE = 0.01;

const INITIAL_ASSETS: Asset[] = [
  { id: "COMI", ticker: "COMI", weight: 40 },
  { id: "TMGH", ticker: "TMGH", weight: 35 },
  { id: "HRHO", ticker: "HRHO", weight: 25 },
];

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined) return "N/A";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function ScoreCard({
  label,
  value,
  digits = 2,
  percent = false,
}: {
  label: string;
  value: number | null | undefined;
  digits?: number;
  percent?: boolean;
}) {
  const isKnown = value !== null && value !== undefined;
  const displayValue = isKnown && percent ? value * 100 : value;
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-neutral-400">
        {label}
      </p>
      <p className="mt-2 text-2xl font-semibold tabular-nums text-slate-900 dark:text-neutral-50">
        {formatNumber(displayValue, digits)}
        {isKnown && percent ? "%" : ""}
      </p>
    </div>
  );
}

export default function Home() {
  const idPrefix = useId();
  const [assets, setAssets] = useState<Asset[]>(INITIAL_ASSETS);
  const [riskFreeRate, setRiskFreeRate] = useState(18);
  const [thresholdReturn, setThresholdReturn] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const pctSystematic = result?.risk_decomposition?.pct_systematic;
  const systematicRiskPct = pctSystematic != null ? pctSystematic * 100 : undefined;
  const unsystematicRiskPct = pctSystematic != null ? (1 - pctSystematic) * 100 : undefined;

  const totalWeight = assets.reduce((sum, asset) => sum + (Number(asset.weight) || 0), 0);
  const weightIsValid = Math.abs(totalWeight - 100) < WEIGHT_TOLERANCE;
  const tickersAreValid = assets.length > 0 && assets.every((asset) => asset.ticker.trim().length > 0);
  const canSubmit = weightIsValid && tickersAreValid && !loading;
  const thresholdBelowRiskFree = thresholdReturn < riskFreeRate;

  function updateAsset(id: string, patch: Partial<Pick<Asset, "ticker" | "weight">>) {
    setAssets((prev) => prev.map((asset) => (asset.id === id ? { ...asset, ...patch } : asset)));
  }

  function addAsset() {
    setAssets((prev) => [...prev, { id: `${idPrefix}-${prev.length}-${Date.now()}`, ticker: "", weight: 0 }]);
  }

  function removeAsset(id: string) {
    setAssets((prev) => (prev.length > 1 ? prev.filter((asset) => asset.id !== id) : prev));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const payload = {
      assets: assets.map((asset) => ({
        ticker: asset.ticker.trim().toUpperCase(),
        weight: asset.weight / 100,
      })),
      risk_free_rate: riskFreeRate / 100,
      threshold_return: thresholdReturn / 100,
    };

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        let message = `Request failed with status ${res.status}.`;
        try {
          const body = await res.json();
          if (typeof body?.detail === "string") {
            message = body.detail;
          } else if (Array.isArray(body?.detail)) {
            // FastAPI/Pydantic 422 validation errors: detail is an array of
            // { msg, loc, type, ... } objects, not a string.
            const msgs = (body.detail as { msg?: string }[])
              .map((item) => item?.msg)
              .filter((m): m is string => Boolean(m));
            if (msgs.length > 0) {
              message = msgs.join(" ");
            }
          } else if (res.status === 400) {
            message = "API Error or weights do not sum to 100%.";
          }
        } catch {
          // response had no JSON body; fall back to the generic message above
        }
        throw new Error(message);
      }

      const data: AnalysisResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to reach the analysis service. Confirm the backend is running.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-neutral-950">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <header className="mb-8">
          <h1 className="text-2xl font-semibold text-slate-900 dark:text-neutral-50">
            Portfolio Risk Analytics
          </h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-neutral-400">
            Configure your portfolio and run a quantitative risk analysis against the EGX analytics engine.
          </p>
        </header>

        {error && (
          <div className="mb-8 flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
            <CircleX className="mt-0.5 h-5 w-5 flex-none" />
            <div>
              <p className="font-medium">Analysis failed</p>
              <p className="mt-0.5 text-red-700 dark:text-red-400">{error}</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          <section className="relative flex flex-col gap-6">
            {loading && (
              <div className="absolute inset-0 z-10 flex items-center justify-center rounded-xl bg-white/70 backdrop-blur-sm dark:bg-neutral-950/70">
                <LoaderCircle className="h-8 w-8 animate-spin text-slate-500 dark:text-neutral-400" />
              </div>
            )}

            <form onSubmit={handleSubmit} className="flex flex-col gap-6">
              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
                <h2 className="text-sm font-semibold text-slate-900 dark:text-neutral-50">
                  Global Parameters
                </h2>
                <div className="mt-4 grid grid-cols-2 gap-4">
                  <label className="flex flex-col gap-1.5 text-sm">
                    <span className="text-slate-600 dark:text-neutral-400">Risk-Free Rate (Rf) %</span>
                    <input
                      type="number"
                      step="0.01"
                      value={riskFreeRate}
                      onChange={(e) => setRiskFreeRate(Number(e.target.value))}
                      className="rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-900 outline-none focus:border-slate-500 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-50"
                    />
                  </label>
                  <label className="flex flex-col gap-1.5 text-sm">
                    <span className="text-slate-600 dark:text-neutral-400">Threshold Return (Rl) %</span>
                    <input
                      type="number"
                      step="0.01"
                      value={thresholdReturn}
                      onChange={(e) => setThresholdReturn(Number(e.target.value))}
                      className="rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-900 outline-none focus:border-slate-500 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-50"
                    />
                  </label>
                </div>
                {thresholdBelowRiskFree && (
                  <div className="mt-3 flex items-start gap-2 rounded-md bg-amber-50 p-2.5 text-xs text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
                    <TriangleAlert className="mt-0.5 h-3.5 w-3.5 flex-none" />
                    <span>Targeting a return below the risk-free rate is irrational — cash dominates the portfolio.</span>
                  </div>
                )}
              </div>

              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-slate-900 dark:text-neutral-50">
                    Asset Allocation
                  </h2>
                  <button
                    type="button"
                    onClick={addAsset}
                    className="flex items-center gap-1.5 rounded-md border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
                  >
                    <Plus className="h-3.5 w-3.5" />
                    Add Asset
                  </button>
                </div>

                <table className="mt-4 w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-neutral-400">
                      <th className="pb-2 font-medium">Ticker Name</th>
                      <th className="pb-2 font-medium">Weight %</th>
                      <th className="pb-2 font-medium sr-only">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {assets.map((asset) => (
                      <tr key={asset.id} className="border-t border-slate-100 dark:border-neutral-800">
                        <td className="py-2 pr-2">
                          <input
                            type="text"
                            value={asset.ticker}
                            onChange={(e) => updateAsset(asset.id, { ticker: e.target.value.toUpperCase() })}
                            placeholder="e.g. COMI"
                            className="w-full rounded-md border border-slate-300 bg-white px-3 py-1.5 uppercase text-slate-900 outline-none focus:border-slate-500 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-50"
                          />
                        </td>
                        <td className="py-2 pr-2">
                          <input
                            type="number"
                            step="0.01"
                            value={asset.weight}
                            onChange={(e) => updateAsset(asset.id, { weight: Number(e.target.value) })}
                            className="w-full rounded-md border border-slate-300 bg-white px-3 py-1.5 text-slate-900 outline-none focus:border-slate-500 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-50"
                          />
                        </td>
                        <td className="py-2 text-right">
                          <button
                            type="button"
                            onClick={() => removeAsset(asset.id)}
                            disabled={assets.length === 1}
                            className="rounded-md p-1.5 text-slate-400 hover:bg-slate-100 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-40 dark:hover:bg-neutral-800"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <div
                  className={`mt-3 text-xs font-medium ${
                    weightIsValid
                      ? "text-emerald-600 dark:text-emerald-400"
                      : "text-red-600 dark:text-red-400"
                  }`}
                >
                  Total weight: {formatNumber(totalWeight)}% {weightIsValid ? "" : "(must equal 100%)"}
                </div>
              </div>

              <button
                type="submit"
                disabled={!canSubmit}
                className="flex h-12 w-full items-center justify-center gap-2 rounded-full bg-slate-900 text-sm font-medium text-white transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-neutral-50 dark:text-neutral-900 dark:hover:bg-neutral-200"
              >
                {loading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
                Run Quantitative Analysis
              </button>
            </form>
          </section>

          {result && (
            <section className="flex flex-col gap-6">
              <p className="text-xs text-slate-500 dark:text-neutral-400">
                {result.observations} observations, {result.period_start} to {result.period_end}
              </p>

              {result.stale_return_pct > 0.05 && (
                <div className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800 dark:border-amber-900/50 dark:bg-amber-950/40 dark:text-amber-300">
                  <TriangleAlert className="mt-0.5 h-5 w-5 flex-none" />
                  <div>
                    <p className="font-medium">
                      {formatNumber(result.stale_return_pct * 100, 1)}% of return observations are stale
                    </p>
                    <p className="mt-0.5 text-amber-700 dark:text-amber-400">
                      Stale/non-trading prices deflate volatility and pull beta toward zero, so the risk figures below are understated.
                    </p>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 gap-4">
                <ScoreCard label="Sharpe Ratio" value={result.risk_adjusted_metrics?.sharpe_ratio} />
                <ScoreCard label="Treynor Ratio" value={result.risk_adjusted_metrics?.treynor_ratio} digits={4} />
                <ScoreCard label="Portfolio Beta" value={result.risk_decomposition?.portfolio_beta} />
                <ScoreCard label="Jensen's Alpha" value={result.risk_adjusted_metrics?.jensens_alpha} percent />
                <ScoreCard label="Portfolio Volatility" value={result.portfolio_volatility} percent />
                <ScoreCard label="Market Volatility" value={result.market_volatility} percent />
                <ScoreCard label="Expected Return" value={result.portfolio_expected_return} percent />
                <ScoreCard label="M² Measure" value={result.risk_adjusted_metrics?.m2_measure} percent />
              </div>

              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
                <h3 className="text-sm font-semibold text-slate-900 dark:text-neutral-50">
                  Risk Decomposition
                </h3>
                <div className="mt-3 flex justify-between text-sm">
                  <span className="text-slate-600 dark:text-neutral-400">Systematic Risk</span>
                  <span className="font-medium tabular-nums text-slate-900 dark:text-neutral-50">
                    {formatNumber(systematicRiskPct)}%
                  </span>
                </div>
                <div className="mt-1.5 flex justify-between text-sm">
                  <span className="text-slate-600 dark:text-neutral-400">Unsystematic Risk</span>
                  <span className="font-medium tabular-nums text-slate-900 dark:text-neutral-50">
                    {formatNumber(unsystematicRiskPct)}%
                  </span>
                </div>
              </div>

              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
                <h3 className="text-sm font-semibold text-slate-900 dark:text-neutral-50">
                  Shortfall Risk
                </h3>
                <div className="mt-3 flex justify-between text-sm">
                  <span className="text-slate-600 dark:text-neutral-400">Roy&apos;s Safety-First Ratio</span>
                  <span className="font-medium tabular-nums text-slate-900 dark:text-neutral-50">
                    {formatNumber(result.roys_safety_first_ratio, 4)}
                  </span>
                </div>
                <div
                  className={`mt-4 flex items-center gap-2 rounded-md p-3 text-sm font-medium ${
                    result.threshold_cleared
                      ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400"
                      : "bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-400"
                  }`}
                >
                  {result.threshold_cleared ? (
                    <CircleCheckBig className="h-4 w-4 flex-none" />
                  ) : (
                    <TriangleAlert className="h-4 w-4 flex-none" />
                  )}
                  {result.threshold_cleared
                    ? "Target threshold cleared"
                    : "Target threshold breached"}
                </div>
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
