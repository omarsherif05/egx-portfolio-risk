"use client";

import { useEffect, useId, useRef, useState } from "react";
import {
  Plus,
  Trash2,
  LoaderCircle,
  TriangleAlert,
  CircleCheckBig,
  CircleX,
  Wand2,
  Pencil,
  Copy,
  Check,
} from "lucide-react";
import { InfoTooltip, type InfoTooltipContent } from "./InfoTooltip";

type Asset = {
  id: string;
  ticker: string;
  weight: string;
};

type AnalysisResult = {
  tickers: string[];
  weights: number[];
  observations: number;
  period_start: string;
  period_end: string;
  stale_return_pct: number;
  portfolio_variance: number;
  portfolio_volatility: number;
  portfolio_expected_return: number;
  market_volatility: number;
  asset_betas: Record<string, number>;
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

type RunInputs = {
  assets: Asset[];
  riskFreeRate: string;
  thresholdReturn: string;
};

type BetaRow = { ticker: string; weight: number; beta: number };

const API_URL = `${process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"}/api/v1/analyze-portfolio`;
const WEIGHT_TOLERANCE = 0.01;
const REQUEST_TIMEOUT_MS = 30_000;
const EGX_SUFFIX = ".CA";

const LOADING_STEPS = ["Fetching price history…", "Aligning trading calendars…", "Computing risk metrics…"];

const INITIAL_ASSETS: Asset[] = [
  { id: "COMI", ticker: "COMI", weight: "40" },
  { id: "TMGH", ticker: "TMGH", weight: "35" },
  { id: "HRHO", ticker: "HRHO", weight: "25" },
];

const METRIC_TOOLTIPS: Record<string, InfoTooltipContent> = {
  sharpe: {
    title: "Sharpe Ratio",
    definition:
      "How much return you earn for every unit of total risk you take. It is the most common way to compare two portfolios that carry different levels of risk.",
    formula: "Sharpe = (Rₚ − Rf) / σₚ",
    reading:
      "Higher is better. Above 1.0 is generally considered good. It uses TOTAL risk, so it penalises you for stock-specific risk you could have diversified away.",
  },
  treynor: {
    title: "Treynor Ratio",
    definition:
      "Return earned per unit of MARKET risk only. It ignores the risk that comes from picking individual stocks, on the assumption you have diversified that away.",
    formula: "Treynor = (Rₚ − Rf) / βₚ",
    reading:
      "Higher is better. Compare it with Sharpe: if Treynor looks good but Sharpe looks weak, you are carrying a lot of undiversified stock-specific risk.",
  },
  beta: {
    title: "Portfolio Beta",
    definition:
      "How much your portfolio moves when the market moves. A beta of 1.0 means it moves in step with the EGX30.",
    formula: "βₚ = Σ wᵢβᵢ, where βᵢ = Cov(Rᵢ, Rm) / σ²ₘ",
    reading: "Above 1.0 = more volatile than the index. Below 1.0 = more defensive. Beta measures only market risk, not total risk.",
  },
  alpha: {
    title: "Jensen's Alpha",
    definition:
      "How much you beat (or missed) the return that CAPM says you SHOULD have earned, given how much market risk you took. This is the closest thing to a measure of skill.",
    formula: "αₚ = Rₚ − [Rf + βₚ(Rm − Rf)]",
    reading: "Positive means you earned more than your risk level justified. This is backward-looking — past alpha does not predict future alpha.",
  },
  m2: {
    title: "M² Measure (Modigliani–Modigliani)",
    definition:
      "What your return WOULD have been if your portfolio had been adjusted to carry exactly the same risk as the market. It lets you compare your portfolio against the index on equal footing.",
    formula: "M² = (Rₚ − Rf) × (σₘ / σₚ) + Rf",
    reading:
      "Compare it directly against the market's own return. If your M² is below your raw return, part of your outperformance came from simply taking more risk, not from better selection.",
  },
  roysSafetyFirst: {
    title: "Roy's Safety-First Ratio",
    definition:
      "How many standard deviations your expected return sits above the minimum return you said you could tolerate. It measures the risk of falling short of your target.",
    formula: "SFRatio = (E(Rₚ) − RL) / σₚ",
    reading: "Higher is better. If returns are normally distributed, the probability of falling below your threshold is approximately N(−SFRatio).",
  },
  portfolioVolatility: {
    title: "Portfolio Volatility",
    definition: "Total risk. How much your portfolio's value swings around, up or down, in a typical year.",
    formula: "σₚ = √(wᵀΣw)",
    reading: "A 30% volatility means that in roughly two years out of three, your annual return lands within ±30% of its average.",
  },
  marketVolatility: {
    title: "Market Volatility",
    definition: "The same measure of total risk, but for the EGX30 index itself. Your benchmark for whether your portfolio is unusually risky.",
    formula: "σₘ = √(Var(Rm))",
    reading: "Compare against your portfolio volatility. If yours is higher, you are taking more total risk than simply buying the index.",
  },
  expectedReturn: {
    title: "Expected Return",
    definition: "The average annual return of your portfolio over the historical period analysed, using each holding's weight.",
    formula: "E(Rₚ) = Σ wᵢE(Rᵢ)",
    reading: "This is a historical average, not a forecast. It says what happened, not what will happen.",
  },
  systematicRisk: {
    title: "Systematic Risk",
    definition: "The share of your risk that comes from the whole market moving. You cannot diversify this away — it is the price of being invested at all.",
    formula: "σ²_systematic = β²ₚ × σ²ₘ",
    reading: "The only way to reduce this is to hold less equity, not to hold different equities.",
  },
  unsystematicRisk: {
    title: "Unsystematic Risk",
    definition: "The share of your risk that comes from the specific companies you picked. Adding more, less-correlated holdings reduces it.",
    formula: "σ²_unsystematic = σ²ₚ − β²ₚσ²ₘ",
    reading:
      "A high number means your portfolio is concentrated. This is the risk you are NOT compensated for taking, because you could have removed it for free by diversifying.",
  },
  riskFreeRateInput: {
    title: "Risk-Free Rate",
    definition:
      "The return you could earn with essentially no risk at all. It is the baseline every risky investment has to beat — if a stock portfolio can't outperform it, you were never being paid for the risk you took.",
    reading:
      "What to enter: In Egypt, use the yield on short-term Egyptian Treasury bills (T-bills), issued by the Ministry of Finance and auctioned weekly by the Central Bank. As of recent years these have ranged roughly between 18% and 27% depending on the rate cycle. Enter it as an annual percentage — for example, 18 for an 18% T-bill yield. Why it matters: Rf sits inside Sharpe, Treynor, Jensen's Alpha, and M². Egypt's risk-free rate is unusually high by global standards — an 18% T-bill is a serious hurdle that a US investor comparing against a 4% Treasury never has to clear. A portfolio returning 20% a year is impressive in New York and merely adequate in Cairo.",
  },
  thresholdReturnInput: {
    title: "Threshold Return",
    definition:
      "The minimum return you can live with. Not your target, and not your hope — the floor below which the outcome counts as a failure for you.",
    reading:
      'What to enter: Whatever a bad year actually means for you, as an annual percentage. Two common ways to set it: your risk-free rate, meaning "if I can\'t beat T-bills, I should have just bought T-bills"; or Egypt\'s inflation rate, meaning "if I don\'t beat inflation, I lost purchasing power even though the number went up." Why it matters: This is the only input on this page that is a personal judgement rather than a market fact. It feeds Roy\'s Safety-First Ratio, which measures how far your expected return sits above this floor and estimates the chance of falling below it. Setting the threshold BELOW your risk-free rate is irrational — it would mean accepting less than cash pays — which is why this page warns you when you do.',
  },
};

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined) return "N/A";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

// Empty string coerces to 0 under Number(), which would silently treat a
// blank/incomplete field as a valid zero. Callers must be able to tell
// "no value yet" apart from "the value is zero".
function parseNumeric(raw: string): number | null {
  if (raw.trim() === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

// Strips leading zeros from the integer part only (on blur, never while
// typing) so "035.5" -> "35.5" and "00" -> "0", while a legitimate leading
// zero before a decimal point ("0.5", "0.05") is left untouched.
function normalizeNumericInput(raw: string): string {
  if (raw.trim() === "") return raw;

  const isNegative = raw.startsWith("-");
  let value = isNegative ? raw.slice(1) : raw;

  if (value === "" || value === ".") {
    if (value === ".") value = "0.";
    return (isNegative ? "-" : "") + value;
  }

  const [intPart, ...rest] = value.split(".");
  let strippedInt = intPart.replace(/^0+(?=\d)/, "");
  if (strippedInt === "") strippedInt = "0";

  const normalized = rest.length > 0 ? `${strippedInt}.${rest.join(".")}` : strippedInt;
  return (isNegative ? "-" : "") + normalized;
}

// Mirrors api.py's _format_egx_ticker exactly, so collisions like
// "COMI" + "COMI.CA" are caught client-side before a network round-trip.
function formatEgxTicker(rawTicker: string): string {
  const normalized = rawTicker.trim().toUpperCase();
  if (normalized.startsWith("^") || normalized.endsWith(EGX_SUFFIX)) return normalized;
  return `${normalized}${EGX_SUFFIX}`;
}

// --------------------------------------------------------------------------
// "What this means" interpretation panel — pure, deterministic, fixed rules
// applied to an already-returned AnalysisResult. No LLM, no external call.
// --------------------------------------------------------------------------

// Standard normal CDF via the Abramowitz & Stegun erf approximation
// (max error ~1.5e-7) — used only for an approximate shortfall probability.
function normalCdf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * absX);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) *
      Math.exp(-absX * absX);
  return 0.5 * (1 + sign * y);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function marketExposureSentence(result: AnalysisResult): string | null {
  const beta = result.risk_decomposition?.portfolio_beta;
  if (!isFiniteNumber(beta)) return null;

  if (beta > 1.15) {
    return `Your portfolio amplifies the market. If the EGX30 falls 10%, your portfolio would be expected to fall about ${formatNumber(beta * 10, 1)}%.`;
  }
  if (beta >= 0.85) {
    return `Your portfolio moves roughly in step with the EGX30. A 10% move in the index implies about a ${formatNumber(beta * 10, 1)}% move for you.`;
  }
  if (beta > 0) {
    return `Your portfolio is defensive. It absorbs only about ${formatNumber(beta * 100, 1)}% of the market's moves.`;
  }
  return "Your portfolio moves against the market — unusual, and worth checking your holdings are what you intended.";
}

function riskLevelSentence(result: AnalysisResult): string | null {
  const portfolioVol = result.portfolio_volatility;
  const marketVol = result.market_volatility;
  if (!isFiniteNumber(portfolioVol) || !isFiniteNumber(marketVol) || marketVol <= 0) return null;

  const ratio = portfolioVol / marketVol;
  if (ratio > 1.1) {
    return `You carry ${formatNumber((ratio - 1) * 100, 1)}% more total risk than simply holding the index.`;
  }
  if (ratio >= 0.9) {
    return "Your total risk is close to the index's own.";
  }
  return `You carry ${formatNumber((1 - ratio) * 100, 1)}% less total risk than the index.`;
}

function riskCompositionSentence(result: AnalysisResult): string | null {
  const pctSystematic = result.risk_decomposition?.pct_systematic;
  if (!isFiniteNumber(pctSystematic)) return null;

  const pct = pctSystematic * 100;
  if (pct > 75) {
    return `${formatNumber(pct, 1)}% of your risk is market risk you cannot diversify away. Your portfolio behaves much like the index itself.`;
  }
  if (pct >= 40) {
    return `${formatNumber(pct, 1)}% of your risk is market risk. The remaining ${formatNumber(100 - pct, 1)}% comes from the specific companies you hold, and would fall if you added more, less-correlated names.`;
  }
  return `Only ${formatNumber(pct, 1)}% of your risk is market risk. The other ${formatNumber(100 - pct, 1)}% is stock-specific — your portfolio is concentrated, and that risk is not compensated: diversification would remove it at no cost to expected return.`;
}

function concentrationSentences(result: AnalysisResult): string[] {
  const tickers = result.tickers;
  const weights = result.weights;
  if (!Array.isArray(tickers) || !Array.isArray(weights) || tickers.length !== weights.length || tickers.length === 0) {
    return [];
  }

  const holdings = tickers.map((ticker, i) => ({
    ticker,
    weight: weights[i],
    beta: result.asset_betas?.[ticker],
  }));

  const sentences: string[] = [];

  const dominant = holdings.find((h) => isFiniteNumber(h.weight) && h.weight > 0.5);
  if (dominant) {
    sentences.push(
      `${dominant.ticker} alone is ${formatNumber(dominant.weight * 100, 1)}% of the portfolio and drives most of what you see above.`,
    );
  }

  const withBeta = holdings.filter((h): h is { ticker: string; weight: number; beta: number } => isFiniteNumber(h.beta));
  if (withBeta.length > 0) {
    const highestBeta = withBeta.reduce((max, h) => (h.beta > max.beta ? h : max));
    sentences.push(
      `${highestBeta.ticker} is your highest-beta holding at ${formatNumber(highestBeta.beta, 2)}, contributing most to your market sensitivity.`,
    );
  }

  return sentences;
}

function performanceSentence(result: AnalysisResult): string | null {
  const alpha = result.risk_adjusted_metrics?.jensens_alpha;
  const observations = result.observations;
  if (!isFiniteNumber(alpha) || !isFiniteNumber(observations)) return null;

  const alphaPct = alpha * 100;
  if (alphaPct > 2) {
    return `You beat what CAPM predicts for your level of market risk by ${formatNumber(alphaPct, 2)}% a year. This is a historical result over ${observations} trading days, not a forecast.`;
  }
  if (alphaPct >= -2) {
    return "Your return is close to what CAPM predicts for your level of market risk — no meaningful historical alpha.";
  }
  return `You fell short of what CAPM predicts for your level of market risk by ${formatNumber(Math.abs(alphaPct), 2)}% a year.`;
}

// The single most important sentence in the panel: whether raw outperformance
// survives being risk-adjusted to the index's own volatility.
function riskAdjustedRealityCheckSentence(result: AnalysisResult): string | null {
  const m2 = result.risk_adjusted_metrics?.m2_measure;
  const expectedReturn = result.portfolio_expected_return;
  if (!isFiniteNumber(m2) || !isFiniteNumber(expectedReturn)) return null;

  const m2Pct = m2 * 100;
  const returnPct = expectedReturn * 100;
  if (m2 < expectedReturn) {
    return `Adjusted to the index's risk level, your ${formatNumber(returnPct, 2)}% return becomes ${formatNumber(m2Pct, 2)}%. The gap is the part of your return that came from taking more risk rather than from selection.`;
  }
  return `Even adjusted to the index's risk level, your return holds at ${formatNumber(m2Pct, 2)}%. Your result is not explained by simply taking more risk.`;
}

// treynor_implied = sharpe_ratio * (portfolio_volatility / portfolio_beta) is
// the conceptual motivation for this sentence (the Treynor you'd see if all
// risk were market risk) but, per the specified copy, isn't itself quoted.
function sharpeVsTreynorSentence(result: AnalysisResult): string | null {
  const sharpe = result.risk_adjusted_metrics?.sharpe_ratio;
  const pctSystematic = result.risk_decomposition?.pct_systematic;
  if (!isFiniteNumber(sharpe) || !isFiniteNumber(pctSystematic)) return null;
  if (!(sharpe > 1.0 && pctSystematic < 0.5)) return null;

  return `Your Sharpe ratio of ${formatNumber(sharpe, 2)} is healthy, but half your risk is stock-specific. Sharpe charges you for that risk; Treynor does not. The gap between them is the cost of being under-diversified.`;
}

function shortfallSentence(result: AnalysisResult): string | null {
  const sfr = result.roys_safety_first_ratio;
  const expectedReturn = result.portfolio_expected_return;
  const portfolioVol = result.portfolio_volatility;
  if (!isFiniteNumber(sfr) || !isFiniteNumber(expectedReturn) || !isFiniteNumber(portfolioVol)) return null;

  // Roy's SFRatio = (E(Rp) - R_L) / σp, all annualized, so the threshold
  // itself is reconstructed exactly from fields already in the response —
  // no backend change needed to expose it separately.
  const impliedThreshold = expectedReturn - sfr * portfolioVol;
  const shortfallProbabilityPct = normalCdf(-sfr) * 100;

  return `Your expected return sits ${formatNumber(sfr, 2)} standard deviations above your ${formatNumber(impliedThreshold * 100, 2)}% threshold. Assuming returns are normally distributed, that implies roughly a ${formatNumber(shortfallProbabilityPct, 1)}% chance of finishing a year below it — though real equity returns have fatter tails than the normal distribution, so treat that as a floor, not a guarantee.`;
}

function dataQualitySentence(result: AnalysisResult): string | null {
  const stalePct = result.stale_return_pct;
  if (!isFiniteNumber(stalePct) || stalePct <= 0.03) return null;

  return `${formatNumber(stalePct * 100, 1)}% of the return observations are exactly zero, meaning the price did not move — typically a non-trading or halted day on the EGX. Stale prices mechanically understate volatility and pull beta toward zero, so the risk figures above are conservative.`;
}

function sampleSizeSentence(result: AnalysisResult): string | null {
  const observations = result.observations;
  if (!isFiniteNumber(observations) || observations >= 250) return null;

  return `This analysis rests on ${observations} trading days (${result.period_start} to ${result.period_end}), under a year of data. Estimates from short samples are unstable.`;
}

function buildInterpretation(result: AnalysisResult): string[] {
  return [
    marketExposureSentence(result),
    riskLevelSentence(result),
    riskCompositionSentence(result),
    ...concentrationSentences(result),
    performanceSentence(result),
    riskAdjustedRealityCheckSentence(result),
    sharpeVsTreynorSentence(result),
    shortfallSentence(result),
    dataQualitySentence(result),
    sampleSizeSentence(result),
  ].filter((sentence): sentence is string => Boolean(sentence));
}

function ScoreCard({
  label,
  value,
  digits = 2,
  percent = false,
  tooltip,
  emphasis = false,
}: {
  label: string;
  value: number | null | undefined;
  digits?: number;
  percent?: boolean;
  tooltip?: InfoTooltipContent;
  emphasis?: boolean;
}) {
  const isKnown = value !== null && value !== undefined;
  const displayValue = isKnown && percent ? value * 100 : value;
  return (
    <div
      className={`rounded-xl border bg-white p-5 shadow-sm dark:bg-neutral-900 ${
        emphasis ? "border-slate-300 dark:border-neutral-700" : "border-slate-200 dark:border-neutral-800"
      }`}
    >
      <p className="flex items-center text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-neutral-400">
        {label}
        {tooltip && <InfoTooltip content={tooltip} />}
      </p>
      <p
        className={`mt-2 tabular-nums text-slate-900 dark:text-neutral-50 ${
          emphasis ? "text-3xl font-bold" : "text-2xl font-semibold"
        }`}
      >
        {formatNumber(displayValue, digits)}
        {isKnown && percent ? "%" : ""}
      </p>
    </div>
  );
}

function pctText(value: number | null | undefined, digits = 2): string {
  if (!isFiniteNumber(value)) return "N/A";
  return `${formatNumber(value * 100, digits)}%`;
}

// Plain-text summary for the "Copy results" button — no markdown, so it
// pastes cleanly into an email or document. Padded labels give a readable
// column alignment in a monospace or proportional font alike.
function buildResultsSummaryText(
  result: AnalysisResult,
  riskFreeRatePct: number,
  thresholdReturnPct: number,
  systematicRiskPct: number | undefined,
  unsystematicRiskPct: number | undefined,
  interpretation: string[],
): string {
  const pad = (label: string) => label.padEnd(26, " ");
  const lines: string[] = [];

  lines.push("PORTFOLIO RISK ANALYSIS");
  lines.push("");
  lines.push("Portfolio");
  result.tickers.forEach((ticker, i) => {
    lines.push(`  ${pad(ticker)}${formatNumber((result.weights[i] ?? 0) * 100, 2)}%`);
  });
  lines.push("");
  lines.push("Parameters");
  lines.push(`  ${pad("Risk-Free Rate (Rf)")}${formatNumber(riskFreeRatePct, 2)}%`);
  lines.push(`  ${pad("Threshold Return (Rl)")}${formatNumber(thresholdReturnPct, 2)}%`);
  lines.push("");
  lines.push("Data");
  lines.push(`  ${pad("Observations")}${result.observations}`);
  lines.push(`  ${pad("Period")}${result.period_start} to ${result.period_end}`);
  lines.push("");
  lines.push("Metrics");
  lines.push(`  ${pad("Sharpe Ratio")}${formatNumber(result.risk_adjusted_metrics?.sharpe_ratio)}`);
  lines.push(`  ${pad("Treynor Ratio")}${formatNumber(result.risk_adjusted_metrics?.treynor_ratio, 4)}`);
  lines.push(`  ${pad("Portfolio Beta")}${formatNumber(result.risk_decomposition?.portfolio_beta)}`);
  lines.push(`  ${pad("Jensen's Alpha")}${pctText(result.risk_adjusted_metrics?.jensens_alpha)}`);
  lines.push(`  ${pad("M2 Measure")}${pctText(result.risk_adjusted_metrics?.m2_measure)}`);
  lines.push(`  ${pad("Expected Return")}${pctText(result.portfolio_expected_return)}`);
  lines.push(`  ${pad("Portfolio Volatility")}${pctText(result.portfolio_volatility)}`);
  lines.push(`  ${pad("Market Volatility")}${pctText(result.market_volatility)}`);
  lines.push(`  ${pad("Systematic Risk")}${formatNumber(systematicRiskPct)}%`);
  lines.push(`  ${pad("Unsystematic Risk")}${formatNumber(unsystematicRiskPct)}%`);
  lines.push(`  ${pad("Roy's Safety-First Ratio")}${formatNumber(result.roys_safety_first_ratio, 4)}`);
  lines.push(`  ${pad("Threshold Cleared")}${result.threshold_cleared ? "Yes" : "No"}`);
  lines.push("");
  lines.push("What this means");
  interpretation.forEach((sentence) => lines.push(`  ${sentence}`));

  return lines.join("\n");
}

// navigator.clipboard requires a secure context and can be blocked outright
// in some browsers; execCommand is a deprecated but still-working fallback
// for the same user gesture.
async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
    throw new Error("Clipboard API unavailable");
  } catch {
    try {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      const succeeded = document.execCommand("copy");
      document.body.removeChild(textarea);
      return succeeded;
    } catch {
      return false;
    }
  }
}

// Mounted fresh each time the loading overlay appears (via its parent's
// `{loading && ...}` conditional), so its own step index naturally starts
// at 0 without needing an effect to reset state imperatively.
function LoadingStepLabel() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => {
      setStep((prev) => (prev < LOADING_STEPS.length - 1 ? prev + 1 : prev));
    }, 800);
    return () => clearInterval(interval);
  }, []);
  return <p className="text-sm font-medium text-slate-500 dark:text-neutral-400">{LOADING_STEPS[step]}</p>;
}

function EmptyResultsState() {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
      <h3 className="text-sm font-semibold text-slate-900 dark:text-neutral-50">What this engine does</h3>
      <div className="mt-3 flex flex-col gap-2 text-sm leading-relaxed text-slate-500 dark:text-neutral-400">
        <p>Fetches three years of daily prices for your holdings from Yahoo Finance, and the EGX30 index from Investing.com.</p>
        <p>Aligns both onto a strict shared trading calendar — no forward-filling, no invented prices.</p>
        <p>Computes covariance, beta, and CAPM risk-adjusted performance, and tells you in plain English what the numbers mean.</p>
      </div>
    </div>
  );
}

// Renders the full results block. Takes `isCollapsed` purely to choose
// grid arrangements — the underlying cards are identical either way, so
// this stays a single implementation instead of two diverging copies.
function ResultsSection({
  result,
  betaRows,
  systematicRiskPct,
  unsystematicRiskPct,
  interpretation,
  resultsAreStale,
  isCollapsed,
  riskFreeRatePct,
  thresholdReturnPct,
}: {
  result: AnalysisResult;
  betaRows: BetaRow[];
  systematicRiskPct: number | undefined;
  unsystematicRiskPct: number | undefined;
  interpretation: string[];
  resultsAreStale: boolean;
  isCollapsed: boolean;
  riskFreeRatePct: number;
  thresholdReturnPct: number;
}) {
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");

  async function handleCopy() {
    const text = buildResultsSummaryText(
      result,
      riskFreeRatePct,
      thresholdReturnPct,
      systematicRiskPct,
      unsystematicRiskPct,
      interpretation,
    );
    const succeeded = await copyTextToClipboard(text);
    setCopyState(succeeded ? "copied" : "failed");
    setTimeout(() => setCopyState("idle"), 2000);
  }

  const scoreCardGridClass = isCollapsed
    ? "grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4"
    : "grid grid-cols-1 gap-4 sm:grid-cols-2";

  const riskDecompositionCard = (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
      <h3 className="text-sm font-semibold text-slate-900 dark:text-neutral-50">Risk Decomposition</h3>
      <div className="mt-3 flex justify-between text-sm">
        <span className="flex items-center text-slate-600 dark:text-neutral-400">
          Systematic Risk
          <InfoTooltip content={METRIC_TOOLTIPS.systematicRisk} />
        </span>
        <span className="font-medium tabular-nums text-slate-900 dark:text-neutral-50">
          {formatNumber(systematicRiskPct)}%
        </span>
      </div>
      <div className="mt-1.5 flex justify-between text-sm">
        <span className="flex items-center text-slate-600 dark:text-neutral-400">
          Unsystematic Risk
          <InfoTooltip content={METRIC_TOOLTIPS.unsystematicRisk} />
        </span>
        <span className="font-medium tabular-nums text-slate-900 dark:text-neutral-50">
          {formatNumber(unsystematicRiskPct)}%
        </span>
      </div>
    </div>
  );

  const assetBetasCard =
    betaRows.length > 0 ? (
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-neutral-50">Asset Betas</h3>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[280px] text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-neutral-400">
                <th className="pb-2 font-medium">Ticker</th>
                <th className="pb-2 font-medium text-right">Weight %</th>
                <th className="pb-2 font-medium text-right">Beta</th>
              </tr>
            </thead>
            <tbody>
              {betaRows.map((row) => (
                <tr key={row.ticker} className="border-t border-slate-100 dark:border-neutral-800">
                  <td className="py-1.5 font-medium text-slate-900 dark:text-neutral-50">{row.ticker}</td>
                  <td className="py-1.5 text-right tabular-nums text-slate-600 dark:text-neutral-400">
                    {formatNumber(row.weight)}%
                  </td>
                  <td className="py-1.5 text-right tabular-nums font-medium text-slate-900 dark:text-neutral-50">
                    {formatNumber(row.beta, 2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    ) : null;

  const shortfallRiskCard = (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
      <h3 className="text-sm font-semibold text-slate-900 dark:text-neutral-50">Shortfall Risk</h3>
      <div className="mt-3 flex justify-between text-sm">
        <span className="flex items-center text-slate-600 dark:text-neutral-400">
          Roy&apos;s Safety-First Ratio
          <InfoTooltip content={METRIC_TOOLTIPS.roysSafetyFirst} />
        </span>
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
        {result.threshold_cleared ? "Target threshold cleared" : "Target threshold breached"}
      </div>
    </div>
  );

  const whatThisMeansCard =
    interpretation.length > 0 ? (
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-neutral-50">What this means</h3>
        <div
          className={`mt-3 divide-y divide-slate-100 dark:divide-neutral-800 ${
            isCollapsed ? "columns-1 gap-8 lg:columns-2" : ""
          }`}
        >
          {interpretation.map((sentence, i) => (
            <p
              key={i}
              className="break-inside-avoid py-3 text-sm leading-relaxed text-slate-600 first:pt-0 last:pb-0 dark:text-neutral-400"
            >
              {sentence}
            </p>
          ))}
        </div>
      </div>
    ) : null;

  return (
    <section className="flex flex-col gap-6">
      <div className="flex justify-end">
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1.5 rounded-md border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
        >
          {copyState === "copied" ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          {copyState === "copied" ? "Copied" : copyState === "failed" ? "Copy failed" : "Copy results"}
        </button>
      </div>

      {resultsAreStale && (
        <div className="rounded-md bg-amber-50 px-3 py-2 text-xs font-medium text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
          Inputs changed — re-run analysis
        </div>
      )}

      <div className={`flex flex-col gap-6 transition-opacity ${resultsAreStale ? "opacity-50" : "opacity-100"}`}>
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-neutral-50">
            Data through {result.period_end}
          </h3>
          <p className="text-xs text-slate-500 dark:text-neutral-400">
            {result.observations} observations, {result.period_start} to {result.period_end}
          </p>
        </div>

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

        <div className={scoreCardGridClass}>
          <ScoreCard
            label="Sharpe Ratio"
            value={result.risk_adjusted_metrics?.sharpe_ratio}
            tooltip={METRIC_TOOLTIPS.sharpe}
            emphasis
          />
          <ScoreCard
            label="Treynor Ratio"
            value={result.risk_adjusted_metrics?.treynor_ratio}
            digits={4}
            tooltip={METRIC_TOOLTIPS.treynor}
          />
          <ScoreCard
            label="Portfolio Beta"
            value={result.risk_decomposition?.portfolio_beta}
            tooltip={METRIC_TOOLTIPS.beta}
            emphasis
          />
          <ScoreCard
            label="Jensen's Alpha"
            value={result.risk_adjusted_metrics?.jensens_alpha}
            percent
            tooltip={METRIC_TOOLTIPS.alpha}
            emphasis
          />
          <ScoreCard label="Portfolio Volatility" value={result.portfolio_volatility} percent tooltip={METRIC_TOOLTIPS.portfolioVolatility} />
          <ScoreCard label="Market Volatility" value={result.market_volatility} percent tooltip={METRIC_TOOLTIPS.marketVolatility} />
          <ScoreCard
            label="Expected Return"
            value={result.portfolio_expected_return}
            percent
            tooltip={METRIC_TOOLTIPS.expectedReturn}
          />
          <ScoreCard label="M² Measure" value={result.risk_adjusted_metrics?.m2_measure} percent tooltip={METRIC_TOOLTIPS.m2} />
        </div>

        {isCollapsed ? (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            {riskDecompositionCard}
            {assetBetasCard}
            {shortfallRiskCard}
          </div>
        ) : (
          <>
            {riskDecompositionCard}
            {assetBetasCard}
            {shortfallRiskCard}
          </>
        )}

        {whatThisMeansCard}
      </div>
    </section>
  );
}

export default function Home() {
  const idPrefix = useId();
  const [assets, setAssets] = useState<Asset[]>(INITIAL_ASSETS);
  const [riskFreeRate, setRiskFreeRate] = useState("18");
  const [thresholdReturn, setThresholdReturn] = useState("10");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [lastRunInputs, setLastRunInputs] = useState<RunInputs | null>(null);
  const [formCollapsed, setFormCollapsed] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const isCollapsed = formCollapsed && result !== null;

  const pctSystematic = result?.risk_decomposition?.pct_systematic;
  const systematicRiskPct = pctSystematic != null ? pctSystematic * 100 : undefined;
  const unsystematicRiskPct = pctSystematic != null ? (1 - pctSystematic) * 100 : undefined;

  const interpretation = result ? buildInterpretation(result) : [];

  const parsedWeights = assets.map((asset) => parseNumeric(asset.weight));
  const allWeightsParseable = parsedWeights.every((w) => w !== null);
  const hasNegativeWeight = parsedWeights.some((w) => w !== null && w < 0);
  const totalWeight = parsedWeights.reduce<number>((sum, w) => sum + (w ?? 0), 0);
  const weightIsValid = allWeightsParseable && !hasNegativeWeight && Math.abs(totalWeight - 100) < WEIGHT_TOLERANCE;

  const riskFreeRateNum = parseNumeric(riskFreeRate);
  const thresholdReturnNum = parseNumeric(thresholdReturn);
  const rateInputsValid = riskFreeRateNum !== null && thresholdReturnNum !== null;
  const thresholdBelowRiskFree =
    riskFreeRateNum !== null && thresholdReturnNum !== null && thresholdReturnNum < riskFreeRateNum;

  const tickersAreValid = assets.length > 0 && assets.every((asset) => asset.ticker.trim().length > 0);

  const formattedGroups = new Map<string, string[]>();
  for (const asset of assets) {
    const raw = asset.ticker.trim().toUpperCase();
    if (!raw) continue;
    const formatted = formatEgxTicker(raw);
    const group = formattedGroups.get(formatted) ?? [];
    group.push(raw);
    formattedGroups.set(formatted, group);
  }
  const tickerCollisions = Array.from(formattedGroups.entries()).filter(([, raws]) => raws.length > 1);
  const hasTickerCollision = tickerCollisions.length > 0;

  const canSubmit = weightIsValid && tickersAreValid && rateInputsValid && !hasTickerCollision && !loading;

  const resultsAreStale =
    result !== null &&
    lastRunInputs !== null &&
    (JSON.stringify(assets) !== JSON.stringify(lastRunInputs.assets) ||
      riskFreeRate !== lastRunInputs.riskFreeRate ||
      thresholdReturn !== lastRunInputs.thresholdReturn);

  const betaRows: BetaRow[] = (lastRunInputs?.assets ?? [])
    .map((asset) => {
      const ticker = asset.ticker.trim().toUpperCase();
      return {
        ticker,
        weight: parseNumeric(asset.weight) ?? 0,
        beta: result?.asset_betas?.[ticker],
      };
    })
    .filter((row): row is BetaRow => row.beta !== undefined)
    .sort((a, b) => b.beta - a.beta);

  const summaryWeights = (lastRunInputs?.assets ?? []).map((asset) => ({
    ticker: asset.ticker.trim().toUpperCase(),
    weightPct: parseNumeric(asset.weight) ?? 0,
  }));
  const summaryRf = parseNumeric(lastRunInputs?.riskFreeRate ?? "") ?? 0;
  const summaryThreshold = parseNumeric(lastRunInputs?.thresholdReturn ?? "") ?? 0;

  function updateAsset(id: string, patch: Partial<Pick<Asset, "ticker" | "weight">>) {
    setAssets((prev) => prev.map((asset) => (asset.id === id ? { ...asset, ...patch } : asset)));
  }

  function addAsset() {
    setAssets((prev) => [...prev, { id: `${idPrefix}-${prev.length}-${Date.now()}`, ticker: "", weight: "" }]);
  }

  function removeAsset(id: string) {
    setAssets((prev) => (prev.length > 1 ? prev.filter((asset) => asset.id !== id) : prev));
  }

  function normalizeWeights() {
    const parsed = assets.map((asset) => parseNumeric(asset.weight) ?? 0);
    const sum = parsed.reduce((s, w) => s + w, 0);
    if (sum === 0) return;

    const scaled = parsed.map((w) => Math.round(((w / sum) * 100 + Number.EPSILON) * 100) / 100);
    const scaledSum = scaled.reduce((s, w) => s + w, 0);
    const residue = Math.round((100 - scaledSum + Number.EPSILON) * 100) / 100;

    let largestIndex = 0;
    for (let i = 1; i < scaled.length; i++) {
      if (scaled[i] > scaled[largestIndex]) largestIndex = i;
    }
    scaled[largestIndex] = Math.round((scaled[largestIndex] + residue + Number.EPSILON) * 100) / 100;

    setAssets((prev) => prev.map((asset, i) => ({ ...asset, weight: scaled[i].toFixed(2) })));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;

    // Cancel any still-in-flight request so its (possibly later-arriving)
    // response can never be shown as if it were the answer to this run.
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    let didTimeout = false;
    const timeoutId = setTimeout(() => {
      didTimeout = true;
      controller.abort();
    }, REQUEST_TIMEOUT_MS);

    setLoading(true);
    setError(null);

    const payload = {
      assets: assets.map((asset) => ({
        ticker: asset.ticker.trim().toUpperCase(),
        weight: (parseNumeric(asset.weight) ?? 0) / 100,
      })),
      risk_free_rate: (riskFreeRateNum ?? 0) / 100,
      threshold_return: (thresholdReturnNum ?? 0) / 100,
    };

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
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
      setLastRunInputs({ assets, riskFreeRate, thresholdReturn });
      setFormCollapsed(true);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        if (didTimeout) {
          setError("Analysis timed out. The market data provider may be slow — try again.");
        }
        // else: superseded by a newer request — not a real error, stay silent.
      } else {
        setError(
          err instanceof Error
            ? err.message
            : "Unable to reach the analysis service. Confirm the backend is running.",
        );
      }
    } finally {
      clearTimeout(timeoutId);
      // Only the still-current request is allowed to clear the spinner —
      // a superseded request's cleanup must not stop a newer one's spinner.
      if (abortControllerRef.current === controller) {
        setLoading(false);
      }
    }
  }

  const formSection = (
    <section className="relative flex flex-col gap-6">
      {loading && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 rounded-xl bg-white/70 backdrop-blur-sm dark:bg-neutral-950/70">
          <LoaderCircle className="h-8 w-8 animate-spin text-slate-500 dark:text-neutral-400" />
          <LoadingStepLabel />
        </div>
      )}

      <form onSubmit={handleSubmit} className="flex flex-col gap-6">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-neutral-50">Global Parameters</h2>
          <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="flex items-center text-slate-600 dark:text-neutral-400">
                Risk-Free Rate (Rf)
                <InfoTooltip content={METRIC_TOOLTIPS.riskFreeRateInput} />
                {" %"}
              </span>
              <input
                type="text"
                inputMode="decimal"
                value={riskFreeRate}
                onChange={(e) => setRiskFreeRate(e.target.value)}
                onBlur={(e) => setRiskFreeRate(normalizeNumericInput(e.target.value))}
                onFocus={(e) => e.target.select()}
                onWheel={(e) => e.currentTarget.blur()}
                className="rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-900 outline-none focus:border-slate-500 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-50"
              />
            </label>
            <label className="flex flex-col gap-1.5 text-sm">
              <span className="flex items-center text-slate-600 dark:text-neutral-400">
                Threshold Return (Rl)
                <InfoTooltip content={METRIC_TOOLTIPS.thresholdReturnInput} />
                {" %"}
              </span>
              <input
                type="text"
                inputMode="decimal"
                value={thresholdReturn}
                onChange={(e) => setThresholdReturn(e.target.value)}
                onBlur={(e) => setThresholdReturn(normalizeNumericInput(e.target.value))}
                onFocus={(e) => e.target.select()}
                onWheel={(e) => e.currentTarget.blur()}
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
            <h2 className="text-sm font-semibold text-slate-900 dark:text-neutral-50">Asset Allocation</h2>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={normalizeWeights}
                disabled={totalWeight === 0 || weightIsValid}
                className="flex items-center gap-1.5 rounded-md border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
              >
                <Wand2 className="h-3.5 w-3.5" />
                Normalize
              </button>
              <button
                type="button"
                onClick={addAsset}
                className="flex items-center gap-1.5 rounded-md border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
              >
                <Plus className="h-3.5 w-3.5" />
                Add Asset
              </button>
            </div>
          </div>

          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[420px] text-sm">
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
                        type="text"
                        inputMode="decimal"
                        value={asset.weight}
                        onChange={(e) => updateAsset(asset.id, { weight: e.target.value })}
                        onBlur={(e) => updateAsset(asset.id, { weight: normalizeNumericInput(e.target.value) })}
                        onFocus={(e) => e.target.select()}
                        onWheel={(e) => e.currentTarget.blur()}
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
          </div>

          <div
            className={`mt-3 text-xs font-medium ${
              weightIsValid ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"
            }`}
          >
            Total weight: {formatNumber(totalWeight)}% {weightIsValid ? "" : "(must equal 100%)"}
          </div>

          {hasNegativeWeight && (
            <div className="mt-2 text-xs font-medium text-red-600 dark:text-red-400">
              Negative weights are not supported — EGX has no retail short selling.
            </div>
          )}

          {hasTickerCollision && (
            <div className="mt-3 flex items-start gap-2 rounded-md bg-red-50 p-2.5 text-xs text-red-800 dark:bg-red-950/40 dark:text-red-300">
              <TriangleAlert className="mt-0.5 h-3.5 w-3.5 flex-none" />
              <span>
                Duplicate tickers resolve to the same instrument:{" "}
                {tickerCollisions.map(([formatted, raws]) => `${raws.join(" + ")} → ${formatted}`).join("; ")}
              </span>
            </div>
          )}
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
  );

  const resultsSection = result ? (
    <ResultsSection
      result={result}
      betaRows={betaRows}
      systematicRiskPct={systematicRiskPct}
      unsystematicRiskPct={unsystematicRiskPct}
      interpretation={interpretation}
      resultsAreStale={resultsAreStale}
      isCollapsed={isCollapsed}
      riskFreeRatePct={summaryRf}
      thresholdReturnPct={summaryThreshold}
    />
  ) : (
    <EmptyResultsState />
  );

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-neutral-950">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <header className="mb-8">
          <h1 className="text-2xl font-semibold text-slate-900 dark:text-neutral-50">Portfolio Risk Analytics</h1>
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

        {isCollapsed ? (
          <div className="flex flex-col gap-8">
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white p-4 text-sm shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                <span className="font-medium text-slate-900 dark:text-neutral-50">
                  {summaryWeights.map((w) => `${w.ticker} ${formatNumber(w.weightPct, 0)}%`).join(" · ")}
                </span>
                <span className="text-slate-300 dark:text-neutral-700">|</span>
                <span className="text-slate-600 dark:text-neutral-400">
                  Rf {formatNumber(summaryRf, 0)}% · Threshold {formatNumber(summaryThreshold, 0)}%
                </span>
              </div>
              <button
                type="button"
                onClick={() => setFormCollapsed(false)}
                className="flex items-center gap-1.5 rounded-md border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
              >
                <Pencil className="h-3.5 w-3.5" />
                Edit
              </button>
            </div>

            {resultsSection}
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            {formSection}
            {resultsSection}
          </div>
        )}

        <footer className="mt-12 divide-y divide-slate-100 text-xs leading-relaxed text-slate-500 dark:divide-neutral-800 dark:text-neutral-500">
          <p className="py-3">
            Equity prices: Yahoo Finance. EGX30 index: Investing.com. Both are unofficial sources; prices may be
            delayed or incomplete and are not the official record of the Egyptian Exchange.
          </p>
          {result && <p className="py-3">Data last updated: {result.period_end}</p>}
          <p className="py-3">
            This tool performs historical statistical analysis. It is not investment advice, and it makes no
            recommendation to buy, sell, or hold any security.
          </p>
        </footer>
      </div>
    </div>
  );
}
