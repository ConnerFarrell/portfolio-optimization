# -*- coding: utf-8 -*-

'''
 portfolio_optimization.py

 Single-file demo for convex optimization–based portfolio allocation with
 practical constraints and visualization.

 Features
 - Downloads historical data for S&P 500 sector ETFs (via yfinance).
 - Computes returns, mean vector, and covariance matrix.
 - Implements mean-variance portfolio optimization with cvxpy.
 - Supports constraints:
   * L1 penalty → sparsity in allocations.
   * L2 penalty → stability/regularization.
   * Turnover limit: ||w_t – w_{t-1}|| ≤ τ.
   * No-short-sale (w ≥ 0, sum(w) = 1).
 - Solves and compares:
   * Unconstrained Markowitz optimization.
   * Constrained regularized optimization.
 - Plots:
   * Efficient frontier (risk vs. return).
   * Asset allocation bar charts.
 - Saves results:
   * PNG plots, CSV weights, and summary JSON in /outputs/.
'''

!pip -q install numpy pandas matplotlib cvxpy scipy yfinance

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional import for live data; we force synthetic for reproducibility by setting sys.argv below.
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

try:
    import cvxpy as cp  # type: ignore
except Exception as e:
    raise SystemExit("cvxpy is required. Install via: pip install cvxpy") from e

# Silence known future warnings for a clean notebook run
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# --------------------------- Utility & Data Classes ---------------------------
@dataclass
class OptArgs:
    risk_aversion: float
    l1: float
    l2: float
    budget: float
    long_only: bool
    max_weight: float
    turnover_tau: float
    w_prev: np.ndarray | None

# --------------------------- Data Handling -----------------------------------
def fetch_or_simulate_returns(
    tickers: List[str],
    start: str,
    end: str,
    freq: str,
    seed: int,
    no_internet: bool,
) -> pd.DataFrame:
    """
    Fetch adjusted (or close) prices via yfinance and convert to log returns.
    If unavailable or --no_internet, simulate Gaussian returns.

    Returns: DataFrame of log returns (index=date, columns=tickers).
    """
    month_alias = "ME"  # Month-End alias (new pandas recommendation)

    if not no_internet and yf is not None:
        try:
            data = yf.download(
                tickers, start=start, end=end, progress=False, auto_adjust=False, group_by="column"
            )
            if isinstance(data.columns, pd.MultiIndex):
                if "Adj Close" in data.columns.get_level_values(0):
                    prices = data["Adj Close"]
                elif "Close" in data.columns.get_level_values(0):
                    prices = data["Close"]
                else:
                    raise RuntimeError("No 'Adj Close' or 'Close' found in downloaded data.")
            else:
                # Single-level columns (single ticker)
                if "Adj Close" in data.columns:
                    prices = data[["Adj Close"]]
                    prices.columns = [tickers[0] if len(tickers) == 1 else "Adj Close"]
                elif "Close" in data.columns:
                    prices = data[["Close"]]
                    prices.columns = [tickers[0] if len(tickers) == 1 else "Close"]
                else:
                    raise RuntimeError("No 'Adj Close' or 'Close' column after download.")

            prices = prices.dropna(how="all")
            if prices.empty:
                raise RuntimeError("Downloaded price data is empty.")

            if freq.lower().startswith("month"):
                prices = prices.resample(month_alias).last()

            prices = prices.dropna(how="any")
            cols = [t for t in tickers if t in prices.columns]
            if len(cols) != len(tickers):
                raise RuntimeError("Some tickers missing data after download; using synthetic fallback.")
            prices = prices[cols]
            rets = np.log(prices / prices.shift(1)).dropna()
            return rets
        except Exception as e:
            warnings.warn(f"Falling back to synthetic returns due to: {e}")

    # --- Synthetic fallback (fixed seed for replicability) ---
    rng = np.random.default_rng(seed)
    n = len(tickers)
    T = 120 if freq.lower().startswith("month") else 252 * 5  # ~10y monthly, ~5y daily

    if freq.lower().startswith("month"):
        mu = rng.normal(loc=0.06 / 12, scale=0.02 / 12, size=n)  # monthly drift
        A = rng.normal(size=(n, n))
        Sigma = A @ A.T
        eigvals, _ = np.linalg.eigh(Sigma)
        scale = (0.15 / np.sqrt(12))**2 / max(np.mean(eigvals), 1e-12)  # ~15%/yr vol target
        Sigma = Sigma * scale
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=T, freq=month_alias)
    else:
        mu = rng.normal(loc=0.06 / 252, scale=0.02 / 252, size=n)  # daily drift
        A = rng.normal(size=(n, n))
        Sigma = A @ A.T
        eigvals, _ = np.linalg.eigh(Sigma)
        scale = (0.15 / np.sqrt(252))**2 / max(np.mean(eigvals), 1e-12)
        Sigma = Sigma * scale
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=T, freq="B")

    R = rng.multivariate_normal(mean=mu, cov=Sigma, size=T)
    return pd.DataFrame(R, index=idx, columns=tickers)

def estimate_moments(returns: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    assets = list(returns.columns)
    mu = returns.mean().to_numpy()
    Sigma = returns.cov().to_numpy()
    Sigma = nearest_psd(Sigma)
    return mu, Sigma, assets

def nearest_psd(Sigma: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    Sigma = 0.5 * (Sigma + Sigma.T)
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.clip(vals, eps, None)
    Sigma_psd = (vecs * vals) @ vecs.T
    Sigma_psd = 0.5 * (Sigma_psd + Sigma_psd.T)
    return Sigma_psd

# --------------------------- Optimization ------------------------------------
def solve_unconstrained(mu: np.ndarray, Sigma: np.ndarray, budget: float, risk_aversion: float) -> np.ndarray:
    n = mu.shape[0]
    w = cp.Variable(n)
    obj = risk_aversion * cp.quad_form(w, Sigma) - mu @ w
    cons = [cp.sum(w) == budget]
    prob = cp.Problem(cp.Minimize(obj), cons)
    _solve_with_fallback(prob)
    return np.asarray(w.value).ravel()

def solve_markowitz(mu: np.ndarray, Sigma: np.ndarray, opt_args: OptArgs) -> np.ndarray:
    n = mu.shape[0]
    w = cp.Variable(n)
    obj = opt_args.risk_aversion * cp.quad_form(w, Sigma) - mu @ w
    if opt_args.l1 > 0:
        obj += opt_args.l1 * cp.norm1(w)
    if opt_args.l2 > 0:
        obj += opt_args.l2 * cp.sum_squares(w)

    cons: List[cp.Constraint] = [cp.sum(w) == opt_args.budget]
    if opt_args.long_only:
        cons.append(w >= 0)
    if opt_args.max_weight < np.inf:
        cons.append(w <= opt_args.max_weight)
    if opt_args.turnover_tau > 0 and opt_args.w_prev is not None:
        cons.append(cp.norm1(w - opt_args.w_prev) <= opt_args.turnover_tau)

    prob = cp.Problem(cp.Minimize(obj), cons)
    _solve_with_fallback(prob)
    return np.asarray(w.value).ravel()

def build_frontier(mu: np.ndarray, Sigma: np.ndarray, budget: float, long_only: bool, max_weight: float, points: int) -> pd.DataFrame:
    r_min, r_max = float(np.min(mu)), float(np.max(mu))
    targets = np.linspace(r_min, r_max, num=max(points, 3))
    n = mu.shape[0]
    rows = []
    for r_star in targets:
        w = cp.Variable(n)
        cons: List[cp.Constraint] = [cp.sum(w) == budget, mu @ w >= r_star]
        if long_only:
            cons.append(w >= 0)
        if max_weight < np.inf:
            cons.append(w <= max_weight)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), cons)
        try:
            _solve_with_fallback(prob)
            if w.value is None:
                continue
            wv = np.asarray(w.value).ravel()
            var = float(wv @ Sigma @ wv)
            rows.append({"target_return": float(mu @ wv), "variance": var, "stdev": float(np.sqrt(var))})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("stdev").reset_index(drop=True)

def _solve_with_fallback(prob: cp.Problem) -> None:
    solvers = ["ECOS", "OSQP", "SCS"]
    last_err: Exception | None = None
    for s in solvers:
        try:
            prob.solve(solver=getattr(cp, s), verbose=False)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return
        except Exception as e:
            last_err = e
    try:
        prob.solve(verbose=False)
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return
    except Exception as e:
        last_err = e
    raise RuntimeError(f"Solver failed with status={prob.status} and error={last_err}")

# --------------------------- I/O & Plotting ----------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_weights(path: str, assets: List[str], w: np.ndarray) -> None:
    pd.DataFrame({"asset": assets, "weight": w}).set_index("asset").to_csv(path)

def load_w_prev(path: str, assets: List[str]) -> np.ndarray | None:
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "asset" in df.columns:
        df = df.set_index("asset")
    elif df.columns.size == 1:
        df.index = assets
    s = df.iloc[:, 0].reindex(assets)
    if s.isna().any():
        warnings.warn("w_prev CSV had missing assets; falling back to equal-weight prior.")
        return None
    return s.to_numpy()

def plot_frontier(df_frontier: pd.DataFrame, points_to_mark: Dict[str, Tuple[float, float]], out_path: str) -> None:
    plt.figure()
    if not df_frontier.empty:
        plt.plot(df_frontier["stdev"], df_frontier["target_return"], linewidth=2, label="Efficient Frontier")
    for label, (sx, sy) in points_to_mark.items():
        plt.scatter([sx], [sy], s=60, label=label)
    plt.xlabel("Portfolio Standard Deviation (per period)")
    plt.ylabel("Portfolio Mean Return (per period)")
    plt.title("Efficient Frontier")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)

def plot_allocations(assets: List[str], w: np.ndarray, title: str, out_path: str) -> None:
    plt.figure()
    plt.bar(range(len(assets)), w)
    plt.xticks(range(len(assets)), assets, rotation=45, ha="right")
    plt.ylabel("Weight")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)

# --------------------------- Metrics & Summary --------------------------------
def portfolio_stats(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> Tuple[float, float, float]:
    mean = float(mu @ w); var = float(w @ Sigma @ w); std = float(np.sqrt(var))
    return mean, var, std

def l0_sparsity(w: np.ndarray, tol: float = 1e-10) -> int:
    return int(np.sum(np.abs(w) > tol))

def l1_turnover(w: np.ndarray, w_prev: np.ndarray | None) -> float | None:
    if w_prev is None: return None
    return float(np.sum(np.abs(w - w_prev)))

# --------------------------- CLI Pipeline ------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convex optimization–based portfolio allocation demo.")
    p.add_argument("--tickers", type=str, default="XLF,XLI,XLK,XLE,XLP,XLV,XLU,XLY,XLB")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default="2025-01-01")
    p.add_argument("--freq", type=str, choices=["daily", "monthly"], default="monthly")
    p.add_argument("--risk_aversion", type=float, default=10.0)
    p.add_argument("--l1", type=float, default=0.0)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--long_only", action="store_true")
    p.add_argument("--budget", type=float, default=1.0)
    p.add_argument("--max_weight", type=float, default=1.0)
    p.add_argument("--turnover_tau", type=float, default=0.0)
    p.add_argument("--w_prev_csv", type=str, default="")
    p.add_argument("--frontier_points", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--no_internet", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--quick_test", action="store_true")
    # In notebooks, avoid parsing notebook args unless we set sys.argv ourselves
    return p.parse_args(args=[] if 'ipykernel' in sys.modules else None)

def quick_test() -> None:
    rng = np.random.default_rng(0)
    n = 5
    mu = rng.normal(0.01, 0.02, size=n)
    A = rng.normal(size=(n, n))
    Sigma = nearest_psd((A @ A.T) / n)
    w = solve_markowitz(mu, Sigma, OptArgs(5.0, 0.0, 0.1, 1.0, True, 1.0, 0.0, None))
    assert abs(np.sum(w) - 1.0) < 1e-6 and np.all(w >= -1e-8)
    print("quick test passed")

def main() -> None:
    args = parse_args()
    if args.quick_test:
        quick_test(); return

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    ensure_dir(args.output_dir)

    # 1) Data
    returns = fetch_or_simulate_returns(
        tickers=tickers, start=args.start, end=args.end,
        freq=args.freq, seed=args.seed, no_internet=args.no_internet,
    )
    returns.to_csv(os.path.join(args.output_dir, "returns.csv"))
    mu, Sigma, assets = estimate_moments(returns)

    # 2) Unconstrained solution
    w_un = solve_unconstrained(mu, Sigma, budget=args.budget, risk_aversion=args.risk_aversion)
    save_weights(os.path.join(args.output_dir, "weights_unconstrained.csv"), assets, w_un)

    # 3) Constrained solution (if requested)
    do_constrained = args.compare or (args.l1 > 0 or args.l2 > 0 or args.long_only
                                      or args.max_weight < 1.0 or args.turnover_tau > 0)

    w_prev = load_w_prev(args.w_prev_csv, assets)
    if w_prev is None and args.turnover_tau > 0:
        w_prev = np.ones(len(assets)) / len(assets)

    w_con = None
    if do_constrained:
        opt_args = OptArgs(
            risk_aversion=args.risk_aversion,
            l1=args.l1, l2=args.l2,
            budget=args.budget,
            long_only=args.long_only,
            max_weight=args.max_weight if args.max_weight > 0 else np.inf,
            turnover_tau=args.turnover_tau,
            w_prev=w_prev,
        )
        try:
            w_con = solve_markowitz(mu, Sigma, opt_args)
            save_weights(os.path.join(args.output_dir, "weights_constrained.csv"), assets, w_con)
        except Exception as e:
            warnings.warn(f"Constrained optimization failed: {e}")

    # 4) Frontier (no turnover by default)
    df_frontier = build_frontier(
        mu, Sigma, budget=args.budget,
        long_only=args.long_only,
        max_weight=args.max_weight if args.max_weight > 0 else np.inf,
        points=args.frontier_points,
    )
    df_frontier.to_csv(os.path.join(args.output_dir, "frontier.csv"), index=False)

    # 5) Stats & plots
    marks: Dict[str, Tuple[float, float]] = {}
    mu_un, var_un, std_un = portfolio_stats(w_un, mu, Sigma)
    marks["Unconstrained"] = (std_un, mu_un)

    if w_con is not None:
        mu_c, var_c, std_c = portfolio_stats(w_con, mu, Sigma)
        marks["Constrained"] = (std_c, mu_c)

    frontier_png = os.path.join(args.output_dir, "frontier.png")
    plot_frontier(df_frontier, marks, frontier_png)

    alloc_un_png = os.path.join(args.output_dir, "allocations_unconstrained.png")
    plot_allocations(assets, w_un, "Allocations — Unconstrained", alloc_un_png)

    if w_con is not None:
        alloc_c_png = os.path.join(args.output_dir, "allocations_constrained.png")
        plot_allocations(assets, w_con, "Allocations — Constrained", alloc_c_png)

    # Show plots inline
    plt.show()

    # 6) Console summary
    print("=== Data ===")
    print(f"Assets: {', '.join(assets)}")
    print(f"Samples: {returns.shape[0]} {args.freq} observations\n")

    print("=== Solutions ===")
    print("Unconstrained:")
    print(f"  mean={mu_un:.6f} per period, stdev={std_un:.6f}, var={var_un:.6f}")
    print(f"  sparsity (nonzeros): {l0_sparsity(w_un)}/{len(assets)}")
    if w_con is not None:
        print("\nConstrained "
              f"(L1={args.l1:.4f}, L2={args.l2:.4f}, long_only={args.long_only}, tau={args.turnover_tau:.4f}):")
        print(f"  mean={mu_c:.6f}, stdev={std_c:.6f}, var={var_c:.6f}")
        print(f"  sparsity: {l0_sparsity(w_con)}/{len(assets)}")
        t = l1_turnover(w_con, w_prev)
        if t is not None and args.turnover_tau > 0:
            ok = "✔" if t <= args.turnover_tau + 1e-8 else "✘"
            print(f"  turnover vs previous: {t:.6f} (<= {args.turnover_tau:.6f}) {ok}")
    print(f"\nArtifacts saved in: {os.path.abspath(args.output_dir)}")

# ---- Run with fixed, reproducible settings (synthetic data) ----
# Everyone who runs this cell will get the same results (seed=42, 120 months).
# To try live market data later, comment out sys.argv line and re-run the cell.
sys.argv = [
    "",                    # program name
    "--no_internet",       # force synthetic data for exact reproducibility
    "--freq", "monthly",
    "--compare",           # compute constrained in addition to unconstrained
    "--l1", "0.02",        # L1 penalty (sparsity)
    "--l2", "0.10",        # L2 penalty (stability)
    "--long_only",         # no short sales
    "--turnover_tau", "0.10",
    "--frontier_points", "20",
    "--seed", "42",        # fixed RNG seed
    "--output_dir", "outputs",
]
main()
