# report.py
'''
Produce plots and a text summary for the Portfolio Return Simulator.

Outputs (under ./outputs by default):
  - summary.txt 
  - portfolio_path.csv           # two columns: Date, Wealth
  - prices_index_plot.png        # all assets indexed to 100 + portfolio index
  - portfolio_wealth.png         # wealth path
  - portfolio_drawdown.png       # drawdown series
'''
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from portfolio import compute_stats  
import datetime as dt
from portfolio import simple_returns_from_prices, portfolio_returns, wealth_path_from_returns, normalize_weights

# ---------------------------------------------------------------
DEFAULT_OUTDIR = "outputs"


# -------------------------- helpers ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_datetime(dates: list[str]):
    '''Convert 'YYYY-MM-DD' strings to matplotlib-friendly numbers.'''
    result: list[dt.datetime] = []     
    for s in dates:
        parsed = dt.datetime.strptime(s, "%Y-%m-%d") 
        result.append(parsed)
    return result


def drawdown_series(wealth: np.ndarray) -> np.ndarray:
    '''Return drawdown series (<=0): W/rolling_max - 1.'''
    W = np.asarray(wealth, dtype=float)
    if W.ndim != 1 or W.size < 2:
        return np.zeros_like(W, dtype=float)
    
    rollmax = np.maximum.accumulate(W)
    dd = W / rollmax - 1.0
    return dd


def save_prices_index_plot(prices: np.ndarray, wealth: np.ndarray, dates: list[str], tickers: list[str], path: str) -> None:
    '''
    Plot each asset indexed to 100 at t0, plus portfolio wealth indexed to 100.
    '''
    P = np.asarray(prices, dtype=float)
    W = np.asarray(wealth, dtype=float)
    # validation check
    if P.ndim != 2 or W.ndim != 1:
        raise ValueError("prices must be (T,N) and wealth must be (T,).")
    T, N = P.shape
    if W.shape[0] != T or len(dates) != T or len(tickers) != N:
        raise ValueError("Inconsistent lengths among prices/wealth/dates/tickers.")

    x = to_datetime(dates)
    base_prices = P[0, :].reshape(1, -1)
    idx_prices = (P / base_prices) * 100.0
    idx_port = (W / W[0]) * 100.0

    # Plot
    plt.figure(figsize=(9.5, 5.3), dpi=140)

    # assets
    for j in range(N):
        plt.plot(x, idx_prices[:, j], linewidth=1.4, alpha=0.9, label=tickers[j])
    # portfolio
    plt.plot(x, idx_port, linewidth=2.2, label="PORTFOLIO", color="black")

    plt.title("Indexed Performance (Start = 100)")
    plt.ylabel("Index (=100 at start)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(ncol=min(5, N+1), fontsize=9)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_wealth_plot(wealth: np.ndarray, dates: list[str], path: str) -> None:
    x = to_datetime(dates)
    W = np.asarray(wealth, dtype=float)
    plt.figure(figsize=(9.5, 4.8), dpi=140)
    plt.plot(x, W, linewidth=2.0)
    plt.title("Portfolio Wealth")
    plt.ylabel("Wealth ($)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_drawdown_plot(wealth: np.ndarray, dates: list[str], path: str) -> None:
    x = to_datetime(dates)
    dd = drawdown_series(wealth)  
    plt.figure(figsize=(9.5, 3.8), dpi=140)
    plt.plot(x, dd, linewidth=1.6)
    plt.fill_between(x, dd, 0, alpha=0.25)
    plt.title("Portfolio Drawdown")
    plt.ylabel("Drawdown")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_portfolio_path_csv(dates: list[str], wealth: np.ndarray, path: str) -> None:
    with open(path, "w") as f:
        f.write("Date,Wealth\n")
        for d, w in zip(dates, wealth):
            f.write(f"{d},{w:.6f}\n")


def weights_as_str(tickers: list[str], weights: np.ndarray) -> str:
    parts = []                               
    for t, w in zip(tickers, weights):      
        s = f"{t}:{float(w):.4f}"            
        parts.append(s)  
    return "  ".join(parts)


# decimal to percentage
def fmt_pct(x: float) -> str:
    return f"{x:.2%}"


# us dollar
def fmt_dollar(x: float) -> str:
    return f"${x:,.2f}"


def write_summary_txt(
    path: str,
    dates: list[str],
    tickers: list[str],
    weights: np.ndarray,
    initial_capital: float,
    rebalance: bool,
    stats: dict[str, float]
) -> None:
    start, end = dates[0], dates[-1]
    lines = []
    lines.append("=== Portfolio Report ===")
    lines.append(f"Period     : {start} -> {end}  (months: {len(dates)-1})")
    lines.append(f"Tickers    : {', '.join(tickers)}")
    lines.append(f"Weights    : {weights_as_str(tickers, weights)}  (sum=1)")
    lines.append(f"Rebalance  : {'Monthly' if rebalance else 'Buy-and-hold (drift)'}")
    lines.append(f"Initial    : {fmt_dollar(initial_capital)}")
    lines.append("")
    lines.append("--- Performance (monthly data, annualized where relevant) ---")
    lines.append(f"Final Wealth                          : {fmt_dollar(stats['final'])}")
    lines.append(f"Compound Annual Growth Rate           : {fmt_pct(stats['CAGR'])}")
    lines.append(f"Volatility (annual)                   : {fmt_pct(stats['vol_ann'])}")
    lines.append(f"Sharpe Ratio (excess over risk-free)  : {stats['sharpe']:.4f}")
    lines.append(f"Max Drawdown                          : {fmt_pct(stats['max_drawdown'])}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- This report is for simulation; not investment advice.")
    with open(path, "w") as f:
        f.write("\n".join(lines))
# -------------------------------------------------------------


def generate_report(
    prices: np.ndarray,
    dates: list[str],
    tickers: list[str],
    wealth: np.ndarray,
    r_p: np.ndarray, # returns of the portfolio
    rf: np.ndarray,
    weights: np.ndarray,
    initial_capital: float,
    rebalance: bool,
    outdir: str = DEFAULT_OUTDIR
) -> dict[str, str]:
    '''
    Create all report and return their file paths.

    Returns dict with keys:
        summary_txt, portfolio_csv, prices_index_png, wealth_png, drawdown_png
    '''
    ensure_dir(outdir)

    # 1) stats (reuse portfolio.py -> compute_stats)
    stats = compute_stats(wealth=wealth, r_p=r_p, rf=rf, freq=12)

    # 2) plots
    paths = {
        "summary_txt":        os.path.join(outdir, "summary.txt"),
        "portfolio_csv":      os.path.join(outdir, "portfolio_path.csv"),
        "prices_index_png":   os.path.join(outdir, "prices_index_plot.png"),
        "wealth_png":         os.path.join(outdir, "portfolio_wealth.png"),
        "drawdown_png":       os.path.join(outdir, "portfolio_drawdown.png"),
    }

    save_prices_index_plot(prices, wealth, dates, tickers, paths["prices_index_png"])
    save_wealth_plot(wealth, dates, paths["wealth_png"])
    save_drawdown_plot(wealth, dates, paths["drawdown_png"])

    # 3) files
    write_portfolio_path_csv(dates, wealth, paths["portfolio_csv"])
    write_summary_txt(
        paths["summary_txt"],
        dates=dates,
        tickers=tickers,
        weights=np.asarray(weights, dtype=float),
        initial_capital=float(initial_capital),
        rebalance=bool(rebalance),
        stats=stats,
    )

    return paths


# quick check: run `python3 report.py`
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "SPY"]
    dates = ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30", "2024-05-31", "2024-06-30"]
    prices = np.array([
        [100, 105, 120],
        [140, 101,  98],
        [134,  99, 100],
        [133, 100, 101],
        [135, 102, 102],
        [136, 103, 103],
    ], dtype=float)

    # returns & portfolio
    r = simple_returns_from_prices(prices)
    w = normalize_weights(np.array([0.4, 0.3, 0.3], dtype=float))
    rp = portfolio_returns(r, w, rebalance=True)
    W = wealth_path_from_returns(rp, initial_capital=10000.0)
    # dummy rf aligned to rp length
    rf = np.array([0.000025,0.000017,0.000025,0.000017,0.000017], dtype=float)

    out = generate_report(
        prices=prices, dates=dates, tickers=tickers,
        wealth=W, r_p=rp, rf=rf,
        weights=w, initial_capital=10000.0, rebalance=True,
        outdir=DEFAULT_OUTDIR
    )
    print("[report] Wrote files:")
    for key, value in out.items():
        print(f"  - {key}: {value}")
