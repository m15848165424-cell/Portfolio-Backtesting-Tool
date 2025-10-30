# main.py
'''
Portfolio Return Simulator — main entry

Flow:
  1) Interactive inputs: tickers, risk-free series (FRED), start/end dates
  2) Fetch data online -> align on the date intersection -> write ./data/*.csv
  3) Read and validate CSVs (data_io.load_* / align_or_raise)
  4) Interactive inputs: weights, initial capital, monthly rebalance or not
  5) Compute portfolio returns and wealth path
  6) Generate report and figures (report.generate_report)

Run:
    $ python3 main.py
'''

import os
import numpy as np

# --- data fetch / io ---
import get_data as gd
from data_io import load_prices, load_riskfree, align_or_raise

# --- portfolio math & user inputs ---
from portfolio import (
    ask_weights_for_tickers,
    ask_initial_capital,
    ask_rebalance,
    simple_returns_from_prices,
    portfolio_returns,
    wealth_path_from_returns,
)

# --- report output ---
import report


# --------------------------- helpers ---------------------------

def print_header() -> None:
    print("=== Portfolio Return Simulator ===\n")


def friendly_range_str(dates: list[str]) -> str:
    return f"{dates[0]} -> {dates[-1]}  (months: {len(dates)-1})"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ------------------------------ main ------------------------------

def main() -> None:
    print_header()

    # ---- (1)-(3) Inputs: tickers / rf series / start-end dates ----
    while True:
        tickers = gd.ask_tickers(gd.DEFAULT_TICKERS)                       # (1) asset list
        start   = gd.ask_date("Start date (YYYY-MM-DD)", gd.DEFAULT_START)  # (3) start date
        end     = gd.ask_date("End date   (YYYY-MM-DD)", gd.DEFAULT_END)    # (3) end date
        series  = gd.ask_series(gd.DEFAULT_FRED_SERIES)                     # (2) risk-free series (FRED)
        if start <= end:
            break
        print("Error: start date must be <= end date. Please re-enter.\n")

    # ---- Fetch online, strictly align by date intersection, then write to ./data ----
    ensure_dir(gd.OUTDIR)
    prices_m = gd.fetch_prices_stooq_monthly(tickers, start, end)   # monthly close, wide table (cols=assets)
    rf_m     = gd.fetch_rf_fred_monthly(series, start, end)         # monthly risk-free (decimal)

    # exact date intersection to ensure one-to-one alignment
    common_idx = prices_m.index.intersection(rf_m.index)
    prices_m   = prices_m.loc[common_idx]
    rf_m       = rf_m.loc[common_idx]

    prices_path = os.path.join(gd.OUTDIR, "prices_monthly.csv")
    rf_path     = os.path.join(gd.OUTDIR, "riskfree_monthly.csv")
    gd.write_prices_csv(prices_m, prices_path)
    gd.write_rf_csv(rf_m, rf_path)

    print("[OK] CSVs written under ./data/")
    print(f"  prices : {prices_path}")
    print(f"  rf     : {rf_path}\n")

    # ---- Read back & validate (consistent with data_io) ----
    prices, p_dates, tickers_read = load_prices(prices_path)
    rf_full, r_dates = load_riskfree(rf_path)
    align_or_raise(p_dates, r_dates)   # two timelines must match exactly
    assert tickers_read == tickers, "Tickers in CSV do not match the chosen tickers."

    print("=== Data Summary ===")
    print(f"Period  : {friendly_range_str(p_dates)}")
    print(f"Tickers : {', '.join(tickers)}")
    print(f"Shapes  : prices={prices.shape}, rf={rf_full.shape}\n")

    # ---- (4)-(6) Inputs: weights / initial capital / rebalance ----
    w   = ask_weights_for_tickers(tickers)    # (4) weights (Enter => equal weight)
    c0  = ask_initial_capital()               # (5) initial capital (Enter => default)
    reb = ask_rebalance()                     # (6) monthly rebalance? (Enter => default)
    print("")

    # ---- Compute: asset simple returns -> portfolio returns -> wealth path ----
    r   = simple_returns_from_prices(prices)             # (T-1, N)
    r_p = portfolio_returns(r, w, rebalance=reb)         # (T-1,)
    W   = wealth_path_from_returns(r_p, initial_capital=c0)  # (T,)

    # Align rf to r_p (portfolio monthly returns correspond to dates[1:])
    rf_aligned = np.asarray(rf_full[1:], dtype=float)    # (T-1,)
    if rf_aligned.shape[0] != r_p.shape[0]:
        raise ValueError("rf alignment failed: length mismatch with portfolio returns.")

    # ---- Generate report & figures ----
    # call report.generate_report 
    ensure_dir("outputs")

    out_paths = report.generate_report(
        prices=prices,
        dates=p_dates,
        tickers=tickers,
        wealth=W,
        r_p=r_p,                 # portfolio monthly returns (aligned to dates[1:])
        rf=rf_aligned,           # risk-free aligned to r_p (length T-1)
        weights=w,
        initial_capital=c0,
        rebalance=reb,
        outdir="outputs",
    )

    # ---- Final summary ----
    print("=== Done. Files written ===")
    for key, value in out_paths.items():
        print(f"- {key}: {value}")
    print("")


# start
if __name__ == "__main__":
    main()
