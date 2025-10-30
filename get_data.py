# get_data.py
'''
Purpose:
    Generate two CSV files under ./data :
        - prices_monthly.csv   (wide): Date, AAPL, MSFT, AMZN, SPY, IWM
        - riskfree_monthly.csv (two cols): Date, rf_monthly
'''

import os
from datetime import datetime
import pandas as pd
from pandas_datareader import data as pdr

# ---------- defaults (used when user just hits ENTER) ----------
DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "SPY", "IWM"]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-01-31"
DEFAULT_FRED_SERIES = "TB3MS"  # 3-Month T-Bill (annualized percentage)
OUTDIR = "data"
ISO_FMT = "%Y-%m-%d"
# ---------------------------------------------------------------


# ----------------------- input helpers -------------------------
def is_valid_ticker(s: str) -> bool:
    '''Allow letters, digits, dot, hyphen, underscore; at least 1 char.'''
    if s == "":
        return False
    for char in s:
        if char.isalnum() or char in "._-":
            continue
        return False
    return True


def ask_tickers(default: list[str]) -> list[str]:
    '''
    Ask user for comma-separated tickers.
    - Empty input(Enter) -> use default
    - Validate each ticker
    - Uppercase, strip, drop empty, de-duplicate (keep order)
    '''
    while True:
        raw = input(
            f"Tickers (comma-separated) [Enter = default: {', '.join(default)}]: "
        ).strip()
        if raw == "":
            return default[:]  

        raw_parts = raw.split(",")
        parts = [] 
        for p in raw_parts:
            cleaned = p.strip()      
            cleaned = cleaned.upper()
            if cleaned == "":       
                continue
            parts.append(cleaned)    

        # de-duplicate while keeping order
        seen = set()
        tickers = []
        for t in parts:
            if t not in seen:
                tickers.append(t)
                seen.add(t)

        if len(tickers) == 0:
            print("Error: please enter at least one ticker, or press ENTER for default.")
            continue

        # validate simple pattern
        bad = []
        for t in tickers:
            if is_valid_ticker(t):
                continue                 
            bad.append(t)                 

        if bad:
            print(f"Error: invalid ticker(s): {bad}. Allowed: letters/digits/._-")
            continue

        return tickers


def ask_date(prompt: str, default: str) -> str:
    '''
    Ask for date in YYYY-MM-DD.
    Empty input(Enter) -> default.
    Validate using pandas.to_datetime with a fixed format.
    '''
    while True:
        raw = input(f"{prompt} [Enter = default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            dt = pd.to_datetime(raw, format=ISO_FMT)  # strict format
            return dt.strftime(ISO_FMT)
        except Exception:
            print("Error: please enter date as YYYY-MM-DD (e.g., 2019-03-31).")


def ask_series(default: str) -> str:
    '''
    Ask for FRED series code (e.g., TB3MS, DGS3MO).
    Empty input(Enter) -> default.
    Simple validation: uppercase, letters/digits/underscore only.
    '''
    while True:
        raw = input(f"FRED series [Enter = default: {default}]: ").strip()
        if raw == "":
            return default
        code = raw.upper()

        ok = True
        for ch in code:
            if not ch.isalnum() and ch != "_":   
                ok = False
                break

        if not ok:
            print("Error: series should contain only letters/digits/underscore.")
            continue
        return code
# ---------------------------------------------------------------


def fetch_prices_stooq_monthly(tickers, start, end) -> pd.DataFrame:
    '''Fetch daily Close from Stooq, resample to month-end, return a wide DataFrame.'''
    frames = []
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    for t in tickers:
        # Try symbol as-is, then with '.US' suffix (some envs need it)
        df = None
        for sym in (t, f"{t}.US"):
            try:
                tmp = pdr.DataReader(sym, "stooq", start_dt, end_dt)
            except Exception:
                continue  # this attempt failed; try the next variant

            if (tmp is not None) and (len(tmp) > 0):
                df = tmp
                break

        if (df is None) or (len(df) == 0):
            raise ValueError(f"Failed to fetch data for {t} from Stooq.")

        # Ensure ascending chronological order
        df = df.sort_index()

        # Month-end close
        s = df["Close"].resample("ME").last()

        # Keep the ticker name as the column name
        frames.append(s.rename(t))

    # Concatenate to wide table in the SAME order as `tickers`
    prices_m = pd.concat(frames, axis=1)

    # Drop any month that is missing for ANY asset (ensure a complete wide table)
    prices_m = prices_m.dropna(how="any")

    # Keep only dates within [start, end] (label-based slice is inclusive)
    prices_m = prices_m.loc[start_dt:end_dt]

    return prices_m


def fetch_rf_fred_monthly(series, start, end) -> pd.Series:
    '''
    Fetch monthly rate from FRED and convert:
        - annualized percentage  -> monthly decimal return
        - rf_monthly = (1 + ann_pct/100) ** (1/12) - 1
    '''
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    df = pdr.DataReader(series, "fred", start_dt, end_dt).sort_index()

    # Align to month-end (if already monthly, this keeps it monthly)
    ann_pct = df[series].resample("ME").last()

    # Convert to monthly decimal
    rf_m = (1.0 + ann_pct / 100.0) ** (1.0 / 12.0) - 1.0
    rf_m.name = "rf_monthly"
    return rf_m


def write_prices_csv(prices_m: pd.DataFrame, path: str) -> None:
    '''Write prices CSV with Date as the first column and 6-decimal floats.'''
    out = prices_m.copy()
    out.index.name = "Date"
    out.to_csv(path, float_format="%.6f")


def write_rf_csv(rf_m: pd.Series, path: str) -> None:
    '''Write risk-free CSV with two columns: Date, rf_monthly (6-decimal floats).'''
    out = rf_m.to_frame()
    out.index.name = "Date"
    out.to_csv(path, float_format="%.6f")


def main() -> None:
    # 0) Ask user choices (ENTER -> defaults). Loop until start<=end.
    while True:
        tickers = ask_tickers(DEFAULT_TICKERS)
        start = ask_date("Start date (YYYY-MM-DD)", DEFAULT_START)
        end = ask_date("End date   (YYYY-MM-DD)", DEFAULT_END)
        series = ask_series(DEFAULT_FRED_SERIES)

        if pd.to_datetime(start) <= pd.to_datetime(end):
            break
        print("Error: start date must be <= end date. Please re-enter.\n")

    # 1) Ensure output folder exists
    os.makedirs(OUTDIR, exist_ok=True)

    # 2) Fetch prices (monthly, wide)
    prices_m = fetch_prices_stooq_monthly(tickers, start, end)

    # 3) Fetch risk-free (monthly decimal)
    rf_m = fetch_rf_fred_monthly(series, start, end)

    # 4) Align dates by intersection (strict)
    common_idx = prices_m.index.intersection(rf_m.index)
    prices_m = prices_m.loc[common_idx]
    rf_m = rf_m.loc[common_idx]

    # 5) Write CSV files (UTF-8, comma, Date ascending)
    prices_path = os.path.join(OUTDIR, "prices_monthly.csv")
    rf_path = os.path.join(OUTDIR, "riskfree_monthly.csv")
    write_prices_csv(prices_m, prices_path)
    write_rf_csv(rf_m, rf_path)

    # 6) Friendly summary
    start_str = common_idx[0].strftime(ISO_FMT)
    end_str = common_idx[-1].strftime(ISO_FMT)
    print(f"\n[OK] Generated: {prices_path} and {rf_path}")
    print(f"[Range]   {start_str} -> {end_str}")
    print(f"[Tickers] {', '.join(tickers)}")
    print(f"[FRED]    {series}")


if __name__ == "__main__":
    main()
