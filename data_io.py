# data_io.py
'''
Purpose:
    1) Read and validate two CSV files under ./data :
        - prices_monthly.csv  -> wide table: Date,AAPL,MSFT,AMZN,SPY,IWM,AGG
        - riskfree_monthly.csv -> two columns: Date, rf_monthly (decimal monthly return)
    2) Convert data into NumPy arrays for later calculations.
'''

from datetime import datetime
import numpy as np

ISO_FMT = "%Y-%m-%d"

def parse_iso_date(s: str) -> datetime:
    '''Turn 'YYYY-MM-DD' into a datetime object.
       e.g. return datetime(2015, 1, 31)
    '''
    return datetime.strptime(s.strip(), ISO_FMT) 


def read_csv_rows(path: str) -> list[list[str]]:
    '''Read a CSV into a list of rows (list of strings), skipping blank lines.'''
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line == "":
                    continue  # skip empty lines
                
                parts = line.split(',')
                cleaned = []
                for cell in parts:
                    cleaned.append(cell.strip())
                
                rows.append(cleaned)
    except FileNotFoundError:
        print(f'Missing file: {path}')
    
    return rows


def load_prices(path: str) -> tuple[np.ndarray, list[str], list[str]]:
    '''
    Read prices_monthly.csv (wide format) and return:
      prices: np.ndarray (T, N)
      dates : list[str]  (T)      e.g. ['2015-01-31', '2015-02-28', ...] 
      tickers: list[str] (N)      e.g. ['AAPL', 'MSFT', 'AMZN', ...]
    '''
    # validation check
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError("prices_monthly.csv is empty.")  
    header = rows[0]
    if header[0] != "Date":
        raise ValueError("First column must be 'Date' in prices_monthly.csv.")
    tickers = header[1:]
    if len(tickers) == 0:
        raise ValueError("Need at least one asset column in prices_monthly.csv.")

    # define variables
    dates: list[str] = []           
    values: list[list[float]] = []   # store rows of prices, [[110.38, 40.40, ...], [...], ...]
    prev_dt = None  # will hold the previous date (datetime) after the first row

    # Iterate over data lines
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(f"Row {i}: column count mismatch.")
        # date order check
        dt = parse_iso_date(row[0])
        if prev_dt != None and dt <= prev_dt:
            raise ValueError(
                f"Row {i}: dates must be strictly ascending "
                f"({dt.strftime(ISO_FMT)} <= {prev_dt.strftime(ISO_FMT)})."
            )
        prev_dt = dt
        dates.append(dt.strftime(ISO_FMT))
        # parse prices
        row_vals: list[float] = []
        for j, cell in enumerate(row[1:], start=2):
            if cell == "":
                raise ValueError(f"Row {i}, Col {j}: empty price.")
            try:
                x = float(cell)
            except ValueError:
                raise ValueError(f"Row {i}, Col {j}: not a number -> {cell}")
            if x <= 0.0:
                raise ValueError(f"Row {i}, Col {j}: price must be > 0, got {x}.")
            row_vals.append(x)
        values.append(row_vals)

    prices = np.array(values, dtype=float)  # (T, N)
    if True in np.isnan(prices):
        raise ValueError("Found NaN in prices (should not happen).")
    return prices, dates, tickers


def load_riskfree(path: str) -> tuple[np.ndarray, list[str]]:
    '''
    Read riskfree_monthly.csv and return:
      rf_monthly: np.ndarray (T,) (one-dimensional)
      dates     : list[str]  
    '''
    # validation check
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError("riskfree_monthly.csv is empty.")
    header = rows[0]
    if len(header) < 2 or header[0] != "Date" or header[1] != "rf_monthly":
        raise ValueError("Header must be exactly: Date, rf_monthly")

    # define variables
    dates: list[str] = []
    rf_vals: list[float] = []
    prev_dt = None # will hold the previous date (datetime) after the first row

    for i, row in enumerate(rows[1:], start=2):
        if len(row) < 2:
            raise ValueError(f"Row {i}: missing rf_monthly value.")
        dt = parse_iso_date(row[0])
        if prev_dt != None and dt <= prev_dt:
            raise ValueError(
                f"Row {i}: dates must be strictly ascending "
                f"({dt.strftime(ISO_FMT)} <= {prev_dt.strftime(ISO_FMT)})."
            )
        prev_dt = dt
        dates.append(dt.strftime(ISO_FMT))
        try:
            r = float(row[1])
        except ValueError:
            raise ValueError(f"Row {i}: rf_monthly is not a number -> {row[1]}")
        rf_vals.append(r)

    rf_monthly = np.array(rf_vals, dtype=float)
    if True in np.isnan(rf_monthly):
        raise ValueError("Found NaN in rf_monthly (should not happen).")
    return rf_monthly, dates


def align_or_raise(price_dates: list[str], rf_dates: list[str]) -> None:
    '''Ensure two date lists are IDENTICAL in length and order.'''
    if price_dates != rf_dates:
        # Show small hints to the user
        set_p, set_r = set(price_dates), set(rf_dates)
        only_p = sorted(list(set_p - set_r)) # Set difference: dates present in prices but absent from riskfree.
        only_r = sorted(list(set_r - set_p)) # Set difference: dates present in riskfree but absent from prices.
        msg = ["Dates mismatch between prices and riskfree."]
        if only_p:
            msg.append(f"Example only in prices: {only_p}")
        if only_r:
            msg.append(f"Example only in riskfree: {only_r}")
        raise ValueError("\n".join(msg))


# quick check: run `python3 data_io.py` 
if __name__ == "__main__":
    prices_file = "data/prices_monthly.csv"
    rf_file     = "data/riskfree_monthly.csv"

    try:
        prices, p_dates, tickers = load_prices(prices_file)
        rf, r_dates = load_riskfree(rf_file)
        align_or_raise(p_dates, r_dates)
    except FileNotFoundError:
        print("Expected CSVs not found under ./data/. Please check the paths.")
    except Exception as e:
        print("Error while loading CSVs:", e)
    else:
        print("=== Data OK ===")
        print(f"Period : {p_dates[0]} -> {p_dates[-1]} (months: {len(p_dates)})")
        print(f"Tickers: {', '.join(tickers)}")
        print(f"Shapes : prices={prices.shape}, rf={rf.shape}")

