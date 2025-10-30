# portfolio.py
'''
Core portfolio math using NumPy 

What this module provides:
   1) simple_returns_from_prices(prices)
   2) normalize_weights(weights)
   3) portfolio_returns(returns, weights, rebalance=True)
   4) wealth_path_from_returns(r_p, initial_capital)
   5) compute_stats(wealth, r_p=None, rf=None, freq=12)      
        - CAGR, annualized volatility, Sharpe (excess), max drawdown

Notes:
   - prices: 2D array shape (T, N) with T months and N assets (floats)
   - returns: 2D array shape (T-1, N) = month-over-month simple returns
   - weights: 1D array length N, typically sums to 1.0
   - rf: 1D monthly risk-free decimal returns, length T-1 (to match returns)
   - We use monthly data by default (freq=12).
'''

import numpy as np

# ---------- user input helpers ----------

def ask_weights_for_tickers(tickers: list[str]) -> np.ndarray:
    '''
    Ask user for long-only portfolio weights for the given tickers.
    - Separator: spaces only (e.g., '0.4 0.3 0.3')
    - Blank input(Enter) => equal weights (1/N)
    - Validation (non-negative, sum==1, finite) is delegated to normalize_weights
    '''
    N = len(tickers)
    prompt = (
        f"Enter {N} space-separated weights for {' '.join(tickers)} "
        f"(sum=1, long-only) [Enter = default: equal weight]: "
    )

    while True:
        raw = input(prompt).strip()

        # blank -> equal weights
        if raw == "":
            return np.ones(N, dtype=float) / N

        # reject comma-separated input; we require spaces
        if "," in raw:
            print("Error: use spaces, not commas. Example: 0.4 0.3 0.3")
            continue

        parts = raw.split()  # split on whitespace
        if len(parts) != N:
            print(f"Error: expected {N} numbers separated by spaces, got {len(parts)}.")
            continue

        # numeric parse
        try:
            float_list = []
            for p in parts:
                x = float(p)          
                float_list.append(x)
            w = np.array(float_list, dtype=float)           
        except ValueError:
            print("Error: weights must be numeric. Example: 0.4 0.3 0.3")
            continue

        # delegate all validation to normalize_weights
        try:
            w = normalize_weights(w)
        except ValueError as e:
            print(f"Error: {e}")
            continue

        return w


def ask_initial_capital(default: float = 10000.0) -> float:
    '''
    Ask user for initial capital. Empty input(Enter) uses the default.

    Output:
        capital : float > 0
    '''
    while True:
        raw = input(f"Initial capital [Enter = default: {default}]: ").strip()
        if raw == "":
            return float(default)
        try:
            x = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if not np.isfinite(x) or x <= 0.0:
            print("Initial capital must be a positive finite number.")
            continue
        return float(x)


def ask_rebalance(default: bool = True) -> bool:
    '''
    Ask user whether to rebalance monthly (y/n). Empty input(Enter) uses the default.

    Output:
        rebalance : bool
    '''
    if default:
        yn = "y"
    else:
        yn = "n"

    while True:
        raw = input(f"Monthly rebalance? (y/n) [Enter = default: {yn}]: ").strip().lower()
        if raw == "":
            return bool(default)
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer with 'y' or 'n'.")


# ---------- helpers ----------

def as_2d_float(a):
    '''Convert to a 2D float NumPy array (copy), or raise ValueError.'''
    arr = np.array(a, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array (rows: time, cols: assets).")
    return arr


def as_1d_float(a):
    '''Convert to a 1D float NumPy array (copy), or raise ValueError.'''
    arr = np.array(a, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")
    return arr


# ---------- 1) returns from prices ----------

def simple_returns_from_prices(prices):
    '''
    Compute simple month-over-month returns from price matrix.

    Input:
        prices : ndarray (T, N)  ascending in time
    Output:
        returns: ndarray (T-1, N)  where r[t, j] = prices[t+1, j]/prices[t, j] - 1
    '''
    P = as_2d_float(prices)
    T, N = P.shape
    if T < 2:
        raise ValueError("Need at least two rows to compute returns.")
    if np.any(P <= 0.0):
        raise ValueError("Prices must be > 0.")

    # vectorized percent change
    r = P[1:, :] / P[:-1, :] - 1.0
    return r


# ---------- 2) weights normalization ----------

def normalize_weights(weights):
    '''
    Enforce that user-provided portfolio weights already sum to 1.

    Input:
        weights : 1D array-like of length N
    Output:
        w : 1D float NumPy array (length N)
    Raises:
        ValueError if any weight is not finite, or if the sum is not 1.
    '''
    w = as_1d_float(weights)  
    N = w.shape[0]

    # check finite numbers
    for j in range(N):
        if not np.isfinite(w[j]):
            raise ValueError(f"Weight at position {j} is not a finite number.")
    # long_only
    if (w < -1e-12).any():
        raise ValueError("This simulator is long-only; weights must be >= 0.")
    # must sum to exactly 1 (allowing tiny floating error)
    total = float(np.sum(w))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1. Got {total:.6f}.")

    return w


# ---------- 3) portfolio returns (static weights, rebalanced monthly) ----------

def portfolio_returns(returns, weights, rebalance=True):
    '''
    Compute portfolio monthly returns from asset returns and weights.

    Input:
        returns : ndarray (T-1, N) asset simple returns
        weights : ndarray (N,)
        rebalance: if True (default), assume weights are rebalanced monthly
                   (i.e., same weights every month). If False, a simple
                   buy-and-hold approximation is used (weights drift).
    Output:
        r_p : ndarray (T-1,) portfolio returns
    '''
    R = as_2d_float(returns)
    w = normalize_weights(weights)

    Tm1, N = R.shape
    if w.shape[0] != N:
        raise ValueError(f"weights length ({w.shape[0]}) != number of assets ({N}).")

    # Rebalance: same weights each month
    if rebalance:
        r_p = R @ w
        return r_p

    # Simple buy-and-hold approximation
    r_p = np.zeros(Tm1, dtype=float) 
    # start with given weights as capital fractions
    w_curr = w.copy()
    for t in range(Tm1):
        # portfolio return is weighted sum
        r_p[t] = float(np.sum(w_curr * R[t, :]))
        # update weights after returns (no additional cash)
        # new weights proportional to (1 + r_j)
        growth = (1.0 + R[t, :]) * w_curr
        total = float(np.sum(growth))
        if total == 0.0:
            # if everything goes to zero (should not in practice), break
            r_p[t] = -1.0
            break
        w_curr = growth / total
        # continue until we finish all months
    return r_p


# ---------- 4) wealth path from returns ----------

def wealth_path_from_returns(r_p, initial_capital=10000.0):
    '''
    Convert monthly portfolio returns into a wealth (value) path.

    Input:
        r_p : ndarray (T-1,)  portfolio returns
        initial_capital : float, starting value
    Output:
        wealth : ndarray (T,) where wealth[0] = initial_capital
                 and wealth[t] = wealth[t-1] * (1 + r_p[t-1]) for t>=1
    '''
    rp = as_1d_float(r_p)
    if initial_capital <= 0.0:
        raise ValueError("initial_capital must be > 0.")

    Tm1 = rp.shape[0]
    wealth = np.zeros(Tm1 + 1, dtype=float)
    wealth[0] = float(initial_capital)
    for t in range(1, Tm1 + 1):
        wealth[t] = wealth[t - 1] * (1.0 + rp[t - 1])
    return wealth


# ---------- 5) statistics ----------

def max_drawdown(wealth):
    '''
    Compute maximum drawdown from a wealth path.
    Output:
        max_dd : float (non-positive), e.g., -0.25 means -25%
    '''
    W = as_1d_float(wealth)
    if W.shape[0] < 2:
        return 0.0
    peak = W[0]
    max_dd = 0.0
    for t in range(1, W.shape[0]):
        if W[t] > peak:
            peak = W[t]
            continue
        dd = W[t] / peak - 1.0
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


def compute_stats(wealth, r_p, rf, freq=12):
    '''
    Compute common performance statistics.

    Input:
        wealth : ndarray (T,)  wealth path from wealth_path_from_returns
        r_p    : ndarray (T-1,) portfolio monthly returns 
        rf     : ndarray (T-1,) risk-free monthly returns (decimal), same length as r_p
        freq   : periods per year (12 for monthly)
    Output:
        stats : dict with keys:
            'final', 'CAGR', 'vol_ann', 'sharpe', 'max_drawdown'
    '''
    W = as_1d_float(wealth)
    T = W.shape[0]
    if T < 2:
        raise ValueError("Wealth path must have at least 2 points.")

    years = (T - 1) / float(freq)
    if years <= 0.0:
        raise ValueError("Not enough periods to compute annualized metrics.")

    final_val = float(W[-1])
    # compound annual growth rate
    cagr = float((final_val / W[0]) ** (1.0 / years) - 1.0)

    # returns
    rp = as_1d_float(r_p)

    # excess returns (if rf is provided and aligned), else raw returns
    rf_arr = as_1d_float(rf)
    if rf_arr.shape[0] != rp.shape[0]:
        raise ValueError("Length of rf must match returns length.")
    ex = rp - rf_arr

    # volatility: sample std * sqrt(freq); 
    if rp.shape[0] >= 2:
        vol_ann = float(np.std(rp, ddof=1) * np.sqrt(freq))
    else:
        vol_ann = 0.0

    # sharpe ratio: mean(excess) * sqrt(freq) / std(raw)
    if vol_ann > 0.0:
        sharpe = float(np.mean(ex) * np.sqrt(freq) / np.std(rp, ddof=1))
    else:
        sharpe = 0.0
    
    # maximum drawdown
    max_dd = max_drawdown(W)

    return {
        "final": final_val,
        "CAGR": cagr,
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# quick check: run `python3 portfolio.py` 
if __name__ == "__main__":
    # demo tickers and 6 months of prices (rows=time, cols=assets)
    tickers = ["AAPL", "MSFT", "SPY"]
    prices = np.array([
        [100, 105, 120],
        [140, 101,  98],
        [134,  99, 100],
        [133, 100, 101],
        [135, 102, 102],
        [136, 103, 103],
    ], dtype=float)

    # user inputs
    w   = ask_weights_for_tickers(tickers)     # space-separated; Enter => equal weights
    c0  = ask_initial_capital()                # Enter => default 10000.0
    reb = ask_rebalance()                      # Enter => default yes

    # pipeline
    r   = simple_returns_from_prices(prices)   # (T-1, N)
    r_p = portfolio_returns(r, w, rebalance=reb)
    W   = wealth_path_from_returns(r_p, initial_capital=c0)

    # set risk-free series for demo (0% per month, aligned length)
    rf  = np.array([0.000025,0.000017,0.000025,0.000017,0.000017], dtype=float)

    stats = compute_stats(W, r_p, rf, freq=12)

    # report
    print("\n--- Quick check ---")
    print("Tickers: ", ", ".join(tickers))
    print("Weights:", np.round(w, 4).tolist())
    print("Rebalance monthly:", reb)
    print(f"Final wealth: {stats['final']:.4f}")
    print(f"CAGR: {stats['CAGR']:.4%}")
    print(f"Annual volatility: {stats['vol_ann']:.4%}")
    print(f"Sharpe ratio: {stats['sharpe']:.4f}")
    print(f"Max drawdown: {stats['max_drawdown']:.4%}")

    


