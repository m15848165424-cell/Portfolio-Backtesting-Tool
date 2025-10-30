# Portfolio-Backtesting-Tool
Backtest a long-only portfolio with on-demand real market data. Input tickers/risk-free/dates/weights/capital/rebalance; get a one-page report, charts, and CSV.

How to run (local machine):
    Requires: Python 3.10+, numpy, pandas, pandas-datareader, matplotlib
    Install:  pip install numpy pandas pandas-datareader matplotlib
    Entry:    python3 main.py

Notes for marker:
    - Uses third-party libraries (pandas, matplotlib) and renders figures (matplotlib).
    - Fetches data from the internet (Stooq/FRED) in get_data.py.
    - Ed sandbox does not allow networking/graphics, so please run locally.
