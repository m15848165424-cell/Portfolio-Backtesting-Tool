# testing.py
'''
Unit tests for Portfolio Return Simulator.
Run:
    python3 testing.py
'''

import os
import unittest
import tempfile
import numpy as np

# modules under test
from portfolio import (
    simple_returns_from_prices,
    normalize_weights,
    portfolio_returns,
    wealth_path_from_returns,
    max_drawdown,
    compute_stats,
)
from data_io import load_prices, load_riskfree, align_or_raise


# ------------------------- portfolio math -------------------------

class TestPortfolioMath(unittest.TestCase):
    def test_simple_returns_positive(self):
        # prices grow 10% each month on both assets
        P = np.array([[100.0, 200.0],
                      [110.0, 220.0],
                      [121.0, 242.0]], dtype=float)
        r = simple_returns_from_prices(P)
        expected = np.array([[0.10, 0.10],
                             [0.10, 0.10]], dtype=float)
        self.assertTrue(np.allclose(r, expected))

    def test_normalize_weights_happy_path(self):
        w = normalize_weights([0.6, 0.4])
        self.assertTrue(np.allclose(w, np.array([0.6, 0.4])))
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=6)

    def test_normalize_weights_sum_not_one_negative(self):
        # Expect ValueError because weights do not sum to 1
        with self.assertRaises(ValueError):
            normalize_weights([0.5, 0.4]) 

    def test_normalize_weights_negative_weight(self):
        # negative not allowed (long-only)
        with self.assertRaises(ValueError):
            normalize_weights([-0.1, 1.1])  

    def test_portfolio_returns_rebalance_true(self):
        # one period, two assets, rebalanced -> dot product
        R = np.array([[0.10, 0.20]], dtype=float)  # (T-1, N) = (1,2)
        w = np.array([0.25, 0.75], dtype=float)
        rp = portfolio_returns(R, w, rebalance=True)
        self.assertAlmostEqual(float(rp[0]), 0.10*0.25 + 0.20*0.75, places=10)

    def test_portfolio_returns_weight_length_mismatch(self):
        # N mismatch
        R = np.array([[0.02, 0.01]], dtype=float)
        with self.assertRaises(ValueError):
            portfolio_returns(R, [1.0], rebalance=True)  

    def test_wealth_path_edge_single_period(self):
        # one month
        rp = np.array([0.10], dtype=float)  
        W = wealth_path_from_returns(rp, initial_capital=100.0)
        self.assertTrue(np.allclose(W, np.array([100.0, 110.0])))

    def test_max_drawdown_corner(self):
        # rise to 120, drop to 90 -> drawdown = 90/120 - 1 = -0.25
        W = np.array([100.0, 110.0, 120.0, 90.0, 95.0], dtype=float)
        dd = max_drawdown(W)
        self.assertAlmostEqual(dd, -0.25, places=6)

    def test_compute_stats_rf_length_mismatch_negative(self):
        # rp length 2, rf length 1 -> should raise
        W = np.array([100.0, 105.0, 110.25], dtype=float)  # two months
        rp = np.array([0.05, 0.05], dtype=float)
        rf = np.array([0.0], dtype=float)
        with self.assertRaises(ValueError):
            compute_stats(W, rp, rf, freq=12)


# --------------------------- data I/O ---------------------------

class TestDataIO(unittest.TestCase):
    '''Use temporary CSV files to test load_* and date alignment.'''

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prices_path = os.path.join(self.tmp.name, "prices_monthly.csv")
        self.rf_path = os.path.join(self.tmp.name, "riskfree_monthly.csv")

        # Valid small dataset (3 months, 2 assets)
        prices_csv = (
            "Date,A,B\n"
            "2024-01-31,100,200\n"
            "2024-02-29,110,190\n"
            "2024-03-31,120,195\n"
        )
        rf_csv = (
            "Date,rf_monthly\n"
            "2024-01-31,0.0010\n"
            "2024-02-29,0.0012\n"
            "2024-03-31,0.0011\n"
        )
        with open(self.prices_path, "w") as f:
            f.write(prices_csv)
        with open(self.rf_path, "w") as f:
            f.write(rf_csv)

    def tearDown(self):
        self.tmp.cleanup()

    # positive
    def test_load_prices_and_rf_positive(self):
        prices, dates_p, tickers = load_prices(self.prices_path)
        rf, dates_r = load_riskfree(self.rf_path)

        self.assertEqual(tickers, ["A", "B"])
        self.assertEqual(dates_p, ["2024-01-31", "2024-02-29", "2024-03-31"])
        self.assertEqual(dates_p, dates_r)
        self.assertEqual(prices.shape, (3, 2))
        self.assertEqual(rf.shape, (3,))

        # alignment should pass (no exception)
        align_or_raise(dates_p, dates_r)

    # negative: non-ascending dates in prices
    def test_load_prices_non_ascending_dates_negative(self):
        bad_path = os.path.join(self.tmp.name, "prices_bad.csv")
        bad_csv = (
            "Date,A,B\n"
            "2024-01-31,100,200\n"
            "2024-01-15,110,190\n"   # <= not strictly increasing
        )
        with open(bad_path, "w") as f:
            f.write(bad_csv)
        with self.assertRaises(ValueError):
            load_prices(bad_path)

    # negative: missing rf value
    def test_load_riskfree_missing_value_negative(self):
        bad_path = os.path.join(self.tmp.name, "rf_bad.csv")
        bad_csv = (
            "Date,rf_monthly\n"
            "2024-01-31,0.0010\n"
            "2024-02-29,\n"  # missing value
        )
        with open(bad_path, "w") as f:
            f.write(bad_csv)
        with self.assertRaises(ValueError):
            load_riskfree(bad_path)

    # edge/corner: dates mismatch in align_or_raise
    def test_align_or_raise_mismatch_corner(self):
        p_dates = ["2024-01-31", "2024-02-29", "2024-03-31"]
        r_dates = ["2024-01-31", "2024-03-31"]  # missing Feb
        with self.assertRaises(ValueError):
            align_or_raise(p_dates, r_dates)


if __name__ == "__main__":
    unittest.main()
