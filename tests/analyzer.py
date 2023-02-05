import pandas as pd
import unittest

from src.analyzer import StockIndexAnalyzer

''' Unit tests for StockIndexAnalyzer class. 
    Development of this class is in progress.
'''

class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        self.ticker = "AAPL"
        self.df = pd.read_csv("data/AAPL.csv")
        self.analyzer = StockIndexAnalyzer(self.ticker, self.df)

    def test_constructor(self):
        self.assertEqual(self.analyzer.ticker, self.ticker)
        self.assertEqual(self.analyzer.df.equals(self.df), True)

    def test_load_data(self):
        self.analyzer.load_data("data/AAPL.csv")
        self.assertEqual(self.analyzer.df.equals(self.df), True)

    def test_calculate_returns(self):
        self.analyzer.calculate_returns()
        self.assertEqual("returns" in self.analyzer.df.columns, True)

    def test_calculate_rolling_mean(self):
        self.analyzer.calculate_rolling_mean(window=30)
        self.assertEqual("rolling_mean" in self.analyzer.df.columns, True)

    def test_run(self):
        self.analyzer.run()
        self.assertEqual("returns" in self.analyzer.df.columns, True)
        self.assertEqual("rolling_mean" in self.analyzer.df.columns, True)

if __name__ == '__main__':
    unittest.main()