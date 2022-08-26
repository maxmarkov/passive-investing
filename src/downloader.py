import os 
import pandas as pd
from datetime import datetime
import yahoo_fin.stock_info as si
from alive_progress import alive_bar



class StockIndexDownloader:
    """ 
    Class to get Financial Index Data from Yahoo

    Basic Yahoo Finance data downloader

    To get data, see the example:
        http://theautomatic.net/2018/01/25/coding-yahoo_fin-package/
    """

    def __init__(self, stock_index, start_date = None, end_date = None, interval = "1mo"):

        """
        Arguments: 
            stock_index (str): stock index name
            start_date (str): starting date in "DD/MM/YYYY" format
            end_date (str): end date in "DD/MM/YYYY" format
            interval (str): Intervalmust be "1d", "1wk", "1mo", or "1m" for daily, weekly, monthly, or minute data. 
         """

        self.stock_index = stock_index
        self.tickers = self.get_tickers()

        if start_date is None or end_date is None:
            pass
        else:
            start_d = datetime.strptime(start_date, '%m/%d/%Y').date()
            end_d = datetime.strptime(end_date, '%m/%d/%Y').date()
            assert start_d < end_d

        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval  

        self.prices = self.get_prices()   

    def get_tickers(self):
        """ get a list of tickers for the given stock market in"""

        INDEX_LIST = ['dow', 'ftse100', 'ftse250', 'ibovespa',
                      'nasdaq', 'nifty50', 'niftybank', 'sp500',
                      'venture', 'biotech']

        if self.stock_index not in INDEX_LIST:
            raise NameError(f'Stock {self.index_name} is not in the list. Provide a correct stock name from {INDEX_LIST}')

        if self.stock_index == 'sp500':
            tickers = si.tickers_dow()
        if self.stock_index == 'dow':
            tickers = si.tickers_dow()
        elif self.stock_index == 'ftse100':
            tickers = si.tickers_ftse100()
        elif self.stock_index == 'ftse250':
            tickers = si.tickers_ftse250()
        elif self.stock_index == 'ibovespa':
            tickers = si.tickers_ibovespa()
        elif self.stock_index == 'nasdaq':
            tickers = si.tickers_nasdaq()
        elif self.stock_index == 'nifty50':
            tickers = si.tickers_nifty50()
        elif self.stock_index == 'niftybank':
            tickers = si.tickers_niftybank()
        elif self.stock_index == 'ftse100':
            tickers = si.tickers_ftse100()
        elif self.stock_index == 'venture':
            path = 'index-tickers/venture.csv'
            if os.path.isfile(path):
                index_list = pd.read_csv(path, index_col=0)
                tickers = index_list.ticker.tolist()
            else:
                print('Provide file with venture tickers')
        elif self.stock_index == 'biotech':
            path = 'index-tickers/biotech.csv'
            if os.path.isfile(path):
                index_list = pd.read_csv(path, index_col=0)
                tickers = index_list.ticker.tolist()
            else:
                print('Provide file with biotech tickers')
        else:
            ### !!! SHOULD BE IMPROVED !!! ###
            path = 'index-tickers/sp500_sectors/'
            filepath = os.path.join(path, self.stock_index+'_Aug_10_2022.csv')
            if os.path.isfile(filepath):
                index_list = pd.read_csv(filepath)
                index_list['ticker_name'] = index_list['Ticker'].apply(lambda x: x.split(' ')[0])
                tickers = index_list.ticker_name.tolist()

        return tickers

    def get_prices(self) -> pd.DataFrame:
        """ Get pandas DataFrame with prices """

        start_d = datetime.strptime(self.start_date, '%m/%d/%Y').date()
        end_d = datetime.strptime(self.end_date, '%m/%d/%Y').date()
        assert start_d < end_d

        prices_list = []
        print('Total', len(self.tickers))
        #with alive_bar(len(self.tickers), force_tty=True, ctrl_c=True, title=f"Downloading {self.stock_index }") as bar:
        #with alive_bar(len(self.tickers), ctrl_c=True, title=f"Downloading {self.stock_index }") as bar:
        for ticker in self.tickers:
            try:
                prices_list.append(si.get_data(ticker, self.start_date, self.end_date, index_as_date=False, interval=self.interval))
            except:
                pass
        #    bar()

        print(self.stock_index, 'downloaded', len(prices_list), 'from', len(self.tickers))
        if len(prices_list) > 0:
            prices = pd.concat(prices_list, ignore_index=True)
        else:
            prices = pd.DataFrame() # empty dataframe for emtpy list

        return prices

    def save_prices_parquet(self, filename: str) -> None:
        """ Save Stock Index prices into file """
        self.prices.to_parquet(filename, engine='pyarrow')

    def save_prices_csv(self, filename: str) -> None:
        """ Save Stock Index prices into file """
        self.prices.to_csv(filename)

def read_prices_csv(filename: str) -> pd.DataFrame:
    """ read file with prices in csv format """
    df = pd.read_csv(filename, index_col=0)
    return df

def read_prices_parquet(filename: str) -> pd.DataFrame:
    """ read file with prices in parquet format"""
    df = pd.read_parquet(filename)#, index_col=0), interval = "1mo"
    return df