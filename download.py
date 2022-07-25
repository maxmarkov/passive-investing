from src.indexing import StockIndex
import os

FORMAT = 'parquet'
FOLDER = 'data'

STOCK_INDEX_LIST = ['niftybank'] #'venture', 'biotech'] #['dow', 'ftse100', 'ftse250', 'ibovespa', 'nasdaq', 'nifty50', 'niftybank', 'sp500']

for i, stock_index in enumerate(STOCK_INDEX_LIST):
    print(i, stock_index)
    
    stockind = StockIndex(stock_index, start_date="01/01/2005", end_date="01/01/2022")
    
    os.makedirs(FOLDER, exist_ok=True)
    
    if FORMAT == 'parquet':
        stockind.save_prices_parquet(filename=f'{FOLDER}/prices_{stock_index}.parquet.gzip')
    else:
        stockind.save_prices_csv(filename=f'{FOLDER}/prices_{stock_index}.csv')
