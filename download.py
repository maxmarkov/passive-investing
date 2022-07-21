from src.indexing import StockIndex
import os

FORMAT = 'csv'
FOLDER = 'data'

stock_index = 'sp500'

stockind = StockIndex(stock_index, start_date="01/01/2005", end_date="01/01/2022")

os.makedirs(FOLDER, exist_ok=True)

if FORMAT == 'parquet':
    stockind.save_prices_parquet(filename=f'{FOLDER}/prices_{stock_index}.parquet.gzip')
else:
    stockind.save_prices_csv(filename=f'{FOLDER}/prices_{stock_index}.csv')
