from src.analyzer import StockIndexAnalyzer
from src.downloader import read_prices_csv, read_prices_parquet

FORMAT = 'parquet'
FOLDER = 'data' #/sp500_sectors'

stock_index = 'venture' #'SPXL2_Aug_10_2022.csv'

import os

if FORMAT == 'parquet':
    prices = read_prices_parquet(f'{FOLDER}/prices_{stock_index}.parquet.gzip')
else:
    prices = read_prices_csv(f'{FOLDER}/prices_{stock_index}.csv')

stockind_a = StockIndexAnalyzer(prices, stock_index, start_date="03/01/2005", end_date="03/01/2021")
##stockind_a.plot_stock_evolution("stock_evolution")                       #
##stockind_a.plot_stock_evolution("stock_evolution_selected", "selected")  # select stocks with n > n_years 

stockind_a.plot_histogram()
print ("=== EXPERIMENTAL RESULTS ===")
print ("Experimental median price ratio, pct %.2f" %(stockind_a.median_expt))
print ("Experimental mean price ratio, pct %.2f" %(stockind_a.mean_expt))
print ("Experimental std price ratio, pct %.2f" %(stockind_a.std_expt))
#print ("Experimental C value, pct %.2f" %(stockind_a.C_expt))
print ("Experimental mode price ratio, pct %.2f" %(stockind_a.mode_expt))

stockind_a.plot_fit_histogram(save=False)
stockind_a.compare_stats()

stockind_a.pymc3_fit()

#stockind_a.find_best_distribution()

#stockind_a.plot_qq()