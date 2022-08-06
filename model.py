from src.indexing import StockIndexAnalysis, read_prices_csv, read_prices_parquet

FORMAT = 'parquet'
FOLDER = 'data'

stock_index = 'biotech'

if FORMAT == 'parquet':
    prices = read_prices_parquet(f'{FOLDER}/prices_{stock_index}.parquet.gzip')
else:
    prices = read_prices_csv(f'{FOLDER}/prices_{stock_index}.csv')

stockind_a = StockIndexAnalysis(prices, stock_index)
stockind_a.plot_stock_evolution("stock_evolution")                       #
stockind_a.plot_stock_evolution("stock_evolution_selected", "selected")  # select stocks with n > n_years 

stockind_a.plot_var_histogram(bins=100)
print (f"Median annual return, pct {stockind_a.median_return}")   # typical stock has return 25%
print (f"Mean annual return, pct {stockind_a.mean_return}")       # average index return is bigger

stockind_a.plot_fit_histogram(bins=20, filename=f"distribution_fit_{stock_index}.png", save=True)
stockind_a.compare_stats()
