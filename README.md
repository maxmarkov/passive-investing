# Statistical Analysis of Stock Indexes

## Project description

Install all dependencies
```
pip intstall -r requirements.txt
```

## Data

Download data from Google Drive remote storage

```
dvc pull
```

or directly from Yahoo Finance: 

```
python download.py
```

Financial data are accessible for the following indices: Dow Jones, S&P500, FTSE100, FTSE250, IBOVESPA, NIFTY50, NIFTYBANK, NASDAQ. We also grouped stocks in two separate groups: [venture capital](https://github.com/maxmarkov/stock-index/blob/master/index-tickers/venture.csv) and [biotechnology](https://github.com/maxmarkov/stock-index/blob/master/index-tickers/biotech.csv).

## Modeling

Run modeling and analysis
```
python model.py
```

Customize inputs in these to scripts for experimenting.

## Results

Read summary results in a [separate file](https://github.com/maxmarkov/stock-index/blob/master/RESULTS.md).

## References

Read literature review in a [separate file](https://github.com/maxmarkov/stock-index/blob/master/LITERATURE.md)

## TODO list

[ ] Update downloader for index-tickers
[ ] Fix alive bar downloader
[ ] Redownload data with 1 month instead of 1 day interval

References:

- [The single big jump principle](https://www.johndcook.com/blog/2011/08/09/single-big-jump-principle/)
- [Attribution based on tail probabilities](https://www.johndcook.com/blog/2018/07/17/attribution/)
- [Why indexing works?](https://arxiv.org/abs/1510.03550)
- [Broad distribution effects in sums of lognormal random variables](https://www.researchgate.net/publication/2168231_Broad_distribution_effects_in_sums_of_lognormal_random_variables)

We access financial data via Yahoo using [Yahoo Finance Python API](http://theautomatic.net/2018/01/25/coding-yahoo_fin-package/).
