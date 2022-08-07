#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:31:24 2022

@author: vlad

Inspired by:
    https://www.johndcook.com/blog/2011/08/09/single-big-jump-principle/
    https://www.johndcook.com/blog/2018/07/17/attribution/

    https://arxiv.org/abs/1510.03550

    https://www.researchgate.net/publication/2168231_Broad_distribution_effects_in_sums_of_lognormal_random_variables

To get data, see the example:
    http://theautomatic.net/2018/01/25/coding-yahoo_fin-package/

Start with S&P500, Nasdaq should be more interesting because of fatter tails

"""

import os 
import scipy
import numpy as np
import pandas as pd

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import yahoo_fin.stock_info as si

from scipy import stats as st      
from scipy.stats import lognorm

import pymc3 as pm3
import arviz as az

from alive_progress import alive_bar

class StockIndex:
    """ Class to get Financial Index Data from Yahoo """

    def __init__(self, stock_index, start_date = None, end_date = None):

        self.stock_index = stock_index
        self.tickers = self.get_tickers()

        # date checker 
        if start_date is None or end_date is None:
            pass
        else:
            start_d = datetime.strptime(start_date, '%m/%d/%Y').date()
            end_d = datetime.strptime(end_date, '%m/%d/%Y').date()
            assert start_d < end_d

        self.start_date = start_date
        self.end_date = end_date     
        self.prices = self.get_prices()

    def get_tickers(self):
        """ get a list of tickers for the given stock market in"""

        INDEX_LIST = ['dow', 'ftse100', 'ftse250', 'ibovespa',
                      'nasdaq', 'nifty50', 'niftybank', 'sp500',
                      'venture', 'biotech']

        if self.stock_index not in INDEX_LIST:
            raise NameError(f'Stock {self.index_name} is not in the list. Provide a correct stock name from {INDEX_LIST}')

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
            tickers = si.tickers_sp500()
        
        return tickers

    def get_prices(self) -> pd.DataFrame:
        """ Get pandas DataFrame with prices """

        start_d = datetime.strptime(self.start_date, '%m/%d/%Y').date()
        end_d = datetime.strptime(self.end_date, '%m/%d/%Y').date()
        assert start_d < end_d

        prices_list = []
        with alive_bar(len(self.tickers), force_tty=True, ctrl_c=True, title=f"Downloading {self.stock_index }") as bar:
            for ticker in self.tickers:
                try:
                    prices_list.append(si.get_data(ticker, self.start_date, self.end_date, index_as_date=False))
                except:
                    pass
                bar()

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
    df = pd.read_parquet(filename)#, index_col=0)
    return df



class StockIndexAnalysis:
    """ Class to analyze Stock Index Price data """

    def __init__(self, prices, stock_index):

        self.stock_index = stock_index
        self.prices = prices
        self.tickers = self.prices.ticker.unique()
        self.tickers_select = []
        self.mu = self.compute_variation(years_max = 7)

        self.median_return = self.mu.mu.median() #round(self.mu.mu.median()*100., 2) 
        self.mean_return = self.mu.mu.mean() #round(self.mu.mu.mean()*100., 2)

        self.log_fit = self.fit_lognormal()


    def compute_variation(self, years_max : int = 8) -> pd.DataFrame:
        """ compute variation in stock index prices over all years of observation """
        mu_dict = {}

        start_date = min(self.prices.date.unique())
        end_date = max(self.prices.date.unique())
        range_date = pd.date_range(start=start_date, end=end_date, freq='D')

        print(f"Total number of stocks: {len(self.tickers)}")
        counter = 0

        for i, ticker in enumerate(self.tickers):
            df = self.prices[self.prices.ticker==ticker].sort_values(by='date')

            try:
                n_years = df.iloc[-1].date.year - df.iloc[0].date.year
            except:
                n_years = int(df.iloc[-1].date[:4]) - int(df.iloc[0].date[:4])

            change_rate = (df.iloc[-1].close-df.iloc[0].close)/df.iloc[0].close

            # average annual return of prices , multiply by 100 if need percentage
            if n_years >= years_max and not np.isinf(change_rate):

                mu = df.iloc[-1].adjclose/df.iloc[0].adjclose#/n_years

                if not np.isnan(mu):
                    mu_dict[ticker] = mu
                    counter += 1

                if mu > 2:
                    self.tickers_select.append(ticker)
                    print(f'Stock {ticker} more than double on average during {n_years} years')

        print(f"Number of stocks with min {years_max} years: {counter}")
        print(f"Number of stocks with min {years_max} years doubled: {len(self.tickers_select)}")

        if len(mu_dict) == 0:
            import sys
            sys.exit('Empty list mu')

        dm = pd.DataFrame(mu_dict.items(), columns=['ticker', 'mu'])

        return dm

    def plot_frequency_histogram(self, bins: int = 50) -> None:
        """ plot histogram with variations """

        fig, ax = plt.subplots(1,1)

        ax = plt.gca()
        self.mu.hist(column = 'mu', grid = True, bins = bins, ax = ax, label=f"{self.stock_index} index")    # most of stock are around zero while a few increase in value >2 (double) times per year on average
        ax.tick_params(direction='in', length=6, width=1.0, colors='black', grid_color='grey', grid_alpha=0.8)

        yh = 2.

        plt.vlines(x=self.median_return, ymin=0, ymax=yh, color='r', linestyle='--', linewidth=0.8)
        plt.plot(self.median_return, yh, 'ro', markersize=4, label='median return')
        plt.text(self.median_return-2, yh+0.2, f"median {round(self.median_return,1)}", color='r', fontsize=8)

        plt.vlines(x=self.mean_return, ymin=0, ymax=yh+1.5, color='k', linestyle='--', linewidth=0.8)
        plt.plot(self.mean_return, yh+1.5, 'ko', markersize=4, label='mean return')
        plt.text(self.mean_return-2, yh+1.7, f"mean {round(self.mean_return,1)}", color='k', fontsize=8)

        plt.xlabel(r"Stock price ratio $\frac{X(t=T)}{X(t=0)}$")
        plt.ylabel("Frequency")

        plt.title("Stock price ratio distribution")

        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.xlim([0,50])
        plt.show()

    def plot_stock_evolution(self, folder: str, mode: str = "all") -> None:
        """ Plot the time evolution of a stock price for all stock in given index """

        path = os.path.join(folder, self.stock_index)
        os.makedirs(path, exist_ok = True)

        # start and end date for ploting 
        start_date = datetime.strptime("01/01/2004", '%m/%d/%Y').date()
        end_date = datetime.strptime("08/06/2022", '%m/%d/%Y').date()
        
        if mode == "selected":
            tickers = self.tickers_select
        else:
            tickers = self.tickers

        fig, ax = plt.subplots()
        for ticker in tickers:

            #fig, ax = plt.subplots()
            sns.lineplot(x = 'date', y = 'adjclose', data = self.prices[self.prices['ticker']==ticker], label = f"Index {self.stock_index}, stock {ticker}", ax=ax)

            ax.set_xlim(left=start_date, right=end_date)
            ax.set_ylim(bottom=0)

            ax.tick_params(direction='in', length=6, width=1.0, colors='black', grid_color='grey', grid_alpha=0.5)

            plt.xlabel("time")
            plt.ylabel("Closing price after adjustments")

            plt.title(f"Evolution of {ticker} price")

            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(path, f"stock_evolution_{ticker}.png"))
            plt.cla()


    def plot_fit_histogram(self, bins: int = 100, filename: str = 'distribution.png', save: bool = False) -> None:
        """ plot histogram with variations and the lognormal distribution fit """
        x = np.linspace(0.001, 50, 1000)

        fig, ax = plt.subplots(1, 1)

        f_logn = lognorm.pdf(x, self.log_fit[0], self.log_fit[1], self.log_fit[2])
        ax.plot(x, f_logn, 'r-', lw=3, alpha=0.6, label='lognorm pdf')
        ax.hist(self.mu.mu.values, density=True, histtype='stepfilled', bins=bins, alpha=0.2, label="histogram")

        if save:
            self.mu.mu.to_csv("histogram.csv")

            import csv
            with open('lognorm.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(x,f_logn))
    
        ax.legend(loc='best', frameon=False)

        plt.xlabel(r"Stock price ratio $\frac{X(t=T)}{X(t=0)}$")
        plt.ylabel("Frequency")

        plt.grid()
        plt.xlim(0,50)
        #plt.savefig(filename)
        plt.show()


    def fit_lognormal(self):
        """ fit mu histogram with lognormal distribution """
        #log_fit = scipy.stats.lognorm.fit(self.mu.mu)  #Return estimates of shape (if applicable), location, and scale parameters from data.
        #print(a) #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
        log_fit = scipy.stats.lognorm.fit(self.mu.mu, floc=0) #x0 is rawdata x-axis loc=0 for log normal in scipy, redundunt parameter
        return log_fit

    def compare_stats(self):
        """ compare distribution parameters: fit vs experiment """
        estimated_mu = np.log(self.log_fit[2])
        estimated_sigma = self.log_fit[0]

        # let's check fitted values with empirical ones see https://en.wikipedia.org/wiki/Log-normal_distribution
        mean_fit=np.exp(estimated_mu+estimated_sigma**2/2)
        mean_exp=self.mu.mu.mean()
        print('Fit mean %.2f, experimental mean %.2f'%(mean_fit, mean_exp))
        
        median_fit=np.exp(estimated_mu)
        median_exp=self.mu.mu.median()
        print('Fit median %.2f, experimental median %.2f'%(median_fit,median_exp))
        
        var_fit=(np.exp(estimated_sigma**2/2)-1)*np.exp(2.*estimated_mu+estimated_sigma**2)/self.log_fit[2]**2
        var_exp=self.mu.mu.var()
        print('Fit std %.2f, experimental std %.2f'%(np.sqrt(var_fit), np.sqrt(var_exp)))

        ### the behaviour of the sum of log normal variable is defined by parameter C 
        C = np.sqrt(np.exp(estimated_sigma**2)-1)
        print(f"Estimated Sigma^2={estimated_sigma**2}, C = {C}") # moderatly large distribution as estimated_sigma**2>1 but not >>1
        
        # compare n and C^2
        print(f"n = {len(self.mu.mu)}, C^2 = {estimated_sigma**2}, n >> C^2") # n>>C^2
        
        print(C**2, 3/2*C**2)

    def pymc3_fit(self) -> None:
        """ """

        model = pm3.Model()

        with model:
            # define distribution parameters
            mu = pm3.Normal("mu", mu = 0, sigma = 10)
            sgma = pm3.HalfCauchy("sigma", 1)
            C2 = pm3.Deterministic("C2", np.sqrt(np.exp(sgma*sgma)-1))

            # define distribution
            x = pm3.LogNormal('x', mu = mu, sigma = sgma, observed = self.mu.mu)

            # instantiate sampler
            #step = pm3.Metropolis()
            step = pm3.NUTS() 

            # draw 2000 lognormal posterior samples
            result = pm3.sample(draws=2000, step=step, tune=4000)

            # pandas object
            stats = az.summary(result, kind="stats")
            print('Summary stats', stats)
            print('Mean values', stats.iloc[0]['mean'])
            print('Result objects', result.varnames)
        
            # plot trace results for mu, sigma and C2
            fig, axs = plt.subplots(3, 2)

            plt.subplots_adjust(hspace = 0.6)
            az.plot_trace(result, axes=axs, figsize=(20,20))

            axs[0,0].axvline(x=stats.iloc[0]['mean'], linestyle='--', c='r', alpha=0.5)
            axs[0,1].axhline(y=stats.iloc[0]['mean'], linestyle='--', c='r', alpha=0.5)

            axs[1,0].axvline(x=stats.iloc[1]['mean'], linestyle='--', c='r', alpha=0.5)
            axs[1,1].axhline(y=stats.iloc[1]['mean'], linestyle='--', c='r', alpha=0.5)

            axs[2,0].axvline(x=stats.iloc[2]['mean'], linestyle='--', c='r', alpha=0.5)
            axs[2,1].axhline(y=stats.iloc[2]['mean'], linestyle='--', c='r', alpha=0.5)

            fig.suptitle(f'{self.stock_index} stats', fontsize=12)
            plt.savefig(f'pymc3_trace_{self.stock_index}.png')

             # plot trace results for mu, sigma and C2
            fig, axs = plt.subplots(1, 3)       
            az.plot_posterior(result, ax=axs,
                            var_names=["mu", "sigma", "C2"],
                            #ref_val=0,
                            hdi_prob=0.95,
                            figsize=(20, 5))
            fig.suptitle(f'{self.stock_index} log norm fit', fontsize=12)
            plt.savefig(f'pymc3_posterior_{self.stock_index}.png')

        trc = pm3.trace_to_dataframe(result)

        print('Mean (predicted) value of parameters', trc.mean(axis=0))

        