"""
Inspired by:
    https://www.johndcook.com/blog/2011/08/09/single-big-jump-principle/
    https://www.johndcook.com/blog/2018/07/17/attribution/

    https://arxiv.org/abs/1510.03550

    https://www.researchgate.net/publication/2168231_Broad_distribution_effects_in_sums_of_lognormal_random_variables
"""

import os
import sys 
import scipy
import numpy as np
import pandas as pd

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.visualization import hist

# stat analysis
from scipy import stats as st      
from scipy.stats import lognorm
from sklearn.neighbors import KernelDensity
from seaborn_qqplot import pplot
from scipy.stats import (gamma, lognorm, powerlognorm, loglaplace,
                         laplace_asymmetric, skewnorm)

import pymc3 as pm3
import arviz as az
from fitter import Fitter, get_common_distributions
import statsmodels.api as sm

from tqdm import tqdm

def cut_left_tail(df: pd.DataFrame, threshold: float = 2.) -> float:
    """ Cut left tail of the distribution (zombie stocks)

    Args:
        df (pd.DataFrame): dataframe containing the column 'mu' which represents the values to be cut.
        hreshold (float): logarithmic threshold used to cut the left tail of the distribution. Default is 2, i.e. thresh = log 2.
    Returns:
        d_mean (float): Percentage change in mean after cutting left tail, rounded to one decimal place.
    """

    mean = df.mu.mean()

    # threshold for cutting left tail
    df = df[np.log(df['mu'].astype(float)) > threshold]

    mean_cut = df.mu.mean()

    d_mean = 100.*(mean_cut-mean)/mean 
    return round(d_mean,1)

def cut_right_tail(df: pd.DataFrame, cut_ratio: float) -> float:
    """ Cut right tail of the distribution (best stocks) and returns the percentage change in mean after cutting right tail.
    Cut ratio in defined as the quantile of the right tail to be cut.

    Args:
        df (pd.DataFrame): dataframe containing the column 'mu' which represents the values to be cut.
        cut_ratio (float): ratio of the right tail to be cut.
    Returns:
        d_mean (float): Percentage change in mean after cutting right tail, rounded to one decimal place.
    """
    mean = df.mu.mean()

    threshold = df.mu.quantile(q=1.-cut_ratio)

    df = df[df.mu < threshold]

    mean_cut = df.mu.mean()

    d_mean = 100.*(mean_cut-mean)/mean     
    return round(d_mean,1)

def scipy_fit(df: pd.DataFrame) -> dict:
    """ Fit a lognormal distribution to the dataframe column 'mu' and returns various statistics of the fit.
    
    Args:
        df (pd.DataFrame): A dataframe containing the column 'mu' which represents the values to be fitted.
    Returns:
        dict: A dictionary containing statistics of the lognormal fit, such as mu, sigma, mean, median, mode, sigma^2, C.
    """

    # threshold for cutting left tail
    df = df[np.log(df['mu'].astype(float)) >= -2]

    # fit lognormal distribution
    log_fit = scipy.stats.lognorm.fit(df.mu, floc=0)

    logn_mu, logn_sigma = log_fit[2], log_fit[0]
    logn_mu = np.log(logn_mu)

    logn_median, logn_mean, logn_mode = np.exp(logn_mu), np.exp(logn_mu + 0.5 * logn_sigma**2), np.exp(logn_mu - logn_sigma**2)

    C = np.sqrt(np.exp(logn_sigma*logn_sigma) - 1.)

    results = {'mu': logn_mu,
               'sigma': logn_sigma,
               'mean': logn_mean,
               'median': logn_median,
               'mode': logn_mode,
               'sigma^2': logn_sigma*logn_sigma,
               'C': C}
    return results


#def scipy_fit(df: pd.DataFrame) -> dict:
#    df['log_mu'] = np.log(df['mu'].astype(float))
#    df = df.drop(df[df.log_mu < -2].index)
#    
#    log_fit = scipy.stats.lognorm.fit(df.mu, floc=0)
#    
#    logn_mu = np.log(log_fit[2])
#    logn_sigma = log_fit[0]
#
#    logn_median = np.exp(logn_mu)
#    logn_mean = np.exp(logn_mu + 0.5 * logn_sigma**2)
#    logn_mode = np.exp(logn_mu - logn_sigma**2)
#
#    C = np.sqrt(np.exp(logn_sigma*logn_sigma) - 1.)
#
#    results = {'logn mu': logn_mu,
#               'logn sigma': logn_sigma,
#               'logn mean': logn_mean,
#               'logn median': logn_median,
#               'logn mode': logn_mode,
#               'logn sigma2': logn_sigma*logn_sigma,
#               'C': C}
#
#    return results

def add_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """ Add a column 'year' to the dataframe based on the index column.

    Args:
        df (pd.DataFrame): A dataframe containing the index which represents the date of the data.
    Returns:
        df (pd.DataFrame): A dataframe with a new column 'year' added.
    """
    df['Year'] = df.index
    df['Year'] = df['Year'].apply(lambda x: x.year)
    return df

class StockIndexAnalyzer:

    def __init__(self, prices, stock_index, start_year=2006, end_year=2021, exclude_delisted=False):
        """
        Class to analyze Stock Index Price data

        Arguments:
            prices (pd.DataFrame): dataframe with data for all stocks in given index
            stock_index (str): stock index name
            start_year (int): start year for analysis. Default is 2006.
            end_year (int): end year for analysis. Default is 2021.
            exclude_delisted (bool): Delisted stocks are the ones acquired by other companies. 
                                     if True, delisted stocks are excluded from the analysis. Default is False.
        """

        self.stock_index = stock_index

        self.exclude_delisted = exclude_delisted

        assert start_year < end_year
        self.start_year = start_year
        self.end_year = end_year

        # prices are yearly data for all stocks in the index
        self.prices = prices

        # dataframe with price ratio (defined as total return in the article)
        self.mu = self._compute_stock_price_variation()

        self.log_fit = self._scipy_fit_lognormal()


    def _get_index_category(self) -> str:
        """ Get geographic/sector category for stock index """
        if self.stock_index in ['SPX','CCMP','RIY','RTY','RAY','RLV','RLG','NBI']:
            return 'US'
        elif self.stock_index in ['S5COND','S5CONS','S5ENRS','S5FINL','S5HLTH','S5INFT','S5MATR','S5RLST','S5TELS','S5UTIL','S5INDU']: 
            return 'S&P500'
        elif self.stock_index in ['DAX','CAC','UKX','BEL20','IBEX','KFX','OMX','SMI']:
            return 'EU'
        elif self.stock_index in ['AS51','HSI','STI']:
            return 'APAC'
        elif self.stock_index in ['NKY','TPX']:
            return 'Japan'
        elif self.stock_index in ['IBOV','RTSI$','NIFTY','MXIN','SHCOMP','SHSZ300']:
            return 'BRIC'
        else:
            return 'Unknown'

    def _get_tickers(self) -> list:
        """ Get list of ticks for given stock index """
        return list(self.prices.keys())

    def _compute_stock_price_variation(self) -> pd.DataFrame:
        """ Within the given index, select all stocks with available price data at `self.start_date` and `self.end_date`.
        Compute price ratio and return at given dates and write all ratio values into `dm` dataframe.  

        Returns:
            dm (pd.DataFrame): dataframe with index 
        """

        n_years = self.end_year - self.start_year + 1

        mu_dict = {}

        # indexes with doubled stock price during the period
        mu_doubled = []

        tickers = self._get_tickers()

        for ticker in tqdm(tickers):

            df = self.prices[ticker]
            df = add_year_column(df)

            if all(elem in df.columns.tolist() for elem in ['Open', 'Close']):
            
                # stock price change in two consecutive years
                n_unique = sum(abs(df['Close'].diff()) > 0)

                # After acqiuring by other companies, the stock price remains the same.
                # If we want to exclude such stocks, we need to check if the stock price changes.
                if self.exclude_delisted and n_unique < len(df)-1:
                    continue

                # stock_analyzer.start_date.year
                price_stt = df.loc[df['Year']==self.start_year, 'Open'].iloc[0]
                price_end = df.loc[df['Year']==self.end_year, 'Close'].iloc[0]

                if not np.isnan(price_stt) and not np.isnan(price_end):

                    # total return and ratio of prices, multiply by 100 if need percentage
                    return_value = (price_end - price_stt)/price_stt #/n_years
                    ratio_value = price_end / price_stt #/n_years
            
                    mu_dict[ticker] = ratio_value

                    if ratio_value > 2:
                        mu_doubled.append(ticker)
                else:
                    print(f'No price data for {ticker} at {self.start_year} and {self.end_year}')

        print(f"Total number of stocks: {len(tickers)}")
        print(f"Number of stocks with data between {self.start_year} and {self.end_year} ({n_years} years): {len(mu_dict)}")
        print(f"Number of stocks with data between {self.start_year} and {self.end_year} with at least doubled price ({n_years} years): {len(mu_doubled)}")

        if len(mu_dict) == 0:
            raise ValueError(f"No prices found at specified dates for any stock in {self.stock_index}.")

        dm = pd.DataFrame(mu_dict.items(), columns=['ticker', 'mu'])

        return dm

    def _compute_expt_stats(self, mode_method = 'kde_seaborn') -> None:
        """ Compute statistics for the distribution of price ratios 
        Args: 
            mode_method (str): method to compute distribution mode.
        """
        assert mode_method in ['hist', 'kde_sklearn', 'kde_seaborn']

        median_expt = self.mu.mu.median()
        mean_expt = self.mu.mu.mean()
        mode_expt = self._compute_expt_mode(method=mode_method)

        print(f"Expt Median: {median_expt:.3f}")
        print(f"Expt Mean: {mean_expt:.3f}")
        print(f"Expt Mode: {mode_expt:.3f}")


    def _compute_expt_mode(self, method: str = 'kde_sklearn') -> float:
        """ Compute distribution mode with one of three implemented methods:
        Shows plots of the distribution and the three methods.

            Method 1: find x at most frequent histogram bin
            Method 2: fit histogram with Gaussian KDE [scikit-learn] and find x value at maximum
            Method 3: fit histogram with Gaussian KDE [seaborn] and find x value at maximum

            Reference: https://stackabuse.com/kernel-density-estimation-in-python-using-scikit-learn/

        Arguments:
        Returns:
            mode (float): distribution mode
        """

        def get_mode(x: np.array, y: np.array) -> float:
            """ Helper function: get mode from x and y arrays """
            indx =  np.argmax(y)
            mode = x[indx]
            return mode

        n_years = self.end_year - self.start_year + 1

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5))

        # Approach 1: get most frequent value from histogram
        y,x,_ = hist(self.mu.mu, bins='freedman', range=(0,20), ax=ax[0], histtype='stepfilled', alpha=0.2, density=True, label='standard histogram')
        mode_hist = get_mode(x, y)

        # Approach 2: fit with Gaussian Kernel Density Estimation [scikit-learn]
        kde_skln = KernelDensity(kernel="gaussian", bandwidth=5.0)
        kde_skln.fit(self.mu.mu.to_numpy()[:, np.newaxis])

        x_skln = np.linspace(-1, 20, 10000)[:, np.newaxis]
        y_skln = np.exp(kde_skln.score_samples(x_skln))

        ax[1].plot(x_skln, y_skln, c='cyan')
        mode_skln = get_mode(x_skln, y_skln)[0]

        # Approach 3: fit with iGaussian Kernel Density Estimation [seaborn]
        kde_sbn = sns.kdeplot(data=self.mu.mu, legend=False)
        x_sbn, y_sbn = kde_sbn.get_lines()[0].get_data()
        mode_sbn = get_mode(x_sbn, y_sbn)

        y_list = [y, y_skln, y_sbn]
        labels = ['standard histogram', 'Kernel Density Estimation [sklearn]', 'Kernel Density Estimation [seaborn]']
        for i in range(3):
            ax[i].set_title(label=labels[i], fontsize=15)

            ax[i].set_ylabel('')
            ax[i].set_xlabel('')

            ax[i].set_xlim([-1,20])
            ax[i].set_ylim([0,1.1*np.max(y_list[i])])

            ax[i].tick_params(direction='in', length=8, width=1)
            ax[i].tick_params(axis='both', which='major', labelsize=15)

        # save figure
        DIR = 'results/kde_mode_fit'
        os.makedirs(DIR, exist_ok=True)
        plt.savefig(f'{DIR}/KDE_mode_fit_{self.stock_index}_{n_years}years.png')

        if method == 'kde_seaborn':
            return mode_sbn
        elif method == 'kde_sklearn':
            return mode_skln
        else:
            return mode_hist


    def _plot_histogram(self) -> None:
        """ Plot histogram with stock return distribution using the Freedman-Diaconis rule to perform automatic binning on the filtered
        returns data. Highlight the first and last cumulative bins collected according to specific conditions. 
        Results are saved in the results/historam folder.
        """
        n_years = self.end_year - self.start_year + 1

        # returns only
        r = self.mu.mu.copy()

        # condition for the right tail: top 5% of returns
        r_top = len(r[r >= r.quantile(q=0.9)])

        # condition for the left tail
        r_bottom = len(r[np.log(r) < -2])

        # remove left and right tails from the main histogram
        r = r[np.log(r) > -2]
        r = r[r < r.quantile(q=0.9)]

        # automatic binning with the Freedman-Diaconis rule
        bin_edges = np.histogram_bin_edges(r, bins='fd')

        # double the freedman-diaconis bins
        bin_step = (bin_edges[1] - bin_edges[0])/2.
        bin_edges_middle = [bin_edges[i-1] + bin_step for i in range(1,len(bin_edges))]
        bin_edges = sorted(bin_edges.tolist() + bin_edges_middle)

        # width of each bin
        width = (bin_edges[-1] - bin_edges[-2])

        # define the edges of the first bin and the last bin
        edge_bottom = bin_edges[0] - 0.5 * width
        edge_top = bin_edges[-1] + 0.5 * width

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        sns.histplot(data=r, kde=False, fill=True, ax=ax, bins=bin_edges, color='blue', alpha=0.2)
        print('Number of bins: ', len(bin_edges), 'Number of data points: ', len(r))

        # plot the first bin and the last cumulative bins
        ax.bar(edge_bottom, r_bottom, width=width, color='red', alpha=0.5, edgecolor='black', linewidth=0.5)
        ax.bar(edge_top, r_top, width=width, color='red', alpha=0.5, edgecolor='black', linewidth=0.5)

        ax.tick_params(direction='out', length=5., width=1, color = 'grey', labelsize=15)

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_visible(False)
        
        ax.grid(True)

        plt.xlim([edge_bottom - width, edge_top + width])

        plt.ylabel('Number of stocks', fontsize=15)
        plt.xlabel(r'$\rho$', fontsize=18)
        plt.title(f"{self.stock_index} stock total return distribution", fontsize=15)

        # save the figure
        DIR = 'results/histogram'
        os.makedirs(DIR, exist_ok=True)
        fig.savefig(f'{DIR}/histogram_{self.stock_index}_{n_years}years.png')


    def plot_histogram_loglog(self) -> None:
        """ plot histogram in log-log scale"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.tick_params(direction='in', length=8, width=1)
        data, bins, _ = hist(self.mu.mu, bins='freedman', ax=ax, histtype='stepfilled', alpha=0.2, density=True, label='standard histogram')
        plt.grid()
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlim([0.5,50])
        #ax.tick_params(direction='in', length=8, width=1)

        plt.xlabel(r"log stock price ratio $\frac{X(t=T)}{X(t=0)}$")
        plt.ylabel("log frequency")

        DIR = 'results/histogram_loglog'
        os.makedirs(DIR, exist_ok=True)
        plt.savefig(f'{DIR}/histogram_loglog_{self.stock_index}_{self.nyears}years.png')


    def plot_histogram_fit(self, save_data: bool = False) -> None:
        """ plot histogram with variations and the lognormal distribution fit """

        x = np.linspace(0.001, 50, 1000)

        fig, ax = plt.subplots(1, 1)

        f_logn = lognorm.pdf(x, self.log_fit[0], self.log_fit[1], self.log_fit[2])
        ax.plot(x, f_logn, 'r-', lw=3, alpha=0.6, label='lognorm pdf')
        #data = ax.hist(self.mu.mu.values, density=True, histtype='stepfilled', bins=self.bins, alpha=0.2, label="histogram")
        data = hist(self.mu.mu, bins='freedman', range=(0,20), ax=ax, histtype='stepfilled', alpha=0.2, density=True, label='standard histogram')
        
        x_hist = [(data[1][i]+data[1][i-1])/2 for i in range(1, len(data[1]))]
        y_hist = data[0].tolist()

        df_hist = pd.DataFrame({'x': x_hist, 'y': y_hist})

        if save_data:
            df_hist.to_csv(f"histogram_{self.stock_index}.csv", index=False)
            #self.mu.mu.to_csv(f"histogram_.csv")

            import csv
            with open(f'lognorm_{self.stock_index}.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(x,f_logn))
    
        ax.legend(loc='best', frameon=False)

        plt.xlabel(r"Stock price ratio $\frac{X(t=T)}{X(t=0)}$")
        plt.ylabel("Frequency")

        plt.title(f"{self.stock_index} distribution scipy fit ")

        plt.grid()
        plt.xlim(0,20)

        DIR = 'results/distribution_fit'
        os.makedirs(DIR, exist_ok=True)
        plt.savefig(f'{DIR}/distribution_fit_{self.stock_index}_{self.nyears}years.png')


    def _scipy_fit_lognormal(self):
        """ Fit mu histogram with lognormal distribution
        """
        log_fit = scipy.stats.lognorm.fit(self.mu.mu, floc=0)
        return log_fit


    def compare_stats(self) -> dict:
        """ 
        Compare distribution parameters: scipy fit vs experiment
        Reference: https://en.wikipedia.org/wiki/Log-normal_distribution
        """
        logn_mu = np.log(self.log_fit[2])
        logn_sigma = self.log_fit[0]

        logn_median = np.exp(logn_mu)
        logn_mean = np.exp(logn_mu + 0.5 * logn_sigma**2)
        logn_mode = np.exp(logn_mu - logn_sigma**2)

        C = np.sqrt(np.exp(logn_sigma*logn_sigma) - 1.)
        
        ## compare n and C^2
        #print(f"n = {len(self.mu.mu)}, C^2 = {round(estimated_sigma**2,2)}, n >> C^2") # n>>C^2
        #print(f" C^2 = {C**2}, 3/2 C^2 = {3/2*C**2}")

        results = {'logn mu': logn_mu,
                   'logn sigma': logn_sigma,
                   'logn mean': logn_mean,
                   'logn median': logn_median,
                   'logn mode': logn_mode,
                   'logn sigma2': logn_sigma*logn_sigma,
                   'C': C
                  }
        return results


    def pymc3_fit(self, draws: int = 5000, tune: int = 3000) -> dict:
        """ Fit histogram distribution with PyMC3 """

        model = pm3.Model()

        with model:

            # Define parameters for LogNormal distribution
            muh = pm3.Normal('muh', 0.1, 1)                           # Mean stock drift: \hat{\mu}
            sgmah = pm3.HalfNormal('sigmah', 1)                       # Std stock variation: \hat{\sigma}
            sgma = pm3.HalfNormal('sigma', 1)                         # Volatility: \sigma

            # define distribution
            x = pm3.LogNormal('x', mu = muh * self.nyears - 0.5 * sgma**2 * self.nyears,
                                   sigma = np.sqrt(sgma**2 * self.nyears + sgmah**2 * self.nyears**2),
                                   observed = self.mu.mu)

            # instantiate sampler
            step = pm3.NUTS() 

            result = pm3.sample(draws=draws, target_accept=0.98, tune=tune)
            stats = az.summary(result, kind="stats")
            print("\n")
            print(" === PYMC3 FIT === ")
            print(stats)

            # Posterior. Select only last 2000 (most stable)
            ppc = pm3.sample_posterior_predictive(result, 200)
            # early trial are not very stable. we select last 75%
            draws_stable = round(0.5*draws) 
            trc = pm3.trace_to_dataframe(result[-draws_stable:])

            # Internal parameters (mean posterior)
            muh_post = trc['muh'].mean()
            sigh_post = trc['sigmah'].mean()
            sig_post = trc['sigma'].mean()

            ### LOGNORMAL distribution statistics
            logn_mu =  muh_post * self.nyears - 0.5 * sig_post**2 * self.nyears
            logn_sigma = sig_post**2 * self.nyears + sigh_post**2 * self.nyears**2
            C = np.sqrt(np.exp(logn_sigma * logn_sigma) - 1.)

            logn_mean = np.exp(logn_mu + 0.5* logn_sigma**2)
            logn_median = np.exp(logn_mu)
            logn_mode = np.exp(logn_mu - logn_sigma**2)

            #### LOGNORMAL POSTERIOR DISTRIBUTION PARAMETERS ################
            #print('PYMC3 lognorm median %0.2f' % (np.median(ppc['x'])))
            #print('PYMC3 lognorm mean %0.2f' % (ppc['x'].mean()))
            #print('PYMC3 lognorm std %0.2f' % (np.sqrt(np.var(ppc['x']))))


            ### Plot 1: distribution (KDE) and sampled values for muh, sgmah and sgma ###
            fig, axs = plt.subplots(3, 2)

            plt.subplots_adjust(hspace=0.7, wspace=0.3)
            az.plot_trace(result, axes=axs, figsize=(20,10))

            axs[0,0].axvline(x=stats.iloc[0]['mean'], linestyle='--', c='r', alpha=0.5)
            axs[0,1].axhline(y=stats.iloc[0]['mean'], linestyle='--', c='r', alpha=0.5)

            axs[1,0].axvline(x=stats.iloc[1]['mean'], linestyle='--', c='r', alpha=0.5)
            axs[1,1].axhline(y=stats.iloc[1]['mean'], linestyle='--', c='r', alpha=0.5)

            axs[2,0].axvline(x=stats.iloc[2]['mean'], linestyle='--', c='r', alpha=0.5)
            axs[2,1].axhline(y=stats.iloc[2]['mean'], linestyle='--', c='r', alpha=0.5)

            axs[0,0].set_title('mean stock drift')
            axs[0,1].set_title('mean stock drift')

            axs[1,0].set_title('std stock variation')
            axs[1,1].set_title('std stock variation')

            axs[2,0].set_title('volatility')
            axs[2,1].set_title('volatility')

            fig.suptitle(f'{self.stock_index} stats', fontsize=12)

            DIR = 'results/mcmc_plot_trace'
            os.makedirs(DIR, exist_ok=True)
            plt.savefig(f'{DIR}/mcmc_trace_{self.stock_index}_{self.nyears}years.png')


            ### Plot 2: posterior densities for muh, sgmah and sgma ###
            fig, axs = plt.subplots(1, 3)
            plt.subplots_adjust(wspace=2.0)
            az.plot_posterior(result, 
                              ax=axs,
                              var_names=["muh", 'sigmah', "sigma"],
                              #ref_val=0,
                              hdi_prob=0.95,
                              figsize=(20, 10))

            axs[0].set_title(r'mean drift $\hat{\mu}$')
            axs[1].set_title(r'variation $\hat{\sigma}$')
            axs[2].set_title(r'volatility $\sigma$')

            fig.suptitle(f'{self.stock_index} posterior densities', fontsize=12, y=1.01)
            DIR = 'results/mcmc_plot_posterior'
            os.makedirs(DIR, exist_ok=True)
            plt.savefig(f'{DIR}/mcmc_posterior_{self.stock_index}_{self.nyears}years.png')

        trc = pm3.trace_to_dataframe(result)

        results = {'muh': trc['muh'].mean(),
                   'muh std': trc['muh'].std(),
                   'sigmah': trc['sigmah'].mean(),
                   'sigmah std': trc['sigmah'].std(),
                   'sigma': trc['sigma'].mean(),
                   'sigma std': trc['sigma'].std(),
                   'logn mu': logn_mu,
                   'logn sigma': logn_sigma,
                   'logn mean': logn_mean,
                   'logn median': logn_median,
                   'logn mode': logn_mode,
                   'logn sigma2': logn_sigma*logn_sigma,
                   'C': C
                  }

        return results


    def find_best_distribution(self) -> pd.DataFrame:
        """ 
        Find best distribution fit.
        Read more: https://medium.com/the-researchers-guide/finding-the-best-distribution-that-fits-your-data-using-pythons-fitter-library-319a5a0972e9
        """
        
        fig, ax = plt.subplots(1, 1)

        dists = get_common_distributions()
        f = Fitter(self.mu.mu, distributions=dists, xmin=0, xmax=20)
        f.fit()

        summary = f.summary(10)

        DIR = 'results/best_distribution'
        os.makedirs(DIR, exist_ok=True)

        plt.savefig(f'{DIR}/distributions_{self.stock_index}_{self.nyears}years.png')

        return summary



    def plot_qq(self) -> None:
        """ Q-Q plot of the quantiles of x versus the quantiles/ppf of a distribution """
        fig, ax = plt.subplots(1, 1)

        sm.qqplot(np.log(self.mu.mu), line ='s', ax=ax)  # it is good that the right tail is underestimated
        plt.grid()
        DIR = 'results/qqplot'
        os.makedirs(DIR, exist_ok=True)

        plt.savefig(f'{DIR}/qqplot_{self.stock_index}_{self.stock_index}.png')

    def plot_qq_seaborn(self) -> None:
        """ Q-Q plot from searborn """

        self.mu['log_mu'] = np.log(self.mu['mu'].astype(float))

        DIR = 'results/qqplot_seaborn'
        os.makedirs(DIR, exist_ok=True)

        ax1 = pplot(self.mu, x="mu", y=powerlognorm, kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True,"reg":True, "ci":0.025})
        title_text='QQ plot for '+self.stock_index+' fit with powerlognorm'
        plt.title(title_text)
        plt.savefig(f'{DIR}/qqplot_{self.stock_index}_powerlognorm.png',bbox_inches='tight')
        
        ax2 = pplot(self.mu, x="mu", y=lognorm, kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True,"reg":True, "ci":0.025})
        title_text='QQ plot for '+self.stock_index+' fit with lognorm'
        plt.title(title_text)
        plt.savefig(f'{DIR}/qqplot_{self.stock_index}_lognorm.png',bbox_inches='tight')

        ax3 = pplot(self.mu, x="mu", y=loglaplace, kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True,"reg":True, "ci":0.025})
        title_text='QQ plot for '+self.stock_index+' fit with loglaplace'
        plt.title(title_text)
        plt.savefig(f'{DIR}/qqplot_{self.stock_index}_loglaplace.png',bbox_inches='tight')

        ax4 = pplot(self.mu, x="log_mu", y=laplace_asymmetric, kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True,"reg":True, "ci":0.025})
        title_text='QQ plot for '+self.stock_index+' fit with laplace asymmetric'
        plt.title(title_text)
        plt.savefig(f'{DIR}/qqplot_{self.stock_index}_laplace-asymmetric.png',bbox_inches='tight')

        ax5 = pplot(self.mu, x="log_mu", y=skewnorm, kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True,"reg":True, "ci":0.025})
        title_text='QQ plot for '+self.stock_index+' fit with skewnorm'
        plt.title(title_text)
        plt.savefig(f'{DIR}/qqplot_{self.stock_index}_skewnorm.png',bbox_inches='tight')


    #def plot_stock_evolution(self, folder: str, mode: str = "all") -> None:
    #    """ Plot the time evolution of a stock price for all stock in given index """

    #    path = os.path.join(folder, self.stock_index)
    #    os.makedirs(path, exist_ok = True)

    #    # start and end date for ploting 
    #    start_date = datetime.strptime("01/01/2004", '%m/%d/%Y').date()
    #    end_date = datetime.strptime("08/06/2022", '%m/%d/%Y').date()
    #    
    #    if mode == "selected":
    #        tickers = self.tickers_select
    #    else:
    #        tickers = self.tickers

    #    fig, ax = plt.subplots()
    #    for ticker in tickers:

    #        #fig, ax = plt.subplots()
    #        sns.lineplot(x = 'date', y = 'adjclose', data = self.prices[self.prices['ticker']==ticker], label = f"Index {self.stock_index}, stock {ticker}", ax=ax)

    #        ax.set_xlim(left=start_date, right=end_date)
    #        ax.set_ylim(bottom=0)

    #        ax.tick_params(direction='in', length=6, width=1.0, colors='black', grid_color='grey', grid_alpha=0.5)

    #        plt.xlabel("time")
    #        plt.ylabel("Closing price after adjustments")

    #        plt.title(f"Evolution of {ticker} price")

    #        plt.grid(True, linestyle='--', alpha=0.3)
    #        plt.legend()
    #        #plt.savefig(os.path.join(path, f"stock_evolution_{ticker}.png"))
    #        plt.show()
    #        plt.cla()
