"""
Inspired by:
    https://www.johndcook.com/blog/2011/08/09/single-big-jump-principle/
    https://www.johndcook.com/blog/2018/07/17/attribution/

    https://arxiv.org/abs/1510.03550

    https://www.researchgate.net/publication/2168231_Broad_distribution_effects_in_sums_of_lognormal_random_variables
"""

import os
import csv
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

from typing import Tuple

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

        self.nyears = self.end_year - self.start_year + 1

        # prices are yearly data for all stocks in the index
        self.prices = prices
        self.tickers = list(self.prices.keys())

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
                    return_value = (price_end - price_stt)/price_stt #/self.nyears
                    ratio_value = price_end / price_stt #/self.nyears
            
                    mu_dict[ticker] = ratio_value

                    if ratio_value > 2:
                        mu_doubled.append(ticker)
                else:
                    print(f'No price data for {ticker} at {self.start_year} and {self.end_year}')

        print(f"Total number of stocks: {len(tickers)}")
        print(f"Number of stocks with data between {self.start_year} and {self.end_year} ({self.nyears} years): {len(mu_dict)}")
        print(f"Number of stocks with data between {self.start_year} and {self.end_year} with at least doubled price ({self.nyears} years): {len(mu_doubled)}")

        if len(mu_dict) == 0:
            raise ValueError(f"No prices found at specified dates for any stock in {self.stock_index}.")

        dm = pd.DataFrame(mu_dict.items(), columns=['ticker', 'mu'])

        return dm

    def _compute_expt_stats(self, mode_method = 'kde_seaborn') -> Tuple[float, float, float]:
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
        return median_expt, mean_expt, mode_expt


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
        plt.savefig(f'{DIR}/KDE_mode_fit_{self.stock_index}_{self.nyears}years.png')

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
        fig.savefig(f'{DIR}/histogram_{self.stock_index}_{self.nyears}years.png')


    def _cut_left_tail(self, threshold: float = 2.) -> float:
        """ Cut the left tail of the distribution (zombie stocks)

        Args:
            threshold (float): logarithmic threshold used to cut the left tail of the distribution. Default is 2, i.e. thresh = log 2.
        Returns:
            delta_mean (float): Percentage change in mean after cutting left tail, rounded to one decimal place.
        """
        # full dataset mean
        mean = self.mu.mu.mean()

        # threshold for cutting left tail
        df = self.mu.mu[np.log(self.mu.mu.astype(float)) > threshold]

        # mean after cutting left tail
        mean_cut = df.mean()

        # percentage change in mean
        delta_mean = 100.*(mean_cut-mean)/mean 
        return round(delta_mean,1)


    def _cut_right_tail(self, cut_ratio: float) -> float:
        """ Cut right tail of the distribution (best stocks) and returns the percentage change in mean after cutting right tail.
        Cut ratio in defined as the quantile of the right tail to be cut.

        Args:
            cut_ratio (float): ratio of the right tail to be cut.
        Returns:
            delta_mean (float): Percentage change in mean after cutting right tail, rounded to one decimal place.
        """
        # full dataset mean
        mean = self.mu.mu.mean()

        # threshold for cutting right tail. 
        threshold = self.mu.mu.quantile(q=1.-cut_ratio)
        df = self.mu.mu[self.mu.mu < threshold]

        # mean after cutting right tail
        mean_cut = df.mean()

        # percentage change in mean
        delta_mean = 100.*(mean_cut-mean)/mean     
        return round(delta_mean,1)    


    def _get_reverse_cumulative_histogram(self) -> pd.DataFrame:
        """ Plot cumulative histogram in reverse order, i.e. from the right tail to the left tail.
        Returns:
            df (pd.DataFrame): dataframe with the percentage of the right tail cut and the percentage change in mean after cutting the right tail.
        """
        step = 0.01

        # cumulative function of the right tail
        cumulative = list(zip(range(1,100),[-1.*self._cut_right_tail(cut_ratio=step*i) for i in range(1,100)]))

        df = pd.DataFrame(cumulative, columns=['percent_cut_tail', 'contribution'])
        return df

    def _plot_reverse_cumulative_histogram(self) -> None:
        """ Plot and save the cumulative histogram in reverse order """
        fig, ax = plt.subplots()

        df = self._get_reverse_cumulative_histogram()
        df.plot.bar(x='percent_cut_tail', y='contribution', rot=0, color='black', alpha=0.35, ax=ax, label='')

        ax.set_xlim([0,100])
        ax.set_ylim([0,100])

        ax.set_xticks(ticks=[0,25,50,75,100])
        ax.set_xticklabels([0,25,50,75,100])

        ax.set_xlabel("percentage cut", size=15)
        ax.set_ylabel("cumulative contribution",size=15)

        plt.title(f"{self.stock_index} cumulative sum contributions")

        ax.tick_params(direction='in', length=6, width=1.0, colors='black', grid_color='grey', grid_alpha=0.5)
        ax.legend()
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)

        # save the figure
        DIR = 'results/cumulative_plots_'
        os.makedirs(DIR, exist_ok=True)
        plt.savefig(f'{DIR}/cumulative_{self.stock_index}.png',bbox_inches='tight')


    def _empirical_distribution_table(self) -> dict:
        """ Gather results of the empirical distribution analysis in a table. """

        delta_mean_ltail = self._cut_left_tail()

        delta_mean_rtail_5p = self._cut_right_tail(cut_ratio=0.05)
        delta_mean_rtail_10p = self._cut_right_tail(cut_ratio=0.10)
        delta_mean_rtail_25p = self._cut_right_tail(cut_ratio=0.25)

        median_expt, mean_expt, mode_expt = self._compute_expt_stats(mode_method='kde_sklearn')

        results = {'category': self._get_index_category(),
                   'years': self.nyears,
                   'n_stocks': len(self.tickers),
                   'n_stocks_data': len(self.mu),
                   'dmean_ltail': delta_mean_ltail,
                   'dmean top 5%': delta_mean_rtail_5p,
                   'dmean top 10%': delta_mean_rtail_10p,
                   'dmean top 25%': delta_mean_rtail_25p,
                   'mean': mean_expt,
                   'median': median_expt,
                   'mode': mode_expt,
                   'mean/median': mean_expt/median_expt,
                   'mean/mode': mean_expt/mode_expt
                   }
        return results


    def _scipy_fit_lognormal(self):
        """ Fit mu histogram with lognormal distribution
        """
        log_fit = scipy.stats.lognorm.fit(self.mu.mu, floc=0)
        return log_fit


    def _plot_histogram_fit(self, save_data: bool = False, loglog: bool = False) -> None:
        """ Plot histogram and its lognormal fit on one plot.
        Attention: scipyfit is not very good, because it takes into account the whole range of the histogram including the left (zero-inflated) tail.
        Moreover, it should fit the Diaconis-Freedman histogram, since number of bins changes the fit.
        See QQ-plot for a better fit quality assesment.

        Args:
            save_data (bool): if True, save the histogram data and the lognormal fit data
            loglog (bool): if True, plot the histogram and the fit on loglog scale

        Outputs are saved in the results/histogram_fit directory:
            histogram_fit_{self.stock_index}_{self.nyears}years.png (png): histogram with lognormal fit on top

            if save_data is True:
                histogram_{self.stock_index}_{self.nyears}years.csv (csv): raw data of the histogram
                lognormal_{self.stock_index}_{self.nyears}years.csv (csv): raw data of the lognormal fit
        """
        # fit the lognormal distribution using scipy. Fit parameters from self._set_lognormal_fit() -> self.log_fit
        x = np.linspace(0.001, 50, 1000)
        f_logn = lognorm.pdf(x, self.log_fit[0], self.log_fit[1], self.log_fit[2])

        # find bin edges using doubled Freedman-Diaconis rule
        r = self.mu.mu.copy()
        # automatic binning with the Freedman-Diaconis rule
        bin_edges = np.histogram_bin_edges(r, bins='fd')

        # double the freedman-diaconis bins
        bin_step = (bin_edges[1] - bin_edges[0])/2.
        bin_edges_middle = [bin_edges[i-1] + bin_step for i in range(1,len(bin_edges))]
        bin_edges = sorted(bin_edges.tolist() + bin_edges_middle)

        fig, ax = plt.subplots(1,1)

        # plot the fit 
        ax.plot(x, f_logn, 'r-', lw=3, alpha=0.6, label='lognorm pdf')

        # plot the histogram
        data = hist(self.mu.mu, bins=bin_edges, range=(0,20), ax=ax, histtype='stepfilled', alpha=0.2, density=True, label='standard histogram')

        ax.legend(loc='best', frameon=False)

        plt.xlabel(r"Stock price ratio $\frac{X(t=T)}{X(t=0)}$")
        plt.ylabel("Frequency")

        plt.title(f"{self.stock_index} distribution scipy fit ")

        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")

        plt.grid()
        plt.xlim(0,20)

        # save the figure
        DIR = 'results/histogram_fit'
        os.makedirs(DIR, exist_ok=True)
        plt.savefig(f'{DIR}/histogram_fit_{self.stock_index}_{self.nyears}years.png')

        if save_data:
            # extract the histogram data
            x_hist = [(data[1][i]+data[1][i-1])/2 for i in range(1, len(data[1]))]
            y_hist = data[0].tolist()

            # save the histogram data
            df_hist = pd.DataFrame({'x': x_hist, 'y': y_hist})
            df_hist.to_csv(f"{DIR}/histogram_{self.stock_index}_{self.nyears}years.csv", index=False)

            # save the lognormal fit data
            with open(f'{DIR}/lognorm_{self.stock_index}_{self.nyears}years.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(x,f_logn))


    def _scipy_fit(self, cut_left_tail: bool=True) -> dict:
        """ Fit total return emprirical distribution without the left tail.
    
        Args:
            cut_left_tail (bool): if True, cut the left tail of the distribution (i.e. the zero-inflated part)
        Returns:
            dict: A dictionary containing statistics of the lognormal fit, such as mu, sigma, mean, median, mode, sigma^2, C.
        """
        if cut_left_tail:
            df = self.mu.mu[np.log(self.mu.mu.astype(float)) >= -2]
        else:
            df = self.mu.mu

        # fit lognormal distribution
        log_fit = scipy.stats.lognorm.fit(df, floc=0)

        logn_mu, logn_sigma = log_fit[2], log_fit[0]
        logn_mu = np.log(logn_mu)

        logn_median, logn_mean, logn_mode = np.exp(logn_mu), np.exp(logn_mu + 0.5 * logn_sigma**2), np.exp(logn_mu - logn_sigma**2)

        # C parameter from the paper https://doi.org/10.1140/epjb/e2003-00131-6
        C = np.sqrt(np.exp(logn_sigma*logn_sigma) - 1.)

        # dictionary with scipy fit parameters
        results = {'mu': logn_mu,
                    'sigma': logn_sigma,
                    'mean': logn_mean,
                    'median': logn_median,
                    'mode': logn_mode,
                    'sigma2': logn_sigma*logn_sigma,
                    'C': C}
        return results

    def _pymc3_fit(self, draws: int = 5000, tune: int = 3000, cut_left_tail: bool=True) -> dict:
        """ Fit empirical histogram distribution with PyMC3
        Args:
            draws (int): number of draws
            tune (int): number of tuning steps  
        Returns:
            dict: A dictionary containing statistics of the lognormal fit, such as mu, sigma, mean, median, mode, sigma^2, C. 
        """

        r = self.mu.mu.copy()

        if cut_left_tail:
            r = r[np.log(r) > -2]

        model = pm3.Model()

        with model:

            # Define parameters for LogNormal distribution
            muh = pm3.Normal('muh', 0.1, 1)                           # Mean stock drift: \hat{\mu}
            sgmah = pm3.HalfNormal('sigmah', 1)                       # Std stock variation: \hat{\sigma}
            sgma = pm3.HalfNormal('sigma', 1)                         # Volatility: \sigma

            x = pm3.LogNormal('x', mu = muh * self.nyears - 0.5 * sgma**2 * self.nyears,
                                   sigma = np.sqrt(sgma**2 * self.nyears + sgmah**2 * self.nyears**2),
                                   observed = r)

            # instantiate sampler
            step = pm3.NUTS() 

            result = pm3.sample(draws=draws, target_accept=0.98, tune=tune)
            stats = az.summary(result, kind="stats")
            print("\n")
            print(" === PYMC3 FIT === ")
            print(stats)

            # Posterior. Select only last 2000 (most stable) since early trials are not very stable
            ppc = pm3.sample_posterior_predictive(result, 2000)

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


    def _fitter_find_best_distribution(self, cut_left_tail: bool = True) -> pd.DataFrame:
        """ Find best distribution with Fitter
        Read more: https://medium.com/the-researchers-guide/finding-the-best-distribution-that-fits-your-data-using-pythons-fitter-library-319a5a0972e9
        Args:
            cut_left_tail (bool): cut the left tail of the histogram. Default: True
        Returns:
            summary (pd.DataFrame): summary of the best distribution
        """
 
        r = self.mu.mu.copy()

        if cut_left_tail:
            r = r[np.log(r) > -2]

        # run fit over all common distributions
        dists = get_common_distributions()

        fig, ax = plt.subplots(1,1)
        f = Fitter(r, distributions=dists, xmin=0, xmax=20)
        f.fit()

        summary = f.summary(10)

        # save the best distribution
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


    def _plot_stock_evolution(self, ticker: str) -> None:
        """ Plot the time evolution of a stock price for all stock in given index
        Args:
            ticker (str): ticker of the stock
        """
        if not ticker in self.tickers:
            raise ValueError(f"Ticker {ticker} not in the list of tickers")

        fig, ax = plt.subplots()

        sns.lineplot(x = 'Year', y = 'Close', data = self.prices[ticker], label = f"Index {self.stock_index}, stock {ticker}", ax=ax)
        sns.scatterplot(x = 'Year', y = 'Close', data = self.prices[ticker], ax=ax)
        ax.set_ylim(bottom=0)

        ax.tick_params(direction='in', length=6, width=1.0, colors='black', grid_color='grey', grid_alpha=0.5)

        plt.xlabel("time")
        plt.ylabel("Closing price after adjustments")

        plt.title(f"Evolution of {ticker} price")

        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.show()