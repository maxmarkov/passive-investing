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

# stat analysis
from scipy import stats as st      
from scipy.stats import lognorm
from sklearn.neighbors import KernelDensity

import pymc3 as pm3
import arviz as az
from fitter import Fitter, get_common_distributions
#import statsmodels.api as sm

from tqdm import tqdm

class StockIndexAnalyzer:

    def __init__(self, prices, stock_index, start_date="2006-12-29", end_date="2021-12-31"):
        """
        Class to analyze Stock Index Price data

        Arguments:
            prices (pd.DataFrame): dataframe with data for all stocks in given index
            stock_index (str): stock index name
        """

        self.stock_index = stock_index
        self.category = self.get_index_category()

        self.start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        assert self.start_date < self.end_date
        self.nyears = self.end_date.year - self.start_date.year + 1

        self.prices = prices
        self.tickers = list(self.prices.keys())
        self.tickers_select = []                      # tickers stock price ratio being at least doubled

        # dataframe with price ratio for all indices
        self.mu = self.compute_stock_price_variation()

        n_stocks = len(self.mu.mu)
        # index with small number of stocks need less bins. Otherwise, we get a uniformal distribution
        if n_stocks < 30:
            self.bins = int(n_stocks)
        else:
            self.bins = 2*int(n_stocks)

        # Statistics: experimental results
        self.median_expt = self.mu.mu.median()
        self.mean_expt = self.mu.mu.mean()
        self.mode_expt = self.compute_expt_mode(method='kde_seaborn')

        self.log_fit = self.scipy_fit_lognormal()


    def get_index_category(self) -> str:
        """ get geographic/sector category for stock index """
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


    def compute_stock_price_variation(self) -> pd.DataFrame:
        """ 
        Within the given index, select all stocks with available price data at `self.start_date` and `self.end_date`.
        Compute price ratio and return at given dates and write all ratio values into `dm` dataframe.  

        Arguments:

        Returns:
            dm (pd.DataFrame): dataframe with index 
        """

        mu_dict = {}

        for ticker in tqdm(self.tickers):

            df = self.prices[ticker]

            if all(elem in df.columns.tolist() for elem in ['Open', 'Close']):

                # For some index dates day is 29, but not 31.
                # Here were make it more robust but locating a by year
                df['year'] = pd.DatetimeIndex(df.index).year
                # stock_analyzer.start_date.year
                price_stt = df.loc[df['year']==self.start_date.year, 'Open'].iloc[0]
                price_end = df.loc[df['year']==self.end_date.year, 'Close'].iloc[0]

                # less robust way requires precise dates everywhere
                #price_stt = df.loc[self.start_date]['Open']
                #price_end = df.loc[self.end_date]['Close']

                if not np.isnan(price_stt) and not np.isnan(price_end):

                    # total return and ratio of prices, multiply by 100 if need percentage
                    return_value = (price_end - price_stt)/price_stt #/n_years
                    ratio_value = price_end / price_stt #/n_years
            
                    mu_dict[ticker] = ratio_value

                    if ratio_value > 2:
                        self.tickers_select.append(ticker)
                        #print(f'Stock {ticker} more than double on average during {n_years} years')

        print(f"Total number of stocks: {len(self.tickers)}")
        print(f"Number of stocks with data between {self.start_date.year} and {self.end_date.year} ({self.nyears} years): {len(mu_dict)}")
        print(f"Number of stocks with data between {self.start_date.year} and {self.end_date.year} with at least doubled price ({self.nyears} years): {len(self.tickers_select)}")

        if len(mu_dict) == 0:
            raise ValueError(f"No prices found at specified dates for any stock in {self.stock_index}.")

        dm = pd.DataFrame(mu_dict.items(), columns=['ticker', 'mu'])

        return dm


    def compute_expt_mode(self, method: str = 'kde_sklearn') -> float:
        """ 
        Compute distribution mode with one of three implemented methods:

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
        y, x, _ = ax[0].hist(self.mu.mu, bins = self.bins)
        ax[0].set_xlim([-1,20])
        ax[0].set_ylim([0,1.1*np.max(y)])
        ax[0].set_title(label='Index return distribution histogram', fontsize=15)
        ax[0].tick_params(axis='both', which='major', labelsize=15)
        ax[0].tick_params(direction='in', length=8, width=1)
        mode_hist = get_mode(x, y)

        # Approach 2: fit with Gaussian Kernel Density Estimation [scikit-learn]
        #             large bandwidth parameter is required to smooth the curve 
        #             and make curve unimodal
        kde_skln = KernelDensity(kernel="gaussian", bandwidth=5.0)
        kde_skln.fit(self.mu.mu.to_numpy()[:, np.newaxis])

        x_skln = np.linspace(-1, 20, 10000)[:, np.newaxis]
        y_skln = np.exp(kde_skln.score_samples(x_skln))

        ax[1].plot(x_skln, y_skln, c='cyan')
        ax[1].set_xlim([-1,20])
        ax[1].set_ylim([0,1.1*np.max(y_skln)])
        ax[1].set_title(label='Kernel Density Estimation [sklearn]', fontsize=15)
        ax[1].tick_params(axis='both', which='major', labelsize=15)
        ax[1].tick_params(direction='in', length=8, width=1)
        mode_skln = get_mode(x_skln, y_skln)[0]

        # Approach 3: fit with iGaussian Kernel Density Estimation [seaborn]
        #             bandwidth parameter is estimated automatically:
        #             Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#scipy.stats.gaussian_kde
        kde_sbn = sns.kdeplot(data=self.mu.mu, legend=False)
        x_sbn, y_sbn = kde_sbn.get_lines()[0].get_data()
        ax[2].set_xlim([-1,20])
        ax[2].set_ylim([0,1.1*np.max(y_sbn)])
        ax[2].set_title(label='Kernel Density Estimation [seaborn]', fontsize=15)
        ax[2].tick_params(axis='both', which='major', labelsize=15)
        ax[2].set_ylabel('')
        ax[2].set_xlabel('')
        ax[2].tick_params(direction='in', length=8, width=1)
        mode_sbn = get_mode(x_sbn, y_sbn)

        DIR = 'results/kde_mode_fit'
        os.makedirs(DIR, exist_ok=True)
        plt.savefig(f'{DIR}/KDE_mode_fit_{self.stock_index}_{self.nyears}years.png')

        if method == 'kde_seaborn':
            return mode_sbn
        elif method == 'kde_sklearn':
            return mode_skln
        else:
            return mode_hist


    def plot_histogram(self) -> None:
        """ Plot histogram with variations """

        fig, ax = plt.subplots(1,1)

        ax = plt.gca()
        self.mu.hist(column = 'mu', grid = True, bins = self.bins, ax = ax, label=f"{self.stock_index} index")    # most of stock are around zero while a few increase in value >2 (double) times per year on average
        ax.tick_params(direction='in', length=6, width=1.0, colors='black', grid_color='grey', grid_alpha=0.8)

        yh = 2.

        plt.vlines(x=self.median_expt, ymin=0, ymax=yh, color='r', linestyle='--', linewidth=0.8)
        plt.plot(self.median_expt, yh, 'ro', markersize=4, label='Expt median return')
        plt.text(self.median_expt-2, yh+0.2, f"median {round(self.median_expt,1)}", color='r', fontsize=8)

        plt.vlines(x=self.mean_expt, ymin=0, ymax=yh+1.5, color='k', linestyle='--', linewidth=0.8)
        plt.plot(self.mean_expt, yh+1.5, 'ko', markersize=4, label='Expt mean return')
        plt.text(self.mean_expt-2, yh+1.7, f"mean {round(self.mean_expt,1)}", color='k', fontsize=8)

        plt.xlabel(r"Stock price ratio $\frac{X(t=T)}{X(t=0)}$")
        plt.ylabel("Frequency")

        plt.title("Stock price ratio distribution")

        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.xlim([0,20.])

        DIR = 'results/histogram'
        os.makedirs(DIR, exist_ok=True)
        plt.savefig(f'{DIR}/histogram_{self.stock_index}_{self.nyears}years.png')



    def plot_histogram_fit(self, save_data: bool = False) -> None:
        """ plot histogram with variations and the lognormal distribution fit """

        filename = f'distribution_fit_{self.stock_index}.png'

        x = np.linspace(0.001, 50, 1000)

        fig, ax = plt.subplots(1, 1)

        f_logn = lognorm.pdf(x, self.log_fit[0], self.log_fit[1], self.log_fit[2])
        ax.plot(x, f_logn, 'r-', lw=3, alpha=0.6, label='lognorm pdf')
        data = ax.hist(self.mu.mu.values, density=True, histtype='stepfilled', bins=self.bins, alpha=0.2, label="histogram")
        
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


    def scipy_fit_lognormal(self):
        """
        Fit mu histogram with lognormal distribution
        Help:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
        """
        log_fit = scipy.stats.lognorm.fit(self.mu.mu, floc=0)
        return log_fit


    def compare_stats(self):
        """ 
        Compare distribution parameters: scipy fit vs experiment
        """
        print ("\n")
        print(" === SCIPY FIT === ")
        estimated_mu = np.log(self.log_fit[2])
        estimated_sigma = self.log_fit[0]
        print('Scipy lognorm mu =', round(estimated_mu,2), ', Scipy lognorm sigma =', round(estimated_sigma,2))

        median_fit=np.exp(estimated_mu)
        print('Scipy lognorm median = %.2f' %(median_fit))
        
        # let's check fitted values with empirical ones see https://en.wikipedia.org/wiki/Log-normal_distribution
        mean_fit=np.exp(estimated_mu+estimated_sigma**2/2)
        print('Scipy lognorm mean = %.2f' %(mean_fit))

        var_fit=(np.exp(0.5*estimated_sigma**2)-1)*np.exp(2.*estimated_mu+estimated_sigma**2)/self.log_fit[2]**2
        print('Scipy lognorm std = %.2f' %(np.sqrt(var_fit)))

        ### the behaviour of the sum of log normal variable is defined by parameter C 
        C = np.sqrt(np.exp(estimated_sigma**2)-1)
        print('Scipy sigma^2 = %.2f, C = %.2f'%(estimated_sigma**2, C)) # moderatly large distribution as estimated_sigma**2>1 but not >>1
        
        ## compare n and C^2
        #print(f"n = {len(self.mu.mu)}, C^2 = {round(estimated_sigma**2,2)}, n >> C^2") # n>>C^2
        #print(f" C^2 = {C**2}, 3/2 C^2 = {3/2*C**2}")
        print("\n")


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
            print('Summary stats', stats)

            ppc = pm3.sample_posterior_predictive(result, 200)
            trc = pm3.trace_to_dataframe(result[-2000:])        # select only last 2000

            # internal parameters 
            muh_ = trc['muh'].mean()
            sigh_ = trc['sigmah'].mean()
            sig_ = trc['sigma'].mean()

            mu_log_ =  muh_* self.nyears - 0.5 * sig_**2 * self.nyears
            sigma_log_ = sig_**2 * self.nyears + sigh_**2 * self.nyears**2

            mean_log_ = np.exp(mu_log_ + 0.5* sigma_log_**2)
            median_log_ = np.exp(mu_log_)
            mode_log_ = np.exp(mu_log_ - sigma_log_**2)

            print("\n")
            print("INTERNAL PARAMETERS")
            print('muh = %0.3f, sigmah = %0.3f, sigma = %0.3f' %(muh_, sigh_, sig_))

            ### LOGNORMAL POSTERIOR DISTRIBUTION PARAMETERS ################
            print('PYMC3 lognorm median %0.2f' % (np.median(ppc['x'])))
            print('PYMC3 lognorm mean %0.2f' % (ppc['x'].mean()))
            print('PYMC3 lognorm std %0.2f' % (np.sqrt(np.var(ppc['x']))))


            ### CHECK LOGNORMAL via INTERNAL MODEL PARAMETERS (\hat{mu}, \sigma{mu}, \sigma) ################
            print("\n")
            print('Mu contributions %0.2f:' %(mu_log_))
            print('Sigma contributions : %0.2f '% (sigma_log_))
            #print('Mean and median for %0.0f years: %0.2f %0.2f percents'%(T1, (muh*T1+1/2*sigh**2*T1**2)/T1*100,(muh*T1-1/2*sig**2*T1)/T1*100))
            print('Mean computed: %0.2f' %(mean_log_))
            print('Median computed: %0.2f' %(median_log_))
            print('Mode computed: %0.2f' %(mode_log_))

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

        print('Mean (predicted) value of parameters', trc.mean(axis=0))

        sigma_log = np.sqrt(sgma**2*self.nyears + sgmah**2 * self.nyears**2)
        C = np.sqrt(np.exp(sigma_log*sigma_log)-1)
        mean_fit = np.exp(trc.mean(axis=0)[0]+trc.mean(axis=0)[1]**2/2.)
        median_fit = np.exp(trc.mean(axis=0)[0])
        print(f'C = {C}')
        print(f'Mean pymc3 fit = {mean_fit}')
        print(f'Median pymc3 fit = {median_fit}')

        return {'hello': 0}
        
    def find_best_distribution(self) -> None:
        """ 
        Find best distribution fit.
        Read more: https://medium.com/the-researchers-guide/finding-the-best-distribution-that-fits-your-data-using-pythons-fitter-library-319a5a0972e9
        """
        
        fig, ax = plt.subplots(1, 1)

        dists = get_common_distributions()
        f = Fitter(self.mu.mu,  distributions=dists, xmin=0, xmax=100)
        f.fit()

        summary = f.summary(10)
        print(summary.sort_values('sumsquare_error'))

        plt.show()
        plt.savefig(f'distributions_{self.stock_index}.png')

    def plot_qq(self) -> None:
        """ Q-Q plot of the quantiles of x versus the quantiles/ppf of a distribution """
        fig, ax = plt.subplots(1, 1)

        sm.qqplot(np.log(self.mu.mu), line ='s', ax=ax)  # it is good that the right tail is underestimated
        plt.grid()
        plt.savefig(f'qqplot_{self.stock_index}.png')


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
            #plt.savefig(os.path.join(path, f"stock_evolution_{ticker}.png"))
            plt.show()
            plt.cla()
