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

import pymc3 as pm3
import arviz as az
from fitter import Fitter, get_common_distributions
import statsmodels.api as sm

from tqdm import tqdm

class StockIndexAnalyzer:

    def __init__(self, prices, stock_index, start_date="01/01/2005", end_date="01/01/2022"):
        """
        Class to analyze Stock Index Price data

        Arguments:
            prices (pd.DataFrame): dataframe with data for all stocks in given index
            stock_index (str): stock index name
        """

        if stock_index[-3:] == 'csv':
            self.stock_index = stock_index.split('_')[0]
        else:
            self.stock_index = stock_index

        self.start_date = datetime.strptime(start_date, '%m/%d/%Y').date()
        self.end_date = datetime.strptime(end_date, '%m/%d/%Y').date()
        assert self.start_date < self.end_date
        self.nyears = self.end_date.year - self.start_date.year

        self.prices = prices
        self.tickers = self.prices.ticker.unique()
        self.tickers_select = []

        # dataframe with price ratio for all indices
        self.mu = self.compute_stock_price_variation()
        self.bins = 2*int(len(self.mu.mu))

        # Statistics: experimental results
        self.median_expt = self.mu.mu.median()
        self.mean_expt = self.mu.mu.mean()
        self.std_expt = np.sqrt(self.mu.mu.var())
        #self.C_expt = self.mu.mu.var()

        self.mode_expt = self.compute_expt_mode()

        self.log_fit = self.scipy_fit_lognormal()


    def compute_stock_price_variation(self) -> pd.DataFrame:
        """ 
        Within the given index, select all stocks with available price data at `self.start_date` and `self.end_date`.
        Compute price ratio and return at given dates and write all ratio values into `dm` dataframe.  

        Arguments:

        Returns:
            dm (pd.DataFrame): dataframe with index 
        """
        mu_dict = {}

        counter = 0

        for ticker in tqdm(self.tickers):

            df = self.prices[self.prices.ticker==ticker].sort_values(by='date')
            
            price_stt = df[df['date']==self.start_date.strftime("%Y-%m-%d")]
            price_end = df[df['date']==self.end_date.strftime("%Y-%m-%d")]

            if not price_stt.empty and not price_end.empty:

                # total return and ratio of prices, multiply by 100 if need percentage
                return_value = (price_end.adjclose.iloc[0] - price_stt.adjclose.iloc[0])/price_stt.adjclose.iloc[0]#/n_years
                ratio_value = price_end.adjclose.iloc[0] / price_stt.adjclose.iloc[0]#/n_years
                
                mu_dict[ticker] = ratio_value
                counter += 1

                if ratio_value > 2:
                    self.tickers_select.append(ticker)
                    #print(f'Stock {ticker} more than double on average during {n_years} years')

        print(f"Total number of stocks: {len(self.tickers)}")
        print(f"Number of stocks with data between {self.start_date} and {self.end_date}: {counter}")
        print(f"Number of stocks with data between {self.start_date} and {self.end_date} with doubled price: {len(self.tickers_select)}")

        if len(mu_dict) == 0:
            raise ValueError(f"No prices found at specified dates for any stock in {self.stock_index}.")

        dm = pd.DataFrame(mu_dict.items(), columns=['ticker', 'mu'])

        ## additional filter
        #dm = dm[dm['mu'] < 50.]

        return dm


    def compute_expt_mode(self) -> float:
        """ 
        Compute mode as a number in a histogram that appears the most often.
        Arguments:
        Returns:
            mode (float): histogram mode
        """
        plt.figure(1)
        y, x, _ = plt.hist(self.mu.mu, bins = self.bins)
        plt.close(1)
        indx =  np.argmax(y)
        mode = x[indx]
        return mode


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
        plt.show()



    def plot_fit_histogram(self, save: bool = False) -> None:
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

        if save:
            df_hist.to_csv(f"histogram_{self.stock_index}.csv", index=False)
            #self.mu.mu.to_csv(f"histogram_.csv")

            import csv
            with open(f'lognorm_{self.stock_index}.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(x,f_logn))
    
        ax.legend(loc='best', frameon=False)

        plt.xlabel(r"Stock price ratio $\frac{X(t=T)}{X(t=0)}$")
        plt.ylabel("Frequency")

        plt.title(f"{self.stock_index} distribution fit")

        plt.grid()
        plt.xlim(0,50)
        #plt.savefig(filename)
        plt.show()


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


    def pymc3_fit(self, draws: int = 5000, tune: int = 3000) -> None:
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
            ##########################################################################

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
                            var_names=["muh", 'sigmah', "sigma"],
                            #ref_val=0,
                            hdi_prob=0.95,
                            figsize=(20, 5))
            fig.suptitle(f'{self.stock_index} log norm fit', fontsize=12)
            plt.savefig(f'pymc3_posterior_{self.stock_index}.png')
            plt.show()

        trc = pm3.trace_to_dataframe(result)

        print('Mean (predicted) value of parameters', trc.mean(axis=0))

        sigma_log = np.sqrt(sgma**2*self.nyears + sgmah**2 * self.nyears**2)
        C = np.sqrt(np.exp(sigma_log*sigma_log)-1)
        mean_fit = np.exp(trc.mean(axis=0)[0]+trc.mean(axis=0)[1]**2/2.)
        median_fit = np.exp(trc.mean(axis=0)[0])
        print(f'C = {C}')
        print(f'Mean pymc3 fit = {mean_fit}')
        print(f'Median pymc3 fit = {median_fit}')
        
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