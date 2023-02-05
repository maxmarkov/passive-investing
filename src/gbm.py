import os
import numpy as np
import pandas as pd 
import pickle

import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, HuberRegressor

from scipy import stats
from scipy.stats.stats import pearsonr

from scipy.optimize import minimize

from collections import defaultdict

def drift_sigma_maxlikelihood(price: pd.DataFrame, verbose: bool = False):
    """ Maximum likelihood estimator for the drift and volatility of a GBM
    Reference: 
        https://parsiad.ca/blog/2020/maximum-likelihood-estimation-of-gbm-parameters/

    Args:
        price (pd.DataFrame): price series
    Returns:
        drift (float): drift
        mu (float): average return    
        sigma (float): volatility
    """
    price = price['Close'].values
    log_price = np.log(price)                    # X
    delta = log_price[1:] - log_price[:-1]       # ΔX
    n_samples = delta.size                       # N
    n_years = len(price)                         # δt
    total_change = log_price[-1] - log_price[0]  # δX
    
    vol2 = (-total_change**2 / n_samples + np.sum(delta**2)) / n_years

    mu = total_change / n_years
    sigma = np.sqrt(vol2)
    
    drift = mu + 0.5 * vol2

    r1 = np.exp(n_years*drift)
    r2 = np.exp(n_years*mu)

    if verbose:
        print('GBM parameters: drift = {:.4f}, volatility = {:.4f}'.format(drift, sigma))

    return drift, mu, sigma, r1, r2

def compute_gbm_params(data: defaultdict, index_name: str) -> pd.DataFrame:
    """ Compute the GBM parameters for all stocks in a single index
    Args:
        data (defaultdict): dictionary with the price data for an index
        index_name (str): name of the index
    Returns:
        df (pd.DataFrame): dataframe with the GBM parameters
    """
    df_dict = {}

    for ticker in data[index_name].keys():

        df = data[index_name][ticker]
        n_years_traded = sum(abs(df['Close'].diff()) > 0)
        try:
            prices = data[index_name][ticker]
            rho = prices['Close'].iloc[-1] / prices['Open'].iloc[0]
            # cut the left tail
            if np.log(rho) > -2:
                drift, mu, sigma, r1, r2 = drift_sigma_maxlikelihood(prices)
                df_dict[ticker] = {'years_traded': n_years_traded, 'rho': rho, 'drift': drift, 'mu': mu, 'sigma': sigma, 'r1': r1, 'r2': r2}
        except:
            print(f'Cannot do {ticker}')
    
    df = pd.DataFrame.from_dict(df_dict, orient='index')

    # select only stocks that have been traded for at least 15 years
    df = df[df.years_traded >= 15]

    df = df.dropna()
    df = df.loc[~(df['sigma']==0)]

    return df

def plot_kde(df: pd.Series, title: str, xlabel: str, index_name: str, double_bins: bool=False) -> None:
    """ Plot the kernel density estimate for a single index
    Args:
        df (pd.Series): dataframe with the index data
        title (str): title of the plot
        xlabel (str): label of the x-axis
        index_name (str): filename to save the plot   
        double_bins (bool): double the number of bins in Freedman-Diaconis rule   
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # automatic binning with the Freedman-Diaconis rule
    bin_edges = np.histogram_bin_edges(df, bins='fd')

    if double_bins:
        # double the freedman-diaconis bins
        bin_step = (bin_edges[1] - bin_edges[0])/2.
        bin_edges_middle = [bin_edges[i-1] + bin_step for i in range(1,len(bin_edges))]
        bin_edges = sorted(bin_edges.tolist() + bin_edges_middle)

    ax.hist(df, color='blue', alpha=0.3, density=True, bins=bin_edges)
    sns.kdeplot(df, fill=True, color='red', ax=ax)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Density', fontsize=16)

    ax.set_ylim(ymin=0.0)

    ax.set_title(title, fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=15, direction='in', length=8, width=1)

    plt.grid()
    DIR_KDE = 'results/gbm_parameters/kde_{}'.format(xlabel)
    os.makedirs(DIR_KDE, exist_ok=True)

    filename = '{}/kde_{}_index_{}.png'.format(DIR_KDE, xlabel, index_name)
    plt.savefig(filename)

def plot_drift_vs_sigma_fit(X: np.array, Y: np.array, Y_pred: np.array, index_name: str, title: str) -> None:
    """ Plot the linear regression fit for a single index
    Args:
        X (np.array): X data
        Y (np.array): Y data
        Y_pred (np.array): predicted Y data
        index_name (str): name of the index
        title (str): title of the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    plt.scatter(X, Y, alpha=0.3, label='Data')
    plt.plot(X, Y_pred, color='red', label='Linear regression fit', linewidth=3, alpha=0.5)

    ax.set_xlabel('Volatility $\sigma$', fontsize=16)
    ax.set_ylabel('Return $\mu$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=15, direction='in', length=8, width=1)
    
    ax.set_title(title, fontsize=20)
  
    plt.grid()
    plt.legend(fontsize=15)

    DIR = 'results/gbm_parameters' 
    os.makedirs(DIR, exist_ok=True) 
    plt.savefig(f'{DIR}/scatter_fit_index_{index_name}.png')

def linear_regression_fit(df: pd.DataFrame, index_name: str, fit_intercept: bool = True, huber: bool=True) -> tuple:
    """ Fit a linear () regression model to the data:
        Y = b + a*X + error
    where the slope of the line is a, while b is the intercept.

    Args:
        df (pd.DataFrame): dataframe with the index data
        fit_intercept (bool): whether to fit the intercept. Default is True.
        huber (bool): use L2-regularized linear regression model that is robust to outliers. Default is True.
    Returns:
        a (float): slope of the line
        b (float): intercept of the line
        r2 (float): regression score function to estimate fit quality
    """

    df = df.dropna(subset=['mu', 'sigma'])

    X = df.sigma.values.reshape(-1, 1)
    Y = df.mu.values.reshape(-1, 1)

    if huber:
        lr_model = HuberRegressor(fit_intercept=fit_intercept)
    else:
        lr_model = LinearRegression(fit_intercept=fit_intercept)

    lr_model.fit(X, Y)

    Y_pred = lr_model.predict(X)

    # Get fit coefficients
    a = lr_model.coef_[0]
    if fit_intercept:
        b = lr_model.intercept_
    else:
        b = 0.0

    r2 = r2_score(Y, Y_pred)

    # Pearson correlation coefficient between the two variables
    corr = pearsonr(df.drift, df.sigma)[0]

    def sign(x): 
        return int(x>0) 

    if sign(b) > 0: 
        title = 'index: '+ index_name +'  $\mu$=' + str(np.round(a,3)) + "*$\sigma$" + '+' + str(np.round(b,3)) + '  R2='+str(np.round(r2,3))
    else:
        title = 'index: '+ index_name +'  $\mu$=' + str(np.round(a,3)) + "*$\sigma$" + str(np.round(b,3)) + '  R2='+str(np.round(r2,3))

    plot_drift_vs_sigma_fit(X, Y, Y_pred, index_name, title)

    return a, b, r2, corr

def fit_drift_skewnormal(df: pd.DataFrame) -> tuple:
    """ Fit a distribution to the given data """

    def your_density(x):
        return -stats.skewnorm.pdf(x,*params)

    data=np.array(df, dtype=float) 

    # find parameters to fit a skewnorm to the data
    params = stats.skewnorm.fit(data)
    
    # get the parameters of the fitted distribution
    mean, var, skew, kurt = stats.skewnorm.stats(*params, moments='mvsk')  

    mode = minimize(your_density,0).x

    results = {'drift_mean': mean,
                'drift_mode': mode[0],
                'drift_std': np.sqrt(var),
                'drift_skew': skew,
                'drift_skewp': params[0],
                'drift_loc': params[1],
                'drift_scale': params[2],
                'mu_mean': np.mean(data),
                'mu_median': np.median(data), 
                'mu_std': stats.median_abs_deviation(data),
                'mu_mean_to_median': np.mean(data)/np.median(data),
                'mu_std_to_mean': np.std(data)/np.mean(data)}
    return results

def plot_drift_distribution_fits(df: pd.DataFrame, index_name: str) -> None:
    """ Plot the distribution of the drift and the fitted skewnorm distribution """

    fig, ax = plt.subplots(figsize=(10, 8))

    data=np.array(df, dtype=float) 

    sns.distplot(data, kde_kws={'label': 'KDE plot'}, label='histogram', ax=ax)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    # distribution fitting                
    params_skew = stats.skewnorm.fit(data)
    params_alapl = stats.laplace_asymmetric.fit(data)
 
    ax.plot(x, stats.skewnorm.pdf(x, *params_skew), label='skewnorm fit')
    ax.plot(x, stats.laplace_asymmetric.pdf(x, *params_alapl), label='laplace_asymmetric fit')

    plt.legend()
    plt.title('Distribution of percentage drift $\mu$ for '+index_name)
    plt.xlabel('Percentage drift $\mu$')

    plt.grid()
    #plt.savefig(fig_dir+'drift for index '+index_name+'.png')
    plt.show()

def plot_sigma_distribution_fits(df: pd.DataFrame, index_name: str) -> None:
    """"""

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # draw a histogram and kde of the given data
    data=np.array(df, dtype=float)

    ax = sns.distplot(data, kde_kws={'label':'KDE plot'}, label='histogram', ax=ax)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)  

    bounds = {'a':(0, 10),'loc':(-1,1),'scale':(0.05,1)}
    res3 = stats.fit(stats.gamma, data, bounds)
    params=res3.params

    ax.plot(x, stats.gamma.pdf(x, *params), label='approximated gamma')

    plt.legend()
    plt.title('Distribution of percentage volatility $\sigma$ for '+index_name)
    plt.xlabel('Percentage volatility $\sigma$')

    plt.grid()
    #plt.savefig(fig_dir+'sigma for index '+index_name+'.png')
    plt.show()   