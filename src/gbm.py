import os
import numpy as np
import pandas as pd 
import pickle

import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from scipy.stats.stats import pearsonr

from collections import defaultdict

# ====== Maximum likelihood estimation ======#
from tabnanny import verbose

def read_data_pickle(filepath):
    """
    Read data from pickle file
    Parameters:
        filepath (str): path to pickle file
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data

def get_all_indices(exclude_indices: bool = True):
    """
    Get all indices from the data folder
    Parameters:
        exclude_indices (bool): exclude indices with less than 1000 data points
    """
    exclude_list = ['HSI', 'STI', 'S5RLST', 'RTSI$']

    indices = {}

    indices['US'] = ['SPX','CCMP','RIY','RTY','RAY','RLV','RLG','NBI']
    indices['S&P500'] = ['S5COND','S5CONS','S5ENRS','S5FINL','S5HLTH','S5INFT','S5MATR','S5TELS','S5UTIL','S5INDU', 'S5RLST']
    indices['EU'] = ['DAX','CAC','UKX','BEL20','IBEX','KFX','OMX','SMI']
    indices['APAC'] = ['AS51','HSI','STI']
    indices['JP'] = ['NKY','TPX']
    indices['BRIC'] = ['IBOV','NIFTY','MXIN','SHCOMP','SHSZ300','RTSI$']

    indices_all = [item for indx in indices.values() for item in indx]

    if exclude_indices:
        indices_all = [i for i in indices_all if i not in exclude_list]

    return indices_all

def drift_sigma_maxlikelihood(price: pd.DataFrame, verbose: bool = False):
    """ 
    Maximum likelihood estimator for the drift and volatility of a GBM
    Reference: 
        https://parsiad.ca/blog/2020/maximum-likelihood-estimation-of-gbm-parameters/

    Parameters:
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

    print('n_years', n_years, 'n_samples', n_samples)
    
    vol2 = (-total_change**2 / n_samples + np.sum(delta**2)) / n_years

    mu = total_change / n_years
    sigma = np.sqrt(vol2)
    
    drift = mu + 0.5 * vol2

    if verbose:
        print('GBM parameters: drift = {:.4f}, volatility = {:.4f}'.format(drift, sigma))

    return drift, mu, sigma

def compute_gbm_params(data: defaultdict, index_name: str) -> pd.DataFrame:
    """ 
    Compute the GBM parameters for all stocks in a single index
    Parameters:
        data (defaultdict): dictionary with the price data for an index
        index_name (str): name of the index
    Returns:
        df (pd.DataFrame): dataframe with the GBM parameters
    """
    df_dict = {}
    for ticker in data[index_name].keys():
        try:
            prices = data[index_name][ticker]
            rho = np.log(prices['Close'].iloc[-1] / prices['Open'].iloc[0])
            if rho > -2:
                drift, mu, sigma = drift_sigma_maxlikelihood(prices)
                df_dict[ticker] = {'drift': drift, 'mu': mu, 'sigma': sigma}
        except:
            print(f'Cannot do {ticker}')

    df = pd.DataFrame.from_dict(df_dict, orient='index')
    return df

#==== KDE plot ====#
def plot_kde(df: pd.Series, title: str, xlabel: str, filename: str) -> None:
    """ 
    Plot the kernel density estimate for a single index
    Parameters:
        df (pd.Series): dataframe with the index data
        title (str): title of the plot
        xlabel (str): label of the x-axis
        filename (str): filename to save the plot      
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.kdeplot(df, fill=True, color='red')

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Density', fontsize=16)

    ax.set_ylim(ymin=0.0)

    ax.set_title(title, fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=15, direction='in', length=8, width=1)

    plt.grid()
    plt.savefig(filename)

# ====== Linear fit of drift vs volatility ======#

def plot_drift_vs_sigma_fit(X: np.array, Y: np.array, Y_pred: np.array, index_name: str, title: str, dir: str) -> None:
    """
    Plot the linear regression fit for a single index
    Parameters:
        X (np.array): X data
        Y (np.array): Y data
        Y_pred (np.array): predicted Y data
        index_name (str): name of the index
        title (str): title of the plot
        dir (str): directory to save the plot
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
    plt.savefig(f'{dir}/scatter_fit_index_{index_name}.png')

def linear_regression_fit(df: pd.DataFrame, fit_intercept: bool, index_name: str, dir: str) -> dict:
    """
    Fit a linear regression model to the data:
        Y = b + a*X + error
    where the slope of the line is a, while b is the intercept.

    Parameters:
        df (pd.DataFrame): dataframe with the index data
    Returns:
        results (dict): dictionary with the linear regression parameters and fit metrics
    """
    df = df.dropna(subset=['mu', 'sigma'])

    X = df.sigma.values.reshape(-1, 1)
    Y = df.mu.values.reshape(-1, 1)

    lr_model = LinearRegression(fit_intercept=fit_intercept)
    lr_model.fit(X, Y)

    Y_pred = lr_model.predict(X)

    # Get fit coefficients
    a = lr_model.coef_[0][0]
    if fit_intercept:
        b = lr_model.intercept_[0]
    else:
        b = 0.0

    #  Get regression score function to estimate fit quality.
    r2 = r2_score(Y, Y_pred)

    # Pearson correlation coefficient between the two variables
    corr = pearsonr(df.drift, df.sigma)[0]
    
    results = {'index': index_name, 'a': a, 'b': b, 'r2': r2, 'corr': corr}

    if np.sign(b) > 0:
        title = f'Index {index_name}, $\mu$ = {np.round(a,3)} *$\sigma$ + {np.round(b,3)}, R2 = {np.round(r2,3)}'
    else:
        title = f'Index {index_name}, $\mu$ = {np.round(a,3)} *$\sigma$ - {np.round(np.abs(b),3)}, R2 = {np.round(r2,3)}'
        
    plot_drift_vs_sigma_fit(X, Y, Y_pred, index_name, title, dir)

    return results

# ==== Plot histogram of drift and volatility ==== #
def plot_histogram(df: pd.Series) -> None:
    """ """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plt.hist(df)
    #sns.histplot(df, kde=True, color='red')
    
    ax.set_xlabel('Average return', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    
    ax.set_ylim(ymin=0.0)
    
    ax.set_title('Average return distribution', fontsize=20)
    
    ax.tick_params(axis='both', which='major', labelsize=15, direction='in', length=8, width=1)
    
    plt.grid()
    plt.show()
    #plt.savefig('drift.png')