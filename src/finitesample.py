from scipy.stats import lognorm
from scipy.stats import mode

import numpy as np
import matplotlib.pyplot as plt

def mode_theory(sigma: float, N: float, regime: int) -> float:
    """
    Ratio of a typical mean (mode of mean distribution) of sample size N to the true mean
    ratio = S_n^t / n<x>, where S_n^t is the typical sum of n samples, <x> is the average value of x.

    Regime 1: \sigma**2 << 1 (eq. 30 and below)
    Regime 2: \sigma**2 ~ 1  (eq. 37)
    Regime 3: \sigma**2 >> 1 (eq. 54 and 55)

    Args:
        sigma: standard deviation of the lognormal distribution
        N: sample size
        regime: regime of the lognormal distribution
    Returns:
        ratio: ratio of a typical mean (mode of mean distribution) of sample size N to the true mean
    """
    if regime == 1:
        ratio= 1.0
    elif regime == 2:
        C2 = np.exp(sigma**2) - 1.
        ratio = np.power(1. + C2/N, -3./2.)
    elif regime == 3:
        ratio = np.exp(-3/2.*sigma**2/np.power(N, np.log(3./2.)/np.log(2)))
    else:
        raise ValueError("Regime should be 1, 2 or 3")

    return ratio

def mode_montecarlo(sigma: float, mu: float, N: float, n_mc: int = 1000000, bin_low: float = 0.0001, bin_high=1.1) -> float:
    """ 
    Monte-Carlo simulation of the ratio of a typical mean (mode of mean distribution) of sample size N to the true mean
    Args:
        sigma: standard deviation of the lognormal distribution
        mu: mean of the lognormal distribution
        N: sample size
        n_mc: number of Monte-Carlo simulations
        bin_low: lower bound of the histogram
        bin_high: upper bound of the histogram
    Returns:
        mode_of_ratio: ratio of a typical mean (mode of mean distribution) of sample size N to the true mean    
    """
    true_mean = np.exp(mu+sigma**2/2) 

    # sample from lognormal distribution and compute mean for N stocks portfolio
    r = np.random.lognormal(mu, sigma,size=(n_mc,N))
    r_mean=r.mean(axis=1)

    counts, bins = np.histogram(r_mean/true_mean, bins=np.linspace(bin_low, bin_high, 10000))

    mode_of_ratio = bins[np.argmax(counts)]
    return mode_of_ratio

def get_test_sigma_dict(regime: int = 2) -> dict:
    """ Get mocking lognorormal distribution parameters for regime 1, 2 and 3
    Args:
        regime: regime of the lognormal distribution
    Returns:
        sigma_dict: dictionary of standard deviation of the lognormal distribution
    """
    if regime == 1:
        sigma_dict = {'A': {'mu': 0.0, 'sigma': 0.01},
                      'B': {'mu': 0.0, 'sigma': 0.05},
                      'C': {'mu': 0.0, 'sigma': 0.10}}
    elif regime == 2:
        sigma_dict = {'A': {'mu': 0.0, 'sigma': 0.2},
                      'B': {'mu': 0.0, 'sigma': 0.5},
                      'C': {'mu': 0.0, 'sigma': 1.2}}
    elif regime == 3:
        sigma_dict = {'A': {'mu': 0.0, 'sigma': 5.0},
                      'B': {'mu': 0.0, 'sigma':10.0},
                      'C': {'mu': 0.0, 'sigma':100.0}}
    else:
        raise ValueError("Regime should be 1, 2 or 3")
    return sigma_dict

def get_real_sigma_dict() -> dict:
    """ Lognormal distribution parmameters for several indexes derived from Part 1
    According to the parameters in Table 1 of the main text, we are always in the regime 2.

    Args:
        regime: regime of the lognormal distribution
    Returns:
        sigma_dict: dictionary of mean and standard deviation of the lognormal distribution
    """
    sigma_dict = {'UKX': {'mu': 0.72, 'sigma': 0.83},
                  'NKY': {'mu': 0.21, 'sigma': 0.87},
                  'SPX': {'mu': 0.95, 'sigma': 1.02},
                  'NIFTY': {'mu': 1.65, 'sigma': 1.23},
                  }
    return sigma_dict