import pandas as pd
import os
from typing import Tuple

def format_dataframe(df: pd.DataFrame, columns: list, mode: str) -> pd.DataFrame:
    """ Format dataframe by dropping columns and rounding numbers 
    Args:
        df (pd.DataFrame): dataframe to format
        columns (list): list of columns to keep
        mode (str): mode of the fit. Must be emprirical, scipy or mcmc
    Returns:
        df (pd.DataFrame): formatted dataframe
    """
    decim = 2 # number of decimal places

    # define columns 
    if mode == 'empirical':
        col_dec = ['mean','mode','median','mean/median','mean/mode']
        col_drop = []
    elif mode == 'scipy':
        col_dec = ['mu','sigma','mean','median','mode','sigma2','C']
        col_drop = []
    elif mode == 'mcmc':
        col_dec = ['logn mean','logn median','logn mode','logn mu','logn sigma',
                   'logn sigma2','C', 'muh','sigmah','sigma', 'lognorm error',
                   'best distr error']
        col_drop = ['muh std','sigma std', 'sigmah std']
    else:
        raise ValueError('Mode must be empirical, scipy or mcmc')

    if not set(col_dec).issubset(columns):
        raise ValueError('col_dec must be a subset of columns')

    if col_drop:
        df = df.drop(col_drop, axis=1)
    
    df[col_dec] = df[col_dec].astype(float).round(decim)

    return df


def prepare_table(results: dict, mode: str, nyears: int) -> pd.DataFrame:
    """ Convert mcmc fit results into a table in latex format 
    Args:
        results (dict): dictionary of results
        mode (str): mode of the fit. Must be emprirical, scipy or mcmc
    Returns:
        df (pd.DataFrame): prepared dataframe
    """
    decim = 2 # number of decimal places

    columns = list(list(results.values())[0].keys())

    df = pd.DataFrame.from_dict(results).T
    df = format_dataframe(df, columns, mode)

    # save results to csv
    DIR = 'results'
    os.makedirs(DIR, exist_ok=True)
    df.to_csv(os.path.join(DIR, f'data_{mode}_{nyears}years.csv'), header=True)

    return df