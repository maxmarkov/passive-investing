import numpy as np
import random
import matplotlib.pyplot as plt

def drift_distribution(r_median, r_mean, sigma, T):
    """ Get git distribution parameters 
    r_median (float): median index return
    r_mean (float): expected index return
    simga (float): stock volatility
    T (int): period of time in years
    """
    mu_drift = (np.log(1.+r_median) + 0.5 * sigma**2 * T) / T
    sigma_drift = np.sqrt(2. * np.log(1. + r_mean) - 2.* mu_drift * T) / T
    return mu_drift, sigma_drift

def price_model(t: int, mu_b: float, sigma_b: float, sigma: float) -> float:
    """ """
    s0 = 1. # initial price
    s = s0 * np.exp(mu_b * t - 0.5 * sigma**2 * t + np.sqrt(sigma**2 * t + sigma_b**2 * t**2) * np.random.normal(loc=0, scale=1))
    return s

def stock_price(T: int, mu_b: float, sigma_b: float, sigma: float):
    ''' compute price dynamics over T years '''
    for t in range(T):
        s = price_model(t, mu_b, sigma_b, sigma)
    return s

def plot_data(rate_p, rate_a, filename=None):
    """ """
    assert len(rate_p) == len(rate_a)
    x = range(1,len(rate_a)+1)
    plt.plot(x, rate_p, c='k', label='passive')
    plt.plot(x, rate_a, c='orange', label='active')

    plt.xlabel("Sub-portfolio size")
    plt.ylabel("Rate")

    plt.xlim([0,20])
    plt.ylim([0.2,0.7])
    plt.xticks([0, 5, 10, 15, 20])

    plt.grid(visible=True, alpha=0.5)
    plt.legend()

    plt.title( "Sub-portfolio vs Index")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def main():
    """ """
    N_MC = 100          # number of Monte Carlo trials, N_MC = 10000 in the paper
    
    T = 5                # period in years
    N = 500              # number of stocks
    sigma = 0.2          # generic annual stock volatility 
    r_median = 0.1       # median index return
    r_mean = 0.5         # mean index return
    
    mu_drift, sigma_drift = drift_distribution(r_median, r_mean, sigma, T)
    
    
    rate_passive = []
    rate_active = []
    
    for n_pfl in range(1,21):
        print(f'Sub-portfolio size {n_pfl}')
    
        N_active = 0
        N_passive = 0
        for i_MC in range(N_MC):
        
            # index return by the equally weighted portfolio
            S_list = [stock_price(T, mu_drift, sigma_drift, sigma) for n in range(N)]
            I_mean = sum(S_list)/len(S_list)
            
            # portfolio
            S_pfl = random.sample(S_list, n_pfl)
            I_pfl = sum(S_pfl)/len(S_pfl)
            
            if I_pfl > I_mean:
                N_active += 1
        
        N_passive = N_MC - N_active
    
        rate_passive.append(N_passive/N_MC)
        rate_active.append(N_active/N_MC)
    
    plot_data(rate_p=rate_passive, rate_a=rate_active, filename='figure.png')

if __name__ == "__main__":
    main()
