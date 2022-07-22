import numpy as np
import random
import matplotlib.pyplot as plt

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



N_MC = 10000          # number of Monte Carlo trials, N_MC = 10000 in the paper

T = 5                # period in years
N = 500              # number of stocks
sigma = 0.2          # generic annual stock volatility 
mu_drift = 0.04      # drift mean
sigma_drift = 0.13   # drift variance 


rate_passive = []
rate_active = []

for n_pfl in range(1,5):
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

plt.plot(rate_passive, c='k', label='passive')
plt.plot(rate_active, c='orange', label='active')
#plt.legend(np.round(sigma, 2))
plt.xlabel("Sub-portfolio size")
plt.ylabel("Rate")
plt.xlim([0,20])
plt.ylim([0,1])
plt.legend()
#plt.title( "Realizations of Geometric Brownian Motion with different variances\n $\mu=1$")
plt.savefig("figure.png")
#plt.show()
