# Summary of research papers

## Why indexing works? 

[Full text](https://arxiv.org/abs/1510.03550)


__Introduction__. The active management strategy tend to systematically underperform a passive benchmark index. The best performing stocks often perform significantly better than the other stocks in the index. Thus, the average index returns depend heavily on a relatively small set of winners. The strategy based on random selection of a subset of securities from an index maximizes both the chances of outperforming the index and underperforming the index, with the latter chances being greater than the former.

A large positive skewness in returns creates a problem for active management. The non-symmetric shape of the distribution of returns means that random selection will deliver a median return that is worse than the average of the full index of the securities. In reality, the histogram of returns to the securities in an index will change year-to-year. **Missing (or underweighting) the securities that significantly outperform other securities is a strong headwind for an active manager to overcome.** 

__A simple model of stock selection from an index__. We assume that the benchmark index contains $N$ stocks $S^{i}, 1 \leq i \leq N$. The dynamics of stock $S^{i}$ over time $t \in [0,T]$ follows a [geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion) with stock drifts being distrubuted as $\mu_i = \mathscr{N} (\bar{\mu}, \bar{\sigma}^2)$ . We assume that individual stocks maintain their drift $\mu_i$ over time and that the starting value is $S^i_0 = 1$ for all stocks. The geometric Brownian motion, has [the following solution](https://en.wikipedia.org/wiki/Geometric_Brownian_motion#Solving_the_SDE) at time $t = T$: 

$$
S^i_{T} = S_{0}  \exp{\left(\mu_i T - \frac{1}{2}\sigma^2 T\right)} \exp{\left(\sigma W_T\right)} \sim \exp{\left(\mu T - \frac{1}{2}\sigma^2 T + \sqrt{\sigma^2T+\bar{\sigma}^2T^2}Z\right)}
$$

where $W_{T}$ is a Wiener process, or Brownian motion, $Z\sim \mathscr{N}(0,1)$, $\sigma$ is a stock volatility parameter same for all stocks, $\bar{\mu}$ and $\bar{\sigma}$ are meand and variation of the drift parameter $\mu_i$. The above solution $S_t$ is a log-normally distributed random variable with expected value:

$$
E(S_t) = S_0 \exp{\left(\mu t\right)}
$$

and variance

$$
Var(S_t) = S_0^2 \exp{(2\mu t)}\left[\exp{(\sigma^2 t)} - 1\right]
$$

An index return by the equally weighted portfolio is 

$$
I_t^N = \frac{1}{N}\sum_{i=1}^{N}S_{t}^{i}
$$

Some conclusions:

- The cumulative return of a stock follows a log-$\mathscr{N}(\bar{\mu}T - \frac{1}{2}\sigma^2T, \sigma^2T + \bar{\sigma}^2T^2)$ distribution which is heavily positively skewed.

- The median of the stock distribution $\exp{\left(\bar{\mu}T - \frac{1}{2}\sigma^2T\right)}$, so that over time T more than half of all stocks in the index will underperform the index return by a factor of $\exp{\left(\frac{1}{2}\sigma^2T, \sigma^2T + \bar{\sigma}^2T^2\right)}$


__Monte Carlo simulation__ 

__Conclusions__ The developed model suggests that the high cost of active management can be explained by high chance of underperformance that comes with attempts to select stocks. Stock selection increases the chance of underperformance relative to the chance of overperformance in many circumstences. 

When creating a portfolio combining passive and active strategies, return estimation should be adjusted for the inherent statistical disadvantage of the active manager combined with their higher fees. 
