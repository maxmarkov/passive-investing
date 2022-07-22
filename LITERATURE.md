# Summary of research papers

## Why indexing works? 

[Full text](https://arxiv.org/abs/1510.03550)


__Introduction__. The active management strategy tend to systematically underperform a passive benchmark index. The best performing stocks often perform significantly better than the other stocks in the index. Thus, the average index returns depend heavily on a relatively small set of winners. The authors developed a model where random selection of a subset of securities from an index maximizes both the chances of outperforming the index and underperforming the index, with the latter chances being greater than the former.

A large positive skewness in returns creates a problem for activemanagement. The non-symmetric shape of the distribution of returns means that random selection will deliver a median return that is worse than the average of the full index of the securities. In reality, the histogram of returns to the securities in an index will change year-to-year. **Missing (or underweighting) the securities that significantly outperform other securities is a strong headwind for an active manager to overcome.** 

__A simple model of stock selection from an index__. We assume that the benchmark index contains $N$ stocks $S^{i}, 1 \leq i \leq N$. The dynamics of stock $S^{i}$ over time $t \in [0,T]$ follows a geometric Brownian motion with stock drifts being distrubuted as $\mu_i = \mathscr{N} (\bar{\mu}, \bar{\sigma}^2)$ . We assume that individual stocks maintain their drift $\mu_i$ over time and that the starting value is $S^i_0 = 1$ for all stocks. At time $t = T$ the stock value is 

$$
S^i_{T} \sim e^{\bar{\mu}T - \frac{1}{2}\sigma^2T + \sqrt{\sigma^2T + \bar{\sigma}^2 T^2} Z} 
$$

where $Z = \mathscr{N} (0,1)$. An index return by the equally weighted portfolio is 

$$
I_t^N = \frac{1}{N}\sum_{i=1}^{N}S_{t}^{i}
$$
