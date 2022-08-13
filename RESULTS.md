
Table of contents
=================
- [Results](#custom-detection)
- [S&P500](#sp500)
- [Dow Jones](#dow)
- [NASDAQ](#nasdaq)
- [Venture](#venture)
- [Biotech](#biotech)
- [Comparison](#comparison)

<a name="results"></a>
# Results

|           |$\mu$ scipy |   $\mu$ pymc3     |  $\sigma$ scipy  |   $\sigma$ pymc3    |   $C$ scipy     |   $C$ pymc3      |  best distribution | sum square_error |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |
| Biotech   | -0.14   | -0.14 $\pm$ 0.19   |   2.17        |  2.19 $\pm$ 0.14     |   10.50       | 11.50 $\pm$ 4.00   |      lognorm       |    0.011887     |
| Venture   |  2.14   |  2.15 $\pm$ 0.10   |   1.19        |  1.21 $\pm$ 0.07     |    1.78       |  1.82 $\pm$ 0.22   |      lognorm       |    0.005744     |     
| NASDAQ    | -0.05   | -0.05 $\pm$ 0.06   |   2.77        |  2.77 $\pm$ 0.04     |   46.59       | 47.16 $\pm$ 5.80   |      lognorm       |    0.005792     |
| S&P500    |  1.94   |  1.95 $\pm$ 0.05   |   1.03        |  1.04 $\pm$ 0.03     |    1.07       |  1.39 $\pm$ 0.07   |      lognorm       |    0.002876     |
| Dow Jones |  1.98   |  1.98 $\pm$ 0.18   |   0.95        |  0.99 $\pm$ 0.14     |    1.21       |  1.32 $\pm$ 0.33   |       cauchy (lognorm)      |    0.156360 (0.196270)    |
| FTSE100   |  0.58   |  0.58 $\pm$ 0.25   |   1.37        |  1.41 $\pm$ 0.18     |    2.56       |  2.68 $\pm$ 0.99   |       expon  (lognorm)       |    0.694053 (0.706687)     |
| FTSE250   |  0.25   |  0.25 $\pm$ 0.28   |   1.77        |  1.82 $\pm$ 0.20     |    4.68       |  5.66 $\pm$ 3.15   |       exponpow (lognorm)    |    0.425529 (0.527496)    |
| NIFTY50   |  2.90   |  2.9 $\pm$ 0.27   |   1.36        |  1.41 $\pm$ 0.20     |    2.31       |  2.71 $\pm$ 1.16   |       expon (lognorm)       |    0.101994 (0.102374)     |

## Europe

| stock-name | $\mu$ scipy | $\mu$ ymc3 | $\sigma$ scipy | $\sigma$ pymc3 | $C$ scipy   | $C$ pymc3 | best_distribution |  sum square_error |
|----------- |--------- |--------- |------------ |----------- |----------- |-------- |------------------ | ------------- |
|   BEL20    |     -    |    -     |      -      |     -      |    -       |    -    |        - (-)      |     - (-)         |
|   CAC      |   0.94   |  0.93    |    1.17     |   1.28     |  1.72      |  2.34   |  gamma (lognorm)  |     1.18 (  1.23) |
|   DAX      |   1.47   |  1.47    |    0.96     |   1.10     |  6.89      |  1.82   | chi2 (lognorm)    |     0.55 (  0.57) |
|  IBEX      |  -0.04   | -0.04    |    0.70     |   0.82     |  0.80      |  1.06   | exponpow (lognorm)|    85.47 ( 87.87) |
|   KFX      |     -    |    -     |      -      |     -      |    -       |    -    |        - (-)      |     - (-)         |
|   OMX      |     -    |    -     |      -      |     -      |    -       |    -    |        - (-)      |     - (-)         |
|   SMI      |     -    |    -     |      -      |     -      |    -       |    -    |        - (-)      |     - (-)         |
|   UKX      |   0.57   |  0.58    |    1.46     |   1.51     |  2.71      |  3.28   | expon (lognorm)   |     0.48 (  0.52) |

## United States

| stock-name | $\mu$ scipy | $\mu$ pymc3 | $\sigma$ scipy | $\sigma$ pymc3 | $C$ scipy    | $C$ pymc3 | best_distribution |  sum square_error |
|----------- |--------- |--------- |------------ |-----------  |----------- |-------- |------------------ | -------------     |
| CCMP       |  -0.17   |  -0.17 $\pm$ 0.071   |    2.93     | 2.93 $\pm$ 0.050    |    72.20    | 73.46 $\pm$ 10.85 | lognorm  | 0.0076 |
| RIY        |   1.79   |   1.79 $\pm$ 0.036   |    1.01     | 1.01 $\pm$ 0.025    |     1.33    | 1.34 $\pm$ 0.053  | lognorm  | 0.0012 |
| RTY        |   0.70   |   0.70 $\pm$ 0.045   |    1.53     | 1.53 $\pm$ 0.030    |     3.07    | 3.08 $\pm$ 0.165  | lognorm  | 0.0039 |
| RLV        |   1.64   |   1.635 $\pm$ 0.035  |    0.94     | 0.941 $\pm$ 0.025   |     1.19    | 1.194 $\pm$ 0.049 | lognorm  | 0.0010 | 
| RAY        |   1.15   |   1.15 $\pm$ 0.033   |    1.45     | 1.451 $\pm$ 0.023   |     2.68    | 2.686 $\pm$ 0.102 | lognorm  | 0.0009 |
| PEALL      |     -    |         -        |     -       |        -        |      -      |       -       |    -     |  -     |
| SPX        |   1.95   |   1.947 $\pm$ 0.048  |    1.04     | 1.040 $\pm$ 0.034   |     1.39    | 1.398 $\pm$ 0.075 | lognorm  | 0.0029 |
| PEVC       |     -    |         -        |     -       |        -        |      -      |       -       |    -     |        -| 
| NBI        |  -0.14   |  -0.143 $\pm$ 0.193  |    2.16     | 2.176 $\pm$ 0.136   |    10.33    | 11.244 $\pm$ 3.810| lognorm  | 0.0124 |
| RLG        |   2.21   |   2.207 $\pm$ 0.052  |    1.02     | 1.019 $\pm$ 0.036   |     1.34    | 1.353 $\pm$ 0.078 | lognorm  | 0.0029 | 

## Apac countries

| stock-name | $\mu$ scipy | $\mu$  pymc3 | $\sigma$ scipy | $\sigma$ pymc3 | $C$ scipy    | $C$ pymc3 | best_distribution |  sum square_error |
|----------- |--------- |--------- |------------ |-----------  |----------- |-------- |------------------ | -------------     |
| AS51       |  0.82    |  0.821 $\pm$ 0.165 |   1.29      | 1.308 $\pm$ 0.121  |   2.06     |   2.176 $\pm$ 0.455    | expon (lognorm)   | 0.104679 (0.109758) |
| SHCOMP     |    -     |        -       |    -        |      -         |    -       |        -           |                   |                     |
| NIFTY      |    -     |          |    -        |             |    -       |         |                   |                   |
| HSI        |    -     |        -       |    -        |      -         |    -       |        -           |                   |                     |
| MXIN       |    -     |        -       |    -        |      -         |    -       |        -           |                   |                     |
| NKY        |    -     |        -       |    -        |      -         |    -       |        -           |                   |                     |
| STI        |    -     |        -       |    -        |      -         |    -       |        -           |                   |                     |
| SHSZ300    |    -     |        -       |    -        |      -         |    -       |        -           |                   |                     |

<a name="sp500"></a>
## S&P500

![S&P500 distribution](media/distribution_sp500_200bins_7years.png)

![S&P500 trace](media/pymc3_trace_sp500.png)

![S&P500 posterior](media/pymc3_posterior_sp500.png)

![S&P500 qqplot](media/qqplot_sp500.png)

<a name="dow"></a>
## Dow

![Dow distribution](media/distribution_dow_200bins_7years.png)

![Dow trace](media/pymc3_trace_dow.png)

![Dow posterior](media/pymc3_posterior_dow.png)

![Dow qqplot](media/qqplot_dow.png)

<a name="nasdaq"></a>
## NASDAQ

![Nasdaq trace](media/pymc3_trace_nasdaq.png)

![Nasdaq posterior](media/pymc3_posterior_nasdaq.png)

![Nasdaq qqplot](media/qqplot_nasdaq.png)

<a name="venture"></a>
## Venture

![Venture distribution](media/distribution_venture_200bins_7years.png)

![Venture trace](media/pymc3_trace_venture.png)

![Venture posterior](media/pymc3_posterior_venture.png)

![Venture qqplot](media/qqplot_venture.png)

<a name="biotech"></a>
## Biotech

![Biotech distribution](media/distribution_biotech_200bins_7years.png)

![Biotech trace](media/pymc3_trace_biotech.png)

![Biotech posterior](media/pymc3_posterior_biotech.png)

![Biotech qqplot](media/qqplot_biotech.png)

<a name="comparison"></a>
## Comparison

![Comparison](media/distribution_comparison.png)

|     Index     | N | N > 8 | N doubled |Median | Mean | Std  | C |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |
| Dow           |  30 |  29 | 2  | 0.46/0.30  | 0.72/0.96  | 1.20/2.14   |  1.22 | 
| FTSE100       |  58 |  31 | 0  | 0.12/0.15  | 0.32/0.23  | 3.64/0.33   |  2.56 |
| FTSE250       | 137 |  42 | 0  | 0.09/0.15  | 0.39/02.0  | 7.54/0.23   |  4.09 |
| IBOVESPA      | -   | -   | -  | -          |  -         | -           |  -    |  
| NASDAQ        |4718 |1825 |44  | 0.07/0.14  | 3.58/0.37  | 338.83/1.08 | 48.93 |
| NIFTY50       |  29 |  28 | 8  | 1.15/1.05  | 2.80/3.58  |  2.90/8.69  |  2.21 |
| NIFTYBANK     |  -  |  -  | -  | -          | -          | -           |  -    |
| S&P500        | 500 | 474 | 33 | 0.46/0.46  | 0.79/0.86  | 1.46/1.76   |  1.40 |
| Venture       | 152 | 134 | 18 | 0.57/0.52  | 1.18/1.36  | 2.13/2.95   |  1.80 |
| Biotech       | 365 |  99 |  4 | 0.07/0.09  | 1.18/0.38  | 60.84/0.79  | 15.80 |
