import pandas as pd
import numpy as np
np.random.seed(99)
import pickle

import seaborn as sns
from seaborn_qqplot import pplot
from scipy.stats import gamma,norm,exponweib,lognorm
from scipy.stats import gengamma,invgauss,loglaplace,powerlognorm,t

from scipy.stats import gengamma,invgauss,loglaplace,powerlognorm,t
from scipy.stats import truncnorm,truncexpon,expon,loggamma,halfgennorm,exponpow
from scipy.stats import exponnorm,dgamma,betaprime
from scipy.stats import laplace,gennorm,laplace_asymmetric,skewnorm,genhyperbolic

        
import matplotlib.pyplot as plt

import statsmodels.api as sm
np.random.seed(99)


dir_in='./data/'

fname='all_indexes_2006-01-01_2021-12-31.pickle'   
with open(dir_in+fname,'rb') as file:
     d=pickle.load(file)


import os
filter_remove_delisted=False    

if filter_remove_delisted:
 dir_path = './fig/qqplot_logmu_filtered/'
else:
 dir_path = './fig/qqplot_logmu/'    

os.makedirs(dir_path, exist_ok=True)

for k,index_name in enumerate(d.keys()):
    dz=pd.DataFrame(columns=['ticker','mu'],index=np.arange(len(d[index_name].keys())))        
    dz_nc=pd.DataFrame(columns=['ticker','mu_nocut'],index=np.arange(len(d[index_name].keys()))) # no log(mu)=-2 cutoff fitted with gen_hyperbolic
    for i, ticker in enumerate(d[index_name].keys()):
        df=d[index_name][ticker]
        # add filters to remove empty dataframes or with nans
        if (len(df.columns)>0)&(df.isnull().sum().sum()==0):
            n_years=len(df) # 16 years  
            n_different_close=sum(abs(df['Close'].diff())>0)

            if filter_remove_delisted:
                cond=(n_years==(n_different_close+1))
            else:
                cond=True            
            if cond:
                try:
                    mu=df['Close'].iloc[-1]/df['Open'].iloc[0]
                    dz_nc.at[i,'ticker']=ticker            
                    dz_nc.at[i,'mu_nocut']=mu

                    if np.log(mu)>-2.:
                        dz.at[i,'ticker']=ticker            
                        dz.at[i,'mu']=mu
                except:
                    print(ticker)


    dz_nc=dz_nc.dropna()
    dz_nc.mu_nocut=dz_nc.mu_nocut.astype(float)     
    dz_nc['log_mu_nocut']=np.log(dz_nc['mu_nocut'])
    
    
    dz=dz.dropna()
    dz.mu=dz.mu.astype(float)     
    dz['log_mu']=np.log(dz['mu'])
    dz['log_mu_w']=np.maximum(dz['log_mu'],-2.) # winsorize left tail; trivial with if np.log(mu)>-2. above     

#sns.distplot(dz['mu'],bins=100)
#dp=dz.copy(deep=True)

    try:
#        ax1=pplot(dz[:], x="mu", y=powerlognorm, kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True,"reg":True, "ci":0.025})
        fig = sm.qqplot(dz["mu"], powerlognorm, fit=True, line="45")
        title_text='QQ plot for '+index_name+' fit with '+'powerlognorm'
        plt.title(title_text)
        plt.xlabel(r'Empirical distribution of $\rho$')
        plt.ylabel('Theoretical powerlognorm distribution')
        plt.grid()
        plt.savefig(dir_path+title_text+'.png',bbox_inches='tight')
        
#        ax2=pplot(dz[:], x="log_mu_w", y=norm, kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True,"reg":True, "ci":0.025})
        fig = sm.qqplot(dz["log_mu_w"], norm, fit=True, line="45")
        title_text='QQ plot for '+index_name+' fit with '+'lognorm'
        plt.title(title_text)
        plt.xlabel(r'Empirical distribution of $\ln(\rho)$')
        plt.ylabel('Theoretical lognormal distribution')
        plt.grid()
        plt.savefig(dir_path+title_text+'.png',bbox_inches='tight')
        
#        ax3=pplot(dz[1:], x="mu", y=gamma, kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True,"reg":True, "ci":0.025})
#        title_text='QQ plot for '+index_name+' fit with '+'gamma'
#        plt.title(title_text)
#        plt.savefig('./fig/qqplot/'+title_text+'.png',bbox_inches='tight')

#        ax4=pplot(dz[:], x="log_mu_w", y=laplace, kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True,"reg":True, "ci":0.025})
        fig = sm.qqplot(dz["log_mu_w"], laplace, fit=True, line="45")
        title_text='QQ plot for '+index_name+' fit with '+'loglaplace'
        plt.title(title_text)
        plt.xlabel(r'Empirical distribution of $\ln(\rho)$')
        plt.ylabel('Theoretical loglaplace distribution')
        plt.grid()
        plt.savefig(dir_path+title_text+'.png',bbox_inches='tight')
        
#        ax5=pplot(dz[:], x="log_mu_w", y=laplace_asymmetric, kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True,"reg":True, "ci":0.025})
        fig = sm.qqplot(dz["log_mu_w"], laplace_asymmetric, fit=True, line="45")
        title_text='QQ plot for '+index_name+' log mu fit with '+'laplace_asymmetric'
        plt.xlabel(r'Empirical distribution of $\ln(\rho)$')
        plt.ylabel('Theoretical loglaplace distribution')        
        plt.title(title_text)
        plt.grid()
        plt.savefig(dir_path+title_text+'.png',bbox_inches='tight')
  
#        ax6=pplot(dz[:], x="log_mu_w", y=skewnorm, kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True,"reg":True, "ci":0.025})
        fig = sm.qqplot(dz["log_mu_w"], skewnorm, fit=True, line="45")
        title_text='QQ plot for '+index_name+' log mu fit with '+'skewnorm'
        plt.xlabel(r'Empirical distribution of $\ln(\rho)$')
        plt.ylabel('Theoretical skewnorm distribution')        
        plt.title(title_text)
        plt.grid()
        plt.savefig(dir_path+title_text+'.png',bbox_inches='tight')
        
#        ax7=pplot(dz_nc[:], x="log_mu_nocut", y=genhyperbolic, kind='qq', height=4, aspect=2, display_kws={"identity":False, "fit":True,"reg":True, "ci":0.025})
        fig = sm.qqplot(dz["log_mu_w"], genhyperbolic, fit=True, line="45")
        title_text='QQ plot for '+index_name+' log mu fit with '+'gen_hyperbolic'
        plt.xlabel(r'Empirical distribution of $\rho$')
        plt.ylabel('Theoretical genhyperbolic distribution')        
        plt.title(title_text)
        plt.grid()
        plt.savefig(dir_path+title_text+'.png',bbox_inches='tight')       
        
        
    except:
            print('Cannot chart ' +index_name)   