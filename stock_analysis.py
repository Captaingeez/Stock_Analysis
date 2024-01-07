import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from datetime import timedelta

plt.style.use('seaborn')

def get_data(num_of_comp:int,end=date.today(),num_of_years=10):
    '''request for company tickers and return historical price data from yahoo finance'''
    securities=[]
    df=pd.DataFrame()
    for x in range(num_of_comp):
        ticker=input('Kindy input company ticker and press enter \n')
        securities.append(ticker.upper())
        num_of_comp=num_of_comp-1
    for security in securities:
        df[security]=yf.Ticker(security).history(start=end-timedelta(days=num_of_years*365),end=end)['Close']
    df.index=pd.to_datetime(df.index, format='%Y%m%d').to_period('D')
    return df

def ret_cov(series):
    '''return the covariance matrix of series return''' 
    return daily_returns(series).cov()
        
def daily_returns(Series):
    '''return the percentage change or return in stock prices daily'''
    returns=pd.DataFrame(Series.pct_change())
    returns=returns.dropna()
    return returns

def annualize_return(returns,period=12):
    annual=np.prod(1+returns)**(period/returns.shape[0])-1
    return annual
    

def mean_std(Series,annual=True):
    ''' Calculate and return the mean and standard deviation of the series.
    To calculate annualised return we set the parameter annual equals True'''
    returns=pd.DataFrame(Series.pct_change())
    returns=returns.dropna()
    if annual==True:
        Avg=((1+returns.mean())**(252))-1
        Dev=returns.std()*(252**0.5)
        df=pd.DataFrame({'Annual Mean':Avg,"Annual Standard Deviation":Dev})
    else:
        Avg=returns.mean()
        Dev=returns.std()
        df=pd.DataFrame({'Mean':Avg,"Standard Deviation":Dev})
    return df
          
def drawdown(series):
    returns=series.pct_change()
    returns=returns.dropna()
    Wealth=1000*(1+returns).cumprod()
    PreviousPeak=Wealth.cummax()
    Drawdown=(Wealth-PreviousPeak)/PreviousPeak
    Max_drawdown=f'Maximum Drawdown of {np.round(Drawdown.min()*100,2)}% occured on {Drawdown.idxmin()}'
    return pd.DataFrame({'Wealth':Wealth,
                                    'Peak':PreviousPeak,
                                    'Drawdown':Drawdown}).head(5),Max_drawdown

def Max_drawdown(series):
    returns=series.pct_change()
    returns=returns.dropna()
    Wealth=1000*(1+returns).cumprod()
    PreviousPeak=Wealth.cummax()
    Drawdown=(Wealth-PreviousPeak)/PreviousPeak
    MaximumDrawdown= Drawdown.min()
    return pd.DataFrame({'Maximum Drawdown':MaximumDrawdown,'Date':Drawdown.idxmin()})

def skewness(series):
    '''return skewness of the return'''
    returns=series.pct_change()
    returns=returns.dropna()
    mean_return=returns.mean()
    std_return=returns.std()
    skew=((returns-mean_return)**3).mean()/std_return**3
    return pd.DataFrame(skew.sort_values)

def skew_ret(returns):
    '''return skewness of the return'''
    mean_return=returns.mean()
    std_return=returns.std()
    skew=((returns-mean_return)**3).mean()/std_return**3
    return skew


def kurtosis(series):
    '''return Kurtosis of the return'''
    returns=series.pct_change()
    returns=returns.dropna()
    mean_return=returns.mean()
    std_return=returns.std()
    kurt=((returns-mean_return)**4).mean()/std_return**4
    return kurt

def kurt_ret(returns):
    '''return Kurtosis of the return'''
    mean_return=returns.mean()
    std_return=returns.std()
    kurt=((returns-mean_return)**4).mean()/std_return**4
    return kurt

from scipy import stats
def isnormal(series,level=0.01):
    returns=pd.DataFrame(series.pct_change())
    returns=returns.dropna()
    if isinstance(returns,pd.DataFrame):
        df=returns.aggregate(stats.jarque_bera)
    elif isinstance(returns,pd.Series):
        return (returns,isnormal)
    else:
        raise "Type Error: isnormal function expects pandas DataFrame or Series"
    return df.iloc[1,]>level

def semideviation_mean(series,mean=True):
    '''Given the price of stock
    it Compute the deviation of returns less than mean return or zero 
    '''
    returns=series.pct_change()
    if mean==True:
        dev=returns[returns<returns.mean()].std()
    else:
        dev=returns[returns<0].std()
    return dev

def sd_mean_ret(returns,mean=True):
    '''Given the return of stock
    it Compute the deviation of returns less than mean return or zero 
    '''
    if mean==True:
        dev=returns[returns<returns.mean()].std()
    else:
        dev=returns[returns<0].std()
    return dev

def historic_var(series,level=5):
    '''Compute historic return variance of stock'''
    returns=daily_returns(series)
    if isinstance(series,pd.Series):
        return -np.percentile(returns,level)
    elif isinstance(series,pd.DataFrame):
        return series.aggregate(historic_var)
    else:
        raise TypeError("Expected Data to be Series or DataFrame")
        
def hvar(returns,level=5):
    '''Compute historic variance of return'''
    if isinstance(returns,pd.Series):
        return -np.percentile(returns,level)
    elif isinstance(returns,pd.DataFrame):
        return returns.aggregate(hvar)
    else:
        raise TypeError("Expected Data to be Series or DataFrame") 
        
def gaussian_var(series, level=5):
    returns=series.pct_change().dropna()
    Z=stats.norm.ppf(level/100)
    return -(returns.mean()+Z*returns.std())

def gvar(returns, level=5):
    Z=stats.norm.ppf(level/100)
    return -(returns.mean()+Z*returns.std())

def Cornish_var(series, level=5):
    returns=series.pct_change().dropna()
    s=skewness(series)
    k=kurtosis(series)
    P=stats.norm.ppf(level/100)
    Z=P+(1/6)*(P**2-1)*s+(1/24)*(P**3-3*P)*k-(1/36)*(2*(P**3)-5*P)*(s**2)
    return -(returns.mean()+Z*returns.std())

def Cor_var(returns, level=5):
    
    s=skew_ret(returns)
    k=kurt_ret(returns)
    P=stats.norm.ppf(level/100)
    Z=P+(1/6)*(P**2-1)*s+(1/24)*(P**3-3*P)*k-(1/36)*(2*(P**3)-5*P)*(s**2)
    return -(returns.mean()+Z*returns.std())

def summary(returns,level=5,period=12):
    summ=pd.DataFrame()
    summ['Annualized Return']=annualize_return(returns,period=period)
    summ['Skewness']=skew_ret(returns)
    summ['Kurtosis']=kurt_ret(returns)
    summ[f'historical {int(100-level)}%VAR']=hvar(returns,level=level)
    summ[f'Gaussian {int(100-level)}%VAR']=gvar(returns,level=level)
    summ[f'Cornish {int(100-level)}%VAR']=Cor_var(returns,level=level)
    return summ
  
def cvar(series,level=5,method='historic'):
    returns=daily_returns(series)
    if method=='historic':
        if isinstance(series,pd.DataFrame):
            return series.aggregate(cvar,level=level,method=method)
        elif isinstance(series,pd.Series):
            is_beyond=returns<=-historic_var(series,level=level)
            return returns[is_beyond].mean()
        else:
            raise TypeError("Expected Data to be Series or DataFrame")

    elif method=='gaussian':
        if isinstance(series,pd.DataFrame):
            return series.aggregate(cvar,level=level,method=method)
        elif isinstance(series,pd.Series):
            is_beyond=returns<=-gaussian_var(series,level=level)
            return returns[is_beyond].mean()
        else:
            raise TypeError("Expected Data to be Series or DataFrame")

    elif method=='cornish':
        if isinstance(series,pd.DataFrame):
            return series.aggregate(cvar,level=level,method=method)
        elif isinstance(series,pd.Series):
            is_beyond=returns<=-Cornish_var(series,level=level)
            return returns[is_beyond].mean()
        else:
            raise TypeError("Expected Data to be Series or DataFrame")
    else: 
        raise TypeError("Expected 'historic','gaussian' or 'cornish' ")          

        
def portfolio_return(weights,series):
    '''calculate portfolio return'''
    mean=mean_std(series).iloc[:,0]
    return weights.T @ mean

def portfolio_variance(weight,series):
    '''calculate portfolio variance'''
    cov_mat=daily_returns(series).cov()
    return weight.T @ cov_mat @ weight

def portfolio_return_gmv(weights,series):
    '''GMV portfolio return'''
    n=series.shape[1]
    mean=np.repeat(1,n)
    return weights.T @ mean

import numpy as np
def plot_e2(n_points,series):
    '''plot the 2 asset efficient frontiers'''
    if series.shape[1]!=2:
        raise ValueError('can only plot 2-Asset')
    weights=[np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets=[portfolio_return(w,series) for w in weights]
    vols=[portfolio_variance(w,series) for w in weights]
    port=pd.DataFrame({'returns':rets,'variance':vols})
    return port.plot.scatter(x='variance',y='returns',style='.-')

from scipy.optimize import minimize

def minimize_vol(target_return,series):
    n=series.shape[1]
    init_guess=np.repeat(1/n,n)
    bounds=((0,1),)*n
    mean=mean_std(series).iloc[:,0]
    cov=daily_returns(series).cov()
    return_is_target={'type':'eq',
                      'args':(series,),
                     'fun':lambda weight,series:target_return-portfolio_return(weight,series)}
    weight_sum_to_1={'type':'eq', 
                     'fun':lambda weight:np.sum(weight)-1}
    result=minimize(portfolio_variance,init_guess,args=(series,),
                    method='SLSQP',
                    options={'disp':False},
                   constraints=(return_is_target,weight_sum_to_1),
                   bounds=bounds)
    return result['x']

def plot_ef(n_points,series,riskfree_rate,show_cml=True,show_eq=True,show_gmv=True):
    '''plot the 2 asset efficient frontiers'''
    mean=mean_std(series).iloc[:,0]
    target_rs=np.linspace(mean.min(),mean.max(),n_points)
    weights=[minimize_vol(target_return,series) for target_return in target_rs]
    rets=[portfolio_return(w,series) for w in weights]
    vols=[portfolio_variance(w,series) for w in weights]
    port=pd.DataFrame({'returns':rets,'variance':vols})
    ax=port.plot.scatter(x='variance',y='returns',style='.-')
    
    if show_cml:
        w_msr=msr(series,riskfree_rate)
        ret=portfolio_return(w_msr,series)
        vol=portfolio_variance(w_msr,series)
        cml_x=[0,vol]
        cml_y=[riskfree_rate,ret]
        ax.plot(cml_x,cml_y,color='green',marker='o',linestyle='dashed')
       
    if show_eq:
        n=series.shape[1]
        w_eq=np.repeat(1/n,n)
        ret=portfolio_return(w_eq,series)
        vol=portfolio_variance(w_eq,series)
        cml_x=[0,vol]
        cml_y=[riskfree_rate,ret]
        ax.plot(cml_x,cml_y,color='gold',marker='o',linestyle='dashed')
        
    if show_gmv:
       
        w_gmv=gmv(series)
        ret=portfolio_return(w_gmv,series)
        vol=portfolio_variance(w_gmv,series)
        cml_x=[0,vol]
        cml_y=[riskfree_rate,ret]
        ax.plot(cml_x,cml_y,color='blue',marker='o',linestyle='dashed')
  
    return ax


def msr(series,riskfree_rate):
    n=series.shape[1]
    init_guess=np.repeat(1/n,n)
    bounds=((0,1),)*n
    mean=mean_std(series).iloc[:,0]
    cov=daily_returns(series).cov()
    
    weight_sum_to_1={'type':'eq', 
                     'fun':lambda weight:np.sum(weight)-1}
    def neg_sr(weight,series,riskfree_rate):
        sr=(portfolio_return(weight,series)-riskfree_rate)/portfolio_variance(weight,series)
        return -sr
    result=minimize(neg_sr,init_guess,args=(series,riskfree_rate),
                    method='SLSQP',
                    options={'disp':False},
                   constraints=(weight_sum_to_1),
                   bounds=bounds)
    return result['x']

def gmv(series):
    n=series.shape[1]
    init_guess=np.repeat(1/n,n)
    bounds=((0,1),)*n
    weight_sum_to_1={'type':'eq', 
                     'fun':lambda weight:np.sum(weight)-1}
    def neg_sr(weight,series):
        sr=(portfolio_return_gmv(weight,series)-0)/portfolio_variance(weight,series)
        return -sr
    result=minimize(neg_sr,init_guess,args=(series),
                    method='SLSQP',
                    options={'disp':False},
                   constraints=(weight_sum_to_1),
                   bounds=bounds)
    return result['x']

def optimal_weight(n_points,series):
    mean=mean_std(series).iloc[:,0]
    target_rs=np.linspace(mean.min(),mean.max(),n_points)
    weights=np.round([minimize_vol(target_return,series) for target_return in target_rs],3)
    return pd.DataFrame(weights,columns=series.columns,index=target_rs*100)

def CPPI(series,m,floor,riskfree_rate,start,drawdown=None):
    risky_r=daily_returns(series)
    safe_asset=pd.DataFrame().reindex_like(risky_r)
    safe_asset[:]=riskfree_rate/252
    n_step=safe_asset.shape[0]
    floor_value=floor*start
    account_value=start
    peak=start
    Cushion_history=pd.DataFrame().reindex_like(safe_asset)
    Account_history=pd.DataFrame().reindex_like(safe_asset)
    RiskyW_history=pd.DataFrame().reindex_like(safe_asset)
    Floor_history=pd.DataFrame().reindex_like(safe_asset)
    for step in range(n_step):
        if drawdown is not None:
            peak=np.maximum(peak,account_value)
            floor_value=peak*(1-drawdown)
        cushion=((account_value-floor_value)/account_value)
        risky_weight=cushion*m
        risky_weight=np.minimum(risky_weight,1)
        risky_weight=np.maximum(0,risky_weight)
        safety_weight=1-risky_weight
        risky_allocation=risky_weight*account_value
        safety_allocation=safety_weight*account_value
        account_value=risky_allocation*(1+risky_r.iloc[step])+safety_allocation*(1+safe_asset.iloc[step])
        Cushion_history.iloc[step]=cushion
        Account_history.iloc[step]=account_value
        RiskyW_history.iloc[step]=risky_allocation
        Floor_history.iloc[step]=floor_value
        risky_wealth=start*(1+risky_r).cumprod()
    backtest={'Wealth':Account_history,   
              'Risky Wealth':risky_wealth,
              'Risky allocation':RiskyW_history,
              'Risk Budget':Cushion_history,
             'Floor':Floor_history}
    return backtest 

def gbm(mu,std,no_of_years=10,n_scenario=10,days_annual=252):
    s0=100
    steps=(no_of_years*days_annual)+1
    dt=1/days_annual
    rets_plus_1=np.random.normal(loc=1+mu*dt,scale=std*np.sqrt(dt),size=(steps,n_scenario))
    rets_plus_1=pd.DataFrame(rets_plus_1)
    rets_plus_1.iloc[0]=1
    
    return s0*(rets_plus_1).cumprod()


def show_gbm(mu,std,n_scenario):
    s0=100
    price=gbm(mu=mu,std=std,no_of_years=no_of_years,n_scenario=n_scenario)
    ax=price.plot(legend=False,alpha=0.5, color='indianred',linewidth=2,figsize=(12,6))
    ax.axhline(y=100, ls=':',color='black')
    ax.plot(0,s0,ls='--',color='darkred')
def show_cppi(mu=0.08,std=0.12,n_scenario=50,m=5,floor=0.5,riskfree_rate=0.03,start=100,y_max=100):
    s0=100
    price=gbm(mu=mu,std=std,no_of_years=no_of_years,n_scenario=n_scenario)
    Insurance=CPPI(price,m=m,floor=floor,riskfree_rate=riskfree_rate,start=start,drawdown=None)
    wealth=Insurance['Wealth']
    y_max=wealth.values.max()*y_max/100
    ax=wealth.plot(legend=False,alpha=0.5, color='indianred',linewidth=2,figsize=(12,6))
    ax.axhline(y=start, ls=':',color='black')
    ax.axhline(y=start*floor, ls='--',color='indianred')
    ax.set_ylim(top=y_max)