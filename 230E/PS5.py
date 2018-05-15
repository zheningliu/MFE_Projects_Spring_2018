"""
MFE 230-E Assignment 5
Author: Nathan Johnson / Zhening Liu
Date: May 15, 2018
"""

plt.rcParams.update({'font.size' : 24})
from scipy import optimize
import Assignment_5_Helper as helper
import numpy as np
import pandas as pd
from stats_pkg import *
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# part 1-3
df = pd.read_csv("reit_data_2017.csv", skiprows = 7, index_col = 0, thousands=',')
ind1 = df.index.get_loc("Jan-93")
ind2 = df.index.get_loc("Dec-14")
ret_df = df.iloc[ind1:ind2,:5]
ret_df.columns = ["TR", "TI", "PR", "PI", "DY"]
ret_df.iloc[:,[0,2,4]] = ret_df.iloc[:,[0,2,4]].apply(lambda x: x/100)
ret_df["DY_s"] = ret_df["DY"].rolling(12).mean()
index = [i for i in range(len(ret_df.index))]
plt.plot(index, ret_df["DY"],'b-', index, ret_df["DY_s"], 'r-')
plt.title("Dividend Yield")
plt.show()

# part 4-5
ret_df[["log_R", "log_DY", "log_DYs"]] = ret_df.iloc[:,[0,2,4]].apply(lambda x: np.log(1+x))
ann_df = pd.concat([ret_df[["PR", "DY", "DY_s"]].apply(lambda x: np.power(1+x,12)-1), 
	ret_df[["log_R", "log_DY", "log_DYs"]].apply(lambda x: x*12)], axis=1)
print ("Mean:\n%s" % ann_df.mean())
print ("Standard Deviation:\n%s" % ann_df.std())
print ("Autocorrelation:\n%s" % ann_df.apply(lambda col: pd.Series(col).autocorr(), axis=0))
print ("Skewness:\n%s" % ann_df.skew())
print ("Kurtosis:\n%s" % ann_df.kurtosis())

# part 6
for i in range(1, 13):
	ann_df["R_tm%s" % i] = ann_df["PR"].shift(i)
result_a = sm.ols(formula = "PR ~ R_tm1", data = ann_df.iloc[1:,:]).fit()
print ("OLS result a:\n%s" % result_a.summary())
result_b = sm.ols(formula = "PR ~ R_tm1 + R_tm2", data = ann_df.iloc[2:,:]).fit()
print ("OLS result b:\n%s" % result_b.summary())
tm_str = " + ".join(["R_tm%s" % i for i in range(1, 13)])
result_c = sm.ols(formula = "PR ~ %s" % tm_str, data = ann_df.iloc[12:,:]).fit()
print ("OLS result c:\n%s" % result_c.summary())

# part 7
for i in range(1, 13):
	ann_df["logR_tm%s" % i] = ann_df["log_R"].shift(i)
result_a = sm.ols(formula = "log_R ~ logR_tm1", data = ann_df.iloc[1:,:]).fit()
print ("OLS log result a:\n%s" % result_a.summary())
result_b = sm.ols(formula = "log_R ~ logR_tm1 + logR_tm2", data = ann_df.iloc[2:,:]).fit()
print ("OLS log result b:\n%s" % result_b.summary())
tm_str = " + ".join(["logR_tm%s" % i for i in range(1, 13)])
result_c = sm.ols(formula = "log_R ~ %s" % tm_str, data = ann_df.iloc[12:,:]).fit()
print ("OLS log result c:\n%s" % result_c.summary())

# part 9
for param in ["DY", "DY_s", "log_DY", "log_DYs"]:
	ann_df["%s_tm1" % param] = ann_df[param].shift(1)
	if param in ["DY", "DY_s"]:
		result = sm.ols(formula = "PR ~ %s_tm1" % param, data = ann_df.iloc[1:,:]).fit()
		print ("OLS Rt vs. %s:\n%s" % (param, result.summary()))
	else:
		result = sm.ols(formula = "log_R ~ %s_tm1" % param, data = ann_df.iloc[1:,:]).fit()
		print ("OLS logR vs. %s:\n%s" % (param, result.summary()))

# part 11
result_AR = sm.ols(formula = "DY ~ DY_tm1", data = ann_df.iloc[1:,:]).fit()
print ("OLS DY vs. DY_tm1:\n%s" % result_AR.summary())

# part 12
result_Pred = sm.ols(formula = "PR ~ DY_tm1", data = ann_df.iloc[1:,:]).fit()
DY_tm1 = ann_df["DY"][-1]
for i in range(3):
	PR = result_Pred.params[0]+result_Pred.params[1]*DY_tm1
	DY = result_AR.params[0]+result_AR.params[1]*DY_tm1
	print ("Return: %s" % PR)
	DY_tm1 = DY


#PART 2
#####################

#Problem 1
#-------------------------------
sp_data = pd.read_csv('vol_data_homework.csv', usecols=np.array([0,1]),
                      index_col=0, delimiter=',')
exch_data = pd.read_csv('vol_data_homework.csv', usecols=np.array([2,3]),
                        index_col=0, delimiter=',')
oil_data = pd.read_csv('vol_data_homework.csv', usecols=np.array([4,5]),
                       index_col=0, delimiter=',')
oil_data = oil_data[oil_data.index.notnull()]
data = sp_data.join(exch_data)
data = data.join(oil_data)
data[data == 'ND'] = np.nan
final_df = data.astype(float)
date_idx = pd.to_datetime(data.index)
final_df.index = date_idx
cols = ['S&PCOMP', 'Exch Rate', 'NYSE ARCA']

#Problem 2
#-------------------------------
daily_percent_change = final_df.pct_change(1)
daily_log_returns = np.log(1 + daily_percent_change)
daily_log_returns[np.isinf(daily_log_returns)] = np.nan

summary_statistics_daily = pd.DataFrame(index=['Mean', 'Std', 'Autocorr',
                                               'Skewness', 
                                            'Kurtosis'], columns=cols,
                                    dtype=float)
for asset in cols:
    mean = daily_log_returns[asset].mean()
    std = daily_log_returns[asset].std()
    autocorr = daily_log_returns[asset].autocorr()
    skew = daily_log_returns[asset].skew()
    kurtosis = daily_log_returns[asset].kurtosis()
    
    summary_statistics_daily[asset].loc['Mean'] = mean
    summary_statistics_daily[asset].loc['Std'] = std
    summary_statistics_daily[asset].loc['Autocorr'] = autocorr
    summary_statistics_daily[asset].loc['Skewness'] = skew
    summary_statistics_daily[asset].loc['Kurtosis'] = kurtosis

monthly_prices = final_df.resample('BM').apply(lambda x:x[-1])
monthly_percent_change = monthly_prices.pct_change(1)
monthly_log_returns = np.log(1 + monthly_percent_change)
monthly_log_returns[np.isinf(monthly_log_returns)] = np.nan

summary_statistics_monthly = pd.DataFrame(index=['Mean', 'Std', 'Autocorr',
                                                 'Skewness',
                                                 'Kurtosis'], columns=cols,
                                            dtype=float)
for asset in cols:
    mean = monthly_log_returns[asset].mean()
    std = monthly_log_returns[asset].std()
    autocorr = monthly_log_returns[asset].autocorr()
    skew = monthly_log_returns[asset].skew()
    kurtosis = monthly_log_returns[asset].kurtosis()
    
    summary_statistics_monthly[asset].loc['Mean'] = mean
    summary_statistics_monthly[asset].loc['Std'] = std
    summary_statistics_monthly[asset].loc['Autocorr'] = autocorr
    summary_statistics_monthly[asset].loc['Skewness'] = skew
    summary_statistics_monthly[asset].loc['Kurtosis'] = kurtosis

#Problem 3
#-------------------------------
years = daily_log_returns.index.year
months = daily_log_returns.index.month
monthly_std = np.sqrt(252) * daily_log_returns.groupby([years, months]).std()
y = monthly_std.index.get_level_values(0)
m = monthly_std.index.get_level_values(1)
date = pd.to_datetime(y * 10000 + m * 100 + 1, format='%Y%m%d')
values = monthly_std[:].values
monthly_std = pd.DataFrame(index=date, columns=cols, data = values)

#Problem 4
#-------------------------------
#PART A
returns = monthly_log_returns
arch1_vol_df = pd.DataFrame(index=returns.index, columns=cols, dtype=float)
arch1_param_df = pd.DataFrame(index=['c', 'phi', 'xi', 'a1'], columns=cols,
                              dtype=float)
arch1_llf_df = pd.DataFrame(index=['LLF', 'AIC'], columns=cols, dtype=float)

for asset in cols:
    r = returns[asset]
    r = r[~np.isnan(r)]
    idx = r.index
    r = np.array(r)
    r_lag = r[0:-1]
    r = r[1:]
    
    theta_init = np.array([0.001, 0.001, 0.001, 0.001])
    bnds = ((1e-20, 1), (1e-20, 1), (1e-20, 1), (1e-20, 1))
    llf_archm = helper.llk_archm_fn(r, r_lag, 1)
    
    optimal = optimize.minimize(llf_archm, theta_init, bounds=bnds,
                                method='L-BFGS-B')
    theta_hat = optimal.x
    arch1_param_df[asset] = theta_hat
    arch1_llf_df[asset].loc['LLF'] = -llf_archm(theta_hat)
    arch1_llf_df[asset].loc['AIC'] = 2 * 4 + llf_archm(theta_hat)
    
    h_t_m = helper.archm_vol(r, r_lag, 1, theta_hat)
    arch1_vol_df[asset].loc[idx[2:]] = np.sqrt(12 * h_t_m)
arch1_vol_df.astype(float)

#PART B
arch3_vol_df = pd.DataFrame(index=returns.index, columns=cols, dtype=float)
arch3_param_df = pd.DataFrame(index=['c', 'phi', 'xi', 'a1', 'a2', 'a3'],
                              columns=cols, dtype=float)
arch3_llf_df = pd.DataFrame(index=['LLF', 'AIC'], columns=cols, dtype=float)

for asset in cols:
    r = returns[asset]
    r = r[~np.isnan(r)]
    idx = r.index
    r = np.array(r)
    r_lag = r[0:-1]
    r = r[1:]
    
    theta_init = 0.001 * np.ones([3 + 3,])
    bnds = ((1e-20, 1), (1e-20, 1), (1e-20, 1), (1e-20, 1),
            (1e-20, 1), (1e-20, 1))
    llf_archm = helper.llk_archm_fn(r, r_lag, 3)
    
    optimal = optimize.minimize(llf_archm, theta_init, bounds=bnds,
                                method='L-BFGS-B')
    theta_hat = optimal.x
    arch3_param_df[asset] = theta_hat
    arch3_llf_df[asset].loc['LLF'] = -llf_archm(theta_hat)
    arch3_llf_df[asset].loc['AIC'] = 2 * 6 + llf_archm(theta_hat)
    
    h_t_m = helper.archm_vol(r, r_lag, 3, theta_hat)
    arch3_vol_df[asset].loc[idx[4:]] = np.sqrt(12 * h_t_m)
arch3_vol_df.astype(float)

#PART C
arch12_vol_df = pd.DataFrame(index=returns.index, columns=cols, dtype=float)
arch12_param_df = pd.DataFrame(index=['c', 'phi', 'xi', 'a1', 'a2', 
                                      'a3', 'a4', 'a5', 'a6', 'a7',
                                      'a8', 'a9', 'a10', 'a11', 'a12'],
                                columns=cols, dtype=float)
arch12_llf_df = pd.DataFrame(index=['LLF', 'AIC'], columns=cols, dtype=float)

for asset in cols:
    r = returns[asset]
    r = r[~np.isnan(r)]
    idx = r.index
    r = np.array(r)
    r_lag = r[0:-1]
    r = r[1:]
    
    theta_init = 0.001 * np.ones([3 + 12,])
    bnds = ((1e-20, 1), (1e-20, 1), (1e-20, 1), (1e-20, 1), (1e-20, 1),
            (1e-20, 1), (1e-20, 1), (1e-20, 1), (1e-20, 1), (1e-20, 1),
            (1e-20, 1), (1e-20, 1), (1e-20, 1), (1e-20, 1), (1e-20, 1))
    llf_archm = helper.llk_archm_fn(r, r_lag, 12)
    
    optimal = optimize.minimize(llf_archm, theta_init, bounds=bnds,
                                method='L-BFGS-B')
    theta_hat = optimal.x
    arch12_param_df[asset] = theta_hat
    arch12_llf_df[asset].loc['LLF'] = -llf_archm(theta_hat)
    arch12_llf_df[asset].loc['AIC'] = 2 * 15 + llf_archm(theta_hat)
    
    h_t_m = helper.archm_vol(r, r_lag, 12, theta_hat)
    arch12_vol_df[asset].loc[idx[13:]] = np.sqrt(12 * h_t_m)
arch12_vol_df.astype(float)

#PART E
arch_effect_df = pd.DataFrame(index=['Result'], columns=cols)

for asset in cols:
    r = returns[asset]
    r = r[~np.isnan(r)]
    r = np.array(r)
    r_lag = r[0:-1]
    r = r[1:]
    
    arch_effect_df[asset].loc['Result'] = helper.arch_effect_test(r, r_lag)


#Problem 5
#-------------------------------
#PART A
returns = monthly_log_returns
garch11_vol_df = pd.DataFrame(index=returns.index, columns=cols,dtype=float)
garch11_param_df = pd.DataFrame(index = ['c', 'phi', 'xi', 'q', 'p'], 
                                columns=cols, dtype=float)
garch11_llf_df = pd.DataFrame(index=['LLF', 'AIC'], columns=cols,
                              dtype=float)

for asset in cols:
    r = returns[asset]
    r = r[~np.isnan(r)]
    idx = r.index
    r = np.array(r)
    r_lag = r[0:-1]
    r = r[1:]
    
    theta_init = 0.001 * np.ones([3 + 2,])
    bnds = ((1e-10, 1), (1e-10, 1), (1e-10, 1), (1e-10, 1), (1e-10, 1))
    llf_garchpq = helper.llk_garchpq_fn(r, r_lag, 1, 1)
    
    optimal = optimize.minimize(llf_garchpq, theta_init, bounds=bnds,
                                method='L-BFGS-B')
    theta_hat = optimal.x
    garch11_param_df[asset] = theta_hat
    garch11_llf_df[asset].loc['LLF'] = -llf_garchpq(theta_hat)
    garch11_llf_df[asset].loc['AIC'] = 2 * 5 + llf_garchpq(theta_hat)
    
    h_t_m = helper.garchpq_vol(r, r_lag, 1, 1, theta_hat)
    garch11_vol_df[asset].loc[idx[2:]] = np.sqrt(12 * h_t_m)

#PART B
garch22_vol_df = pd.DataFrame(index=returns.index, columns=cols, dtype=float)
garch22_param_df = pd.DataFrame(index=['c', 'phi', 'xi', 'q1', 'q2', 'p1',
                                     'p2'], columns=cols, dtype=float)
garch22_llf_df = pd.DataFrame(index=['LLF', 'AIC'], columns=cols,
                              dtype=float)

for asset in cols:
    r = returns[asset]
    r = r[~np.isnan(r)]
    idx = r.index
    r = np.array(r)
    r_lag = r[0:-1]
    r = r[1:]
    
    theta_init = 0.001 * np.ones([3 + 4,])
    bnds = ((1e-10, 1), (1e-10, 1), (1e-10, 1), (1e-10, 1), (1e-10, 1),
            (1e-10, 1), (1e-10, 1))
    llf_garchpq = helper.llk_garchpq_fn(r, r_lag, 2, 2)
    
    optimal = optimize.minimize(llf_garchpq, theta_init, bounds=bnds,
                                method='L-BFGS-B')
    theta_hat = optimal.x
    garch22_param_df[asset] = theta_hat
    garch22_llf_df[asset].loc['LLF'] = -llf_garchpq(theta_hat)
    garch22_llf_df[asset].loc['AIC'] = 2 * 7 + llf_garchpq(theta_hat)
    
    h_t_m = helper.garchpq_vol(r, r_lag, 2, 2, theta_hat)
    garch22_vol_df[asset].loc[idx[3:]] = np.sqrt(12 * h_t_m)

#PART D
for asset in cols:
    plt.figure()
    garch22_vol_df[asset].plot()
    garch11_vol_df[asset].plot()
    monthly_std[asset].plot()
    plt.legend(['GARCH(2,2) Estimate', 'GARCH(1,1) Estimate', 'Data'])
    plt.xlabel('Time')
    plt.ylabel('Annualized Volatility')
    plt.title(asset)
