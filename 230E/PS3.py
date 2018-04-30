import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

df = pd.read_excel("PredictorData2015_GoyalWelch.xlsx")
index = df.index[df["yyyymm"]==192601][0]
sample_df = df.iloc[index:,:].copy().reset_index(drop=True)

# part 1,2 & 3
sample_df["dp_t"] = np.log(sample_df["D12"] / sample_df["Index"])
sample_df["ep_t"] = np.log(sample_df["E12"] / sample_df["Index"])

# part 4
sample_df["CRSP_SPvw"] = sample_df["CRSP_SPvw"].apply(lambda x: np.log(1+x)*12)
sample_df["CRSP_SPvwx"] = sample_df["CRSP_SPvwx"].apply(lambda x: np.log(1+x)*12)
print ("CRSP_SPvw has mean %.6f and variance %.6f" % (sample_df["CRSP_SPvw"].mean(), sample_df["CRSP_SPvw"].std()**2))
print ("CRSP_SPvwx has mean %.6f and variance %.6f" % (sample_df["CRSP_SPvwx"].mean(), sample_df["CRSP_SPvwx"].std()**2))
plt.plot(sample_df.index, sample_df["CRSP_SPvw"], 'r-', sample_df.index, sample_df["CRSP_SPvwx"], 'g-')
plt.show()

# part 5
dp_t = sample_df["dp_t"].shift()
ols_df = pd.concat([sample_df["dp_t"], dp_t], axis=1)
ols_df.columns = ['dp_t', 'dp_tm1']
dp_rlt = sm.ols(formula = "dp_t ~ dp_tm1", data = ols_df.iloc[1:,:]).fit()
print ("\nStatistics for dp_t: ")
print (dp_rlt.summary())
ols_df["ep_tm1"] = sample_df["ep_t"].shift()
ols_df["ep_t"] = sample_df["ep_t"].copy()
ep_rlt = sm.ols(formula = "ep_t ~ ep_tm1", data = ols_df.iloc[1:,:]).fit()
print ("\nStatistics for ep_t: ")
print (ep_rlt.summary())

# part 6
# pred_df = pd.concat([pd.Series(dp_rlt.predict()[1:]), sample_df["CRSP_SPvw"][2:].reset_index(drop=True)], axis=1)
# pred_df.columns = ['dp_pred', 'r_tp1']
# dppred_rlt = sm.ols(formula = "r_tp1 ~ dp_pred", data = pred_df).fit()
# print ("\nStatistics for r_tp1 against dp_pred: ")
# print (dppred_rlt.summary())
# pred_df["ep_pred"] = pd.Series(ep_rlt.predict()[1:])
# eppred_rlt = sm.ols(formula = "r_tp1 ~ ep_pred", data = pred_df).fit()
# print ("\nStatistics for r_tp1 against ep_pred: ")
# print (eppred_rlt.summary())

ols_df["r_tp1"] = sample_df["CRSP_SPvw"][1:].reset_index(drop=True)
rdp_rlt = sm.ols(formula = "r_tp1 ~ dp_t", data = ols_df).fit()
print ("\nStatistics for r_tp1 against dp_t: ")
print (rdp_rlt.summary())
rep_rlt = sm.ols(formula = "r_tp1 ~ ep_t", data = ols_df).fit()
print ("\nStatistics for r_tp1 against ep_t: ")
print (rep_rlt.summary())

