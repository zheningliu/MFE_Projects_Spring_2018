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
#ret_df["DY"] = df.iloc[1:,4].divide(df.iloc[1:,3])
ret_df.iloc[:,[0,2,4]] = ret_df.iloc[:,[0,2,4]].apply(lambda x: x/100)
ret_df["DY_s"] = ret_df["DY"].rolling(12).mean()
index = [i for i in range(len(ret_df.index))]
# plt.plot(index, ret_df["DY"],'b-', index, ret_df["DY_s"], 'r-')
# plt.title("Dividend Yield")
# plt.show()

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


# # part 1
# price_df = pd.read_csv("price_data.csv", index_col = 0, na_values = ["ND"])
# gas_df = pd.read_csv("oil_gas_data.csv", index_col = 0)
# print (gas_df.head())
# merge_df = price_df.merge(gas_df, how='left', left_index=True, right_index=True)
# print (merge_df.head())
# merge_df = price_df.iloc[:,[0,2,3]]#.pct_change()
# merge_df.columns = ["SP500","EXRATE","OGI"]
# print (merge_df.head())