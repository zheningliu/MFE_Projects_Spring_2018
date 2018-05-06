import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

df = pd.read_excel("countries_daily.xls", na_values=[0.0])
country_ls = ["US", "UNKING", "JAP", "HONG", "CHINA"]
country_df = df[country_ls].copy()
return_df = country_df.sub(country_df.shift()).divide(country_df.iloc[:-1,:]).iloc[1:,:]

for country in country_ls:
	return_df["daily_%s" % country] = np.log(1 + return_df[country])
	return_df["monthly_%s" % country] = return_df["daily_%s" % country] * 21
daily_df = return_df.iloc[:,[2*k+5 for k in range(5)]].copy()
monthly_df = return_df.iloc[:,[2*k+6 for k in range(5)]].copy()

# part 3a
print ("################## Part 3a ##################\n")
daily_mean = daily_df.mean() * 252
mean_df = pd.DataFrame(pd.Series(daily_mean), columns=['Total'])
print (daily_mean)

# part 3b
print ("\n################## Part 3b ##################\n")
daily_std = daily_df.std() * np.sqrt(252)
std_df = pd.DataFrame(pd.Series(daily_std), columns=['Total'])
print (daily_std)

# part 3c&d&e
print ("\n################## Part 3c&d&e ##################\n")
decade_ls = np.divide(daily_df.index.year.values, 10).astype(int).tolist()
decades = [(decade_ls[0] + i) * 10 for i in range(decade_ls[-1] - decade_ls[0] + 1)]
for i in range(len(decades) - 1):
	fi = decade_ls.index(decades[i] / 10)
	li = decade_ls.index(decades[i+1] / 10)
	mean_df["%s\'s" % decades[i]] = daily_df.iloc[fi:li,:].mean() * 252
	std_df["%s\'s" % decades[i]] = daily_df.iloc[fi:li,:].std() * np.sqrt(252)
mean_df["%s\'s" % decades[-1]] = daily_df.iloc[li:,:].mean() * 252	# last decade
std_df["%s\'s" % decades[-1]] = daily_df.iloc[li:,:].std() * np.sqrt(252)
print ("Decade means:\n %s" % mean_df)
print ("\nDecade std dev:\n %s" % std_df)

# part 3d
year_ls = daily_df.index.year.values.tolist()
years = list(sorted(set(year_ls)))
for i in range(len(years) - 1):
	fi = year_ls.index(years[i])
	li = year_ls.index(years[i+1])
	mean_df["%s" % years[i]] = daily_df.iloc[fi:li,:].mean() * 252
	std_df["%s" % years[i]] = daily_df.iloc[fi:li,:].std() * np.sqrt(252)
mean_df["%s" % years[-1]] = daily_df.iloc[li:,:].mean() * 252	# last decade
std_df["%s" % years[-1]] = daily_df.iloc[li:,:].std() * np.sqrt(252)
# # mean plot
# plt.figure()
# mean_df.iloc[:,(len(mean_df.columns) - len(years)):].transpose().plot()
# plt.title("Yearly Mean of Five Countries")
# plt.legend(loc='best')
# plt.show()
# # std plot
# plt.figure()
# std_df.iloc[:,(len(mean_df.columns) - len(years)):].transpose().plot()
# plt.title("Yearly Std of Five Countries")
# plt.legend(loc='best')
# plt.show()

# part 3e
month_ls = (daily_df.index.year.values * 100 + daily_df.index.month.values).tolist()
months = list(sorted(set(month_ls)))
for i in range(len(months) - 1):
	fi = month_ls.index(months[i])
	li = month_ls.index(months[i+1])
	mean_df["%s" % months[i]] = daily_df.iloc[fi:li,:].mean() * 252
	std_df["%s" % months[i]] = daily_df.iloc[fi:li,:].std() * np.sqrt(252)
mean_df["%s" % months[-1]] = daily_df.iloc[li:,:].mean() * 252	# last decade
std_df["%s" % months[-1]] = daily_df.iloc[li:,:].std() * np.sqrt(252)
# # mean plot
# plt.figure()
# mean_df.iloc[:,(len(mean_df.columns) - len(months)):].transpose().plot()
# plt.title("Monthly Mean of Five Countries")
# plt.legend(loc='best')
# plt.show()
# # std plot
# plt.figure()
# std_df.iloc[:,(len(mean_df.columns) - len(months)):].transpose().plot()
# plt.title("Monthly Std of Five Countries")
# plt.legend(loc='best')
# plt.show()

# part 3f
print ("\n################## Part 3f ##################\n")
rv = std_df.iloc[:,(len(mean_df.columns) - len(months)):].transpose().copy()
for country in country_ls:
	rv["daily_%s_tm1" % country] = rv["daily_%s" % country].shift()
	rv_rlt = sm.ols(formula = "daily_%s ~ daily_%s_tm1" % (country, country), data = rv).fit()
	print ("%s: c = %s, phi = %s, t_c = %s, t_phi = %s, R^2 = %s" 
		% (country, rv_rlt.params[0], rv_rlt.params[1], 
			rv_rlt.params[0]/rv_rlt.bse[0], rv_rlt.params[1]/rv_rlt.bse[1], 
			rv_rlt.rsquared))

# part 3h&i
print ("\n################## Part 3i ##################\n")
for country in country_ls:
	col_name = "daily_%s" % country
	dropna_df = pd.DataFrame(rv.dropna(subset = [col_name])[col_name].copy(), columns = [col_name])
	dropna_df[col_name] = dropna_df.replace(dropna_df.nlargest(10, col_name), np.NaN)
	dropna_df["daily_%s_tm1" % country] = dropna_df["daily_%s" % country].shift()
	rvrm_rlt = sm.ols(formula = "daily_%s ~ daily_%s_tm1" % (country, country), data = dropna_df).fit()
	print ("%s: c = %s, phi = %s, t_c = %s, t_phi = %s, R^2 = %s" 
		% (country, rvrm_rlt.params[0], rvrm_rlt.params[1], 
			rvrm_rlt.params[0]/rvrm_rlt.bse[0], rvrm_rlt.params[1]/rvrm_rlt.bse[1], 
			rvrm_rlt.rsquared))