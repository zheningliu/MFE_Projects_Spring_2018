import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

def main():
	# pf_df = pd.read_csv("portfolio_returns_48.csv")
	# factor_df = pd.read_csv("factors_ff.csv")
	update_df = pd.read_csv("48_Industry_Portfolios_update.csv", na_values=[-99.99,99.99])
	update_factor = pd.read_csv("FF_Research_Data_Factors.csv", na_values=[-99.99,99.99])
	m = 48
	# # convert to log return
	# factor_ret = np.log(1 + factor_df.iloc[:,1:] / 100)
	# pf_ret_e = np.log(1 + pf_df.iloc[:,1:] / 100).sub(factor_ret.iloc[:,-1], axis=0)	# R_e = R_i - R_f
	update_factor_ret = np.log(1 + update_factor.iloc[:,1:] / 100)
	update_ret_e = np.log(1 + update_df.iloc[:,1:] / 100).sub(update_factor_ret.iloc[:,-1], axis=0)

	# # part 2a
	# print ("\n################## Part 2a ##################\n")
	# pf_ret_e.loc[:,'Rm_e'] = factor_ret.iloc[:,0]
	# stats_df = run_stats(pf_ret_e, ['Rm_e'], pf_ret_e.columns.values[:-1])
	# stats_df = stats_df.set_index(pf_ret_e.columns.values[:-1])
	# print (stats_df)

	# # part 2b
	# t = [i for i in range(m)]
	# alpha_plot(t, stats_df[['norm_alpha']].squeeze())

	# # part 2a
	# print ("\n################## Part 3a ##################\n")
	# pf_ret_e.loc[:,'SMB'] = factor_ret.iloc[:,1].sub(factor_ret.iloc[:,-1], axis=0)
	# pf_ret_e.loc[:,'HML'] = factor_ret.iloc[:,2].sub(factor_ret.iloc[:,-1], axis=0)
	# xlab = ['Rm_e', 'SMB', 'HML']
	# FF_stats_df = run_stats(pf_ret_e, xlab, pf_ret_e.columns.values[:-3])
	# FF_stats_df = FF_stats_df.set_index(pf_ret_e.columns.values[:-3])
	# print (FF_stats_df)

	# # part 3b
	# t = [i for i in range(m)]
	# alpha_plot(t, FF_stats_df[['norm_alpha']].squeeze())

	# # part 3c
	# excess_ret = pf_ret_e.iloc[:,:-3].mean(axis=0)
	# for i in range(3):
	# 	beta_plot(FF_stats_df[["beta%s" % (i+1)]].squeeze(), excess_ret.squeeze(), pf_ret_e.columns.values[m+i])

	# part 4
	update_ret_e.loc[:,'Rm_e'] = update_factor_ret.iloc[:,0]
	update_stats_df = run_stats(update_ret_e, ['Rm_e'], update_ret_e.columns.values[:-1])
	update_stats_df = update_stats_df.set_index(update_ret_e.columns.values[:-1])
	t = [i for i in range(m)]
	alpha_plot(t, update_stats_df[['norm_alpha']].squeeze())
	update_ret_e.loc[:,'SMB'] = update_factor_ret.iloc[:,1].sub(update_factor_ret.iloc[:,-1], axis=0)
	update_ret_e.loc[:,'HML'] = update_factor_ret.iloc[:,2].sub(update_factor_ret.iloc[:,-1], axis=0)
	xlab = ['Rm_e', 'SMB', 'HML']
	update_stats3_df = run_stats(update_ret_e, xlab, update_ret_e.columns.values[:-3])
	update_stats3_df = update_stats3_df.set_index(update_ret_e.columns.values[:-3])
	alpha_plot(t, update_stats3_df[['norm_alpha']].squeeze())


def run_stats(df, xlabel, ylabel):
	ncol = len(df.columns) - len(xlabel)
	alpha = []
	beta_dict = {}
	for j in range(len(xlabel)):
		beta_dict["beta%s" % (j+1)] = []
	R2 = []
	norm_alpha = []
	for i in range(ncol):
		result = sm.ols(formula = "%s ~ %s" % (ylabel[i], " + ".join([xlab for xlab in xlabel])), data = df).fit()
		alpha.append(result.params[0])
		for j in range(len(xlabel)):
			beta_dict["beta%s" % (j+1)].append(result.params[j+1])
		R2.append(result.rsquared)
		norm_alpha.append(result.params[0] / result.bse[0])
	combine_dict = {'alpha':alpha, 'R2':R2, 'norm_alpha':norm_alpha}
	combine_dict.update(beta_dict)
	stats_df = pd.DataFrame(combine_dict)
	return stats_df


def alpha_plot(t, norm_alpha):
	plt.plot(t, norm_alpha, 'bo', t, [2]*len(t), 'r-', t, [-2]*len(t), 'r-')
	plt.title("Alphas with 2 Standard Deviation")
	plt.show()
	print (np.nonzero(np.abs(norm_alpha)>=2))


def beta_plot(beta, ret_e, beta_name):
	m, b = np.polyfit(beta, ret_e, 1)
	plt.plot(beta, ret_e, 'bo', beta, m*beta + b, '-')
	plt.title("%s Beta Against Excess Returns" % (beta_name))
	plt.show()


if __name__ == "__main__":
	main()