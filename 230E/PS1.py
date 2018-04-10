import pandas as pd
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import random

def main():
	random.seed(12345)

	f = open('PS1.txt','w')
	df = pd.read_csv("Portfolios_Formed_on_ME_daily.csv")
	index = df.index[df["Year"]==2010][0]	#only use data till end of 2009
	port_df = df.iloc[:index,3:].apply(lambda x: np.log(1+x/100))
	n = len(port_df)
	port_n = len(port_df.columns)
	col_names = list(port_df.columns.values)
	# part 1a
	f.write ("Portfolio daily mean is: \n %s\n" % (mean_df(port_df)))
	f.write ("Portfolio daily standard deviation is: \n %s\n" % (std_df(port_df)))

	# part 1b
	f.write ("Portfolio monthly mean is: \n %s\n" % (mean_df(port_df)).multiply(21))
	f.write ("Portfolio monthly standard deviation is: \n %s" % (std_df(port_df)).multiply(math.sqrt(21)))
	f.write ("Portfolio yearly mean is: \n %s\n" % (mean_df(port_df)).multiply(252))
	f.write ("Portfolio yearly standard deviation is: \n %s\n" % (std_df(port_df)).multiply(math.sqrt(21)))

	# part 1c
	pre = df.index[df['Year']==1945][0]
	f.write ("Portfolio daily mean pre-1945 is: \n %s\n" % (mean_df(port_df.iloc[:pre,:])))
	f.write ("Portfolio daily mean post-1945 is: \n %s\n" % (mean_df(port_df.iloc[pre:,:])))

	# part 1e
	t, t_ppf = t_test_compare(port_df.iloc[:pre,:], port_df.iloc[pre:,:], 0.05)
	for i in range(port_n):
		f.write ("t: %s, t*: %s\n" % (t[i],t_ppf))
		if t[i] > t_ppf:
			f.write ("The pre and post 1945 means of portfolio %s are significantly different at 5%% confidence interval\n" 
				% (col_names[i]))
		else:
			f.write ("The pre and post 1945 means of portfolio %s are NOT significantly different at 5%% confidence interval\n" 
				% (col_names[i]))

	# part 2a
	for i in range(port_n):
		phi, c, epsilon = find_par(port_df.iloc[0:n-1,i].reset_index(drop=True), 
			port_df.iloc[1:n,i].reset_index(drop=True))
		f.write ("The OLS regression of portfolio %s lagged return is: r_VW = %s + %sr_i + %s\n"
			% (col_names[i], phi, c, np.mean(epsilon)))

	# part 2b
	for i in range(port_n):
		e_t = np.squeeze(np.asarray(find_par(port_df.iloc[1:n-1,i].reset_index(drop=True), 
			port_df.iloc[2:n,i].reset_index(drop=True))[2]))
		e_prev = np.squeeze(np.asarray(find_par(port_df.iloc[0:n-1,i].reset_index(drop=True), 
			port_df.iloc[1:n,i].reset_index(drop=True))[2]))
		f.write ("Portfolio %s has: %s\n" % (col_names[i], durbin_watson(e_t,e_prev)))

	# part 3a
	for i in range(3):
		phi, c, epsilon = find_par(port_df.iloc[0:n-1,i].reset_index(drop=True), 
			port_df.iloc[1:n,-1].reset_index(drop=True))
		f.write ("The OLS regression of portfolio %s lagged return is: r_VW = %s + %sr_i + %s\n"
			% (col_names[i], phi, c, np.mean(epsilon)))
		# part 3b
		std_err = std_df(epsilon)
		var_cov = var_cov_mat(port_df.iloc[0:n-1,i].reset_index(drop=True), std_err)	#var(beta)=sigma^2(X'X)^-1[1,1]
		s_phi = np.sqrt(var_cov[1,1])	#n-k-1 dof
		t, t_ppf = t_test(phi, 0, s_phi, 0, n, 0, 0.05)
		if t > t_ppf:
			f.write ("t: %s, t*: %s. The slope phi of portfolio %s is significant at 5%% confidence interval\n" 
				% (t, t_ppf, col_names[i]))
		else:
			f.write ("t: %s, t*: %s. The slope phi of portfolio %s is NOT significant at 5%% confidence interval\n" 
				% (t, t_ppf, col_names[i]))

	# part 4
	Y_0 = 0
	c = 0
	phi = 0.1
	T = 100
	E_Yt = c / (1 - phi)
	Var_Yt = 1 / (1 - pow(phi, 2))
	f.write ("E[Y_t] = %s, Var(Y_t) = %.2f\n" % (E_Yt, Var_Yt))
	sim_dict = {}
	for i in range(1000):
		sim_dict[i] = ar1_generator(T, Y_0, phi, i)
	sim_df = pd.DataFrame(sim_dict)	#N by j
	f.write ("The means of Y_t across time is \n%s\n" % (sim_df.mean(axis=0).to_string()))
	f.write ("The means of Y_t at each t across section is \n%s\n" % (sim_df.mean(axis=1).to_string()))
	plt.plot(sim_df.mean(axis=0))
	plt.title("Means of Y_t across time")
	plt.show()
	plt.plot(sim_df.mean(axis=1))
	plt.title("Means of Y_t at each t across section")
	plt.show()

	f.close()

def mean_df(df):
	"""
	Calculate mean of dataframe by column
	"""
	return df.mean()

def std_df(df):
	"""
	Calculate standard deviation of a dataframe by column
	"""
	return df.std()

def t_test_compare(df1, df2, CI):
	"""
	t-test to compare two means
	"""
	mu1 = df1.mean()
	mu2 = df2.mean()
	std1 = df1.std()
	std2 = df2.std()
	n1 = len(df1)
	n2 = len(df2)
	s_p = np.sqrt(((n1-1)*pow(std1,2)+(n2-1)*pow(std2,2))/(n1+n2-2))
	t = np.abs(mu1-mu2)/(s_p*np.sqrt(1/n1+1/n2))
	t_ppf = stats.t.ppf(1-CI/2, n1+n2-2)	#two-tail
	return t, t_ppf

def t_test(mu1, mu2, std1, std2, n1, n2, CI):
	"""
	Returns general t-test to check confidence interval
	"""
	s_p = np.sqrt(((n1-1)*pow(std1,2)+(n2-1)*pow(std2,2))/(n1+n2-2))
	if n2 == 0:
		t = np.abs(mu1-mu2)/(s_p*np.sqrt(1/n1))
	else:
		t = np.abs(mu1-mu2)/(s_p*np.sqrt(1/n1+1/n2))
	t_ppf = stats.t.ppf(1-CI/2, n1+n2-2)	#two-tail
	return t, t_ppf

def var_cov_mat(x, sigma):
	"""
	Returns variance-covariance matrix from linear regression model
	"""
	nrow = len(x)
	one = pd.Series([1 for i in range(nrow)])
	new_df = pd.concat([one,x], axis=1)
	return np.dot(new_df.T,new_df) * np.power(sigma,2)

def find_par(x, y):
	"""
	Returns linear regression model parameters
	"""
	nrow = len(x)
	one = pd.Series([1 for i in range(nrow)])
	new_df = pd.concat([one,x,y], axis=1)
	big_x = np.asmatrix(new_df.iloc[:,:2].values)
	big_y = np.asmatrix(new_df.iloc[:,2].values).T
	xtx = np.dot(big_x.T, big_x)
	xty = np.dot(big_x.T, big_y)
	weight = np.dot(np.linalg.inv(xtx), xty)	#beta = (X'X)^-1X'Y
	err = np.subtract(big_y, np.dot(big_x, np.asmatrix(weight)))
	return weight[0,0], weight[1,0], np.squeeze(np.asarray(err))

def durbin_watson(e_t, e_prev):
	"""
	Returns durbin watson result to test autocorrelation
	"""
	dw = sum(pow(e_t-e_prev[1:],2)) / sum(pow(e_t,2))
	result = ""
	if dw == 2:
		result = "No autocorrelation"
	elif 0 < dw < 2:
		result = "Positive correlation"
	elif 2 < dw < 4:
		result = "Negative correlation"
	else:
		result = "Invalid"
	return result

def ar1_generator(n, Y_0, phi, p):
	"""
	Returns a string of AR(1) simulation numbers
	"""
	Y_t = []
	samples = np.random.normal(0,1,n)
	for i in range(n):
		Y_t.append(phi*Y_0+samples[i])
		Y_0 = Y_t[i]
	return Y_t

if __name__ == "__main__":
	main()