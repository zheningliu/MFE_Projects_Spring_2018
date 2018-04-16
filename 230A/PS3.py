import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

np.random.seed(1000)
n = 1000
R0 = 1.025
E = np.matrix([[1.10], [1.12], [1.07]])
Var = np.matrix([[0.0600, 0.0377, 0.0259], [0.0377, 0.0950, 0.0285], [0.0259, 0.0285, 0.0700]])

def portfolio_prop(w0, w, exp=E, var_cov=Var, R=R0):
	E_port = w0 * R + np.dot(w.T, exp)
	Var_port = np.dot(np.dot(w.T, var_cov), w)
	SR = (E_port-R) / np.sqrt(Var_port)
	return E_port[0,0], Var_port[0,0], SR[0,0]

# part 1a
E_R = []
Var_R = []
for i in range(n):
	w = np.asmatrix(np.random.normal(0, 1, 3)).T
	w0 = 1 - sum(w)
	E_rand, Var_rand, SR_rand = portfolio_prop(w0, w)
	E_R.append(E_rand)
	Var_R.append(np.sqrt(Var_rand))
plt.plot(Var_R, E_R, 'rx')
plt.show()


# part 1b
print ("################## Part 1b ##################\n")
w_risky = np.dot(np.linalg.inv(Var), E-R0)
w0_risky = 1 - sum(w_risky)
E_risky, Var_risky, SR_risky = portfolio_prop(w0_risky, w_risky)
print ("Mean: %s, Standard Deviation: %s, Sharpe: %s" % (E_risky, np.sqrt(Var_risky), SR_risky))

# part 1d
print ("\n################## Part 1d ##################\n")
w_m = np.dot(np.linalg.inv(Var), E-R0) / sum(np.dot(np.linalg.inv(Var), E-R0))
w0_m = 0
E_m, Var_m, SR_m = portfolio_prop(w0_m, w_m)
print ("Mean: %s, Standard Deviation: %s, Sharpe: %s" % (E_m, np.sqrt(Var_m), SR_m))

# part 1e
print ("\n################## Part 1e ##################\n")
beta = np.dot(w_m.T, Var) / Var_m
print (beta)

# part 2a&b
print ("\n################## Part 2a&b ##################\n")
raw = pd.read_csv("PS3_q2_rawdata.csv")
q2_df = np.log(1 + raw.iloc[:,1:]) * 12
er = q2_df.iloc[:,[0,2,3,4]].sub(q2_df.iloc[:,1], axis=0)
Rm_e = np.log(1+0.022) * 12 - q2_df.iloc[:,1].mean()
for i in range(3):
	result = sm.ols(formula = "%s ~ sp500" % (er.columns.values[i+1]), data = er).fit()
	print ("%s. Stock %s:\n" % (i+1, er.columns.values[i+1]))
	print ("Paramters:\n%s\n" % (result.params))
	print("R2: %s\n" % (result.rsquared))
	R_stock = result.params[0] + Rm_e * result.params[1]
	print ("Expected return when monthly market return is 2.2%%: %s\n" % (R_stock))

# part 2c
print ("\n################## Part 2c ##################\n")
index = raw.index[raw['Period']==20070131][0]
Rf = q2_df.iloc[:,1].mean(axis=0)
E_3 = q2_df.iloc[:index,2:5].mean(axis=0)
Var_3 = q2_df.iloc[:index,2:5].cov()
w_3 = np.dot(np.linalg.inv(Var_3), np.asmatrix(E_3-Rf).T) / sum(np.dot(np.linalg.inv(Var_3), np.asmatrix(E_3-Rf).T))
w0_3 = 0
E_mv, Var_mv, SR_mv = portfolio_prop(w0_3, w_3, np.asmatrix(E_3).T, np.asmatrix(Var_3), Rf)
print ("Mean: %s, Standard Deviation: %s, Sharpe: %s" % (E_mv, np.sqrt(Var_mv), SR_mv))

# part 2d
print ("\n################## Part 2d ##################\n")
beta_3 = np.dot(w_3.T, Var_3) / Var_mv
E3_e = E_3 - Rf
Emv_e = E_mv - Rf
print (E3_e, beta_3, Emv_e)

# part 2e
print ("\n################## Part 2e ##################\n")
E_after = q2_df.iloc[index:,2:5].mean(axis=0)
Var_after = q2_df.iloc[index:,2:5].cov()
Emv_after, Varmv_after, SRmv_after = portfolio_prop(w0_3, w_3, np.asmatrix(E_after).T, np.asmatrix(Var_after), Rf)
print ("Mean: %s, Standard Deviation: %s, Sharpe: %s" % (Emv_after, np.sqrt(Varmv_after), SRmv_after))
beta_after = np.dot(w_3.T, Var_after) / Varmv_after
Ee_after = E_after - Rf
Eemv_after = Emv_after - Rf
print (Ee_after, beta_after, Eemv_after)