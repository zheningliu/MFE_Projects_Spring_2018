import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pandas as pd
import math


def main(file_name):
    df = pd.read_csv(file_name, thousands=",")

    #1, 3
    dp = (df.iloc[660:,2].divide(df.iloc[660:,1])).apply(lambda x: np.log(1+x/100))
    #print(dp)

    #2, 3
    ep = (df.iloc[660:,3].divide(df.iloc[660:,1])).apply(lambda x: np.log(1+x/100))
    #print(ep)

    #4
    CRSP_SPvw = df.iloc[660:,16].apply(lambda x: np.log(1+x/100)) * 12
    CRSP_SPvwx = df.iloc[660:,17].apply(lambda x: np.log(1+x/100)) * 12
    print("CRSP_SPvw mean: {}".format(np.mean(CRSP_SPvw)))
    print("CRSP_SPvw std: {}".format(np.std(CRSP_SPvw)))
    print("CRSP_SPvwx mean: {}".format(np.mean(CRSP_SPvwx))) #Excess return
    print("CRSP_SPvwx std: {}".format(np.std(CRSP_SPvwx)))
    
    r = CRSP_SPvw


    #5
    dp_1 = dp.iloc[1:] #2nd data point and onwards
    dp_coef = regression(dp[:-1],dp_1)
    print("DP AR(1) coefficients: {}, {}".format(dp_coef[0],dp_coef[1]))
    dp_std_beta = standard_error(predict(dp_coef,dp[:-1]), dp_1)
    print("DP AR(1) beta std: {}".format(dp_std_beta))
    dp_errors = predict(dp_coef,dp[:-1]) - dp_1
    #print(r_squared(dp_coef[0],dp_coef[1],dp[:-1],dp_1))
    #print(dp_errors)
    #print(standard_error_alpha(dp_std_beta, dp[:-1]))
    print("DP AR(1) t-statistic: {}".format(dp_coef[1]/dp_std_beta))
    

    ep_1 = ep.iloc[1:] #2nd data point and onwards
    ep_coef = regression(ep[:-1],ep_1)
    print("EP AR(1) coefficients: {}, {}".format(ep_coef[0], ep_coef[1]))
    ep_std_beta = standard_error(predict(ep_coef, ep[:-1]),ep_1)
    print("EP AR(1) beta std: {}".format(ep_std_beta))
    ep_errors = predict(ep_coef,ep[:-1]) - ep_1
    print("EP AR(1) t-statistic: {}".format(ep_coef[1]/ep_std_beta))
    #print(standard_error_alpha(ep_std_beta, ep[:-1]))

    #6
    dp_r_coef = regression(dp[:-1],r[1:])
    ep_r_coef = regression(ep[:-1],r[1:])
    dp_r_errors = predict(dp_r_coef,dp[:-1]) - r[1:]
    ep_r_errors = predict(ep_r_coef,ep[:-1]) - r[1:]
    dp_r_std_beta = standard_error(predict(dp_r_coef,dp[:-1]), r[1:])
    ep_r_std_beta = standard_error(predict(ep_r_coef,ep[:-1]), r[1:])

    print("DP coefficients with bias: {}, {}".format(dp_r_coef[0], dp_r_coef[1]))
    print("EP coefficients with bias: {}, {}".format(ep_r_coef[0], ep_r_coef[1]))
    print("DP and r t-statistic: {}".format(dp_r_coef[1]/dp_r_std_beta))
    print("EP and r t-statistic: {}".format(ep_r_coef[1]/ep_r_std_beta))

    #plt.scatter(dp[:-1], r[1:])
    #plt.scatter(ep[:-1], r[1:])
    #plt.plot(dp,dp_r_coef[0] + dp_r_coef[1] * dp)
    #plt.show()
    
    #Can correct for beta
    cov_dp = np.cov(dp_errors, dp_r_errors)
    cov_ep = np.cov(ep_errors, ep_r_errors)
    dp_unbias_beta = dp_r_coef[1] + cov_dp[1,0]/cov_dp[0,0] * (1+3*dp_coef[1])/np.size(dp_1)
    ep_unbias_beta = ep_r_coef[1] + cov_ep[1,0]/cov_ep[0,0] * (1+3*ep_coef[1])/np.size(ep_1)
    print("DP corrected beta for bias: {} {}".format(dp_r_coef[0],dp_unbias_beta))
    print("EP corrected beta for bias: {} {}".format(ep_r_coef[0],ep_unbias_beta))


    dp_r_std_unbias_beta = standard_error(predict(np.column_stack((dp_r_coef[0],dp_unbias_beta)),dp[:-1]), r[1:])
    ep_r_std_unbias_beta = standard_error(predict(np.column_stack((ep_r_coef[0],ep_unbias_beta)),ep[:-1]), r[1:])
    print("DP and r corrected t-statistic: {}".format(dp_unbias_beta/dp_r_std_unbias_beta))
    print("EP and r corrected t-statistic: {}".format(ep_unbias_beta/ep_r_std_unbias_beta))


    
    print("DP biased R^2: {}".format(r_squared(dp_r_coef[0], dp_r_coef[1], dp[:-1], r[1:])))
    print("EP biased R^2: {}".format(r_squared(ep_r_coef[0], ep_r_coef[1], ep[:-1], r[1:])))

    print("DP unbiased R^2: {}".format(r_squared(dp_r_coef[0], dp_unbias_beta, dp[:-1], r[1:])))
    print("EP unbiased R^2: {}".format(r_squared(ep_r_coef[0], ep_unbias_beta, ep[:-1], r[1:])))

    #7
    rf = df.iloc[660:,10].apply(lambda x: np.log(1+x/100))
    rf_1 = rf.iloc[1:] #2nd data point and onwards
    rf_coef = regression(rf[:-1],rf_1)
    rf_errors = predict(rf_coef,rf[:-1]) - rf_1
    print("Risk free AR(1) coefficients: {}, {}".format(rf_coef[0], rf_coef[1]))
    rf_std_beta = standard_error(predict(rf_coef,rf[:-1]), rf_1)
    #print(rf_std_beta)
    print("Risk free AR(1) t-statistic: {}".format(rf_coef[1]/rf_std_beta))

    
    rf_r_coef = regression(rf_1, r[1:])
    print("Risk free and r coefficients with bias: {}, {}".format(rf_r_coef[0], rf_r_coef[1]))
    rf_r_std_beta = standard_error(predict(rf_r_coef,rf[:-1]), r[1:])
    print("Risk free and r t-statistic with bias: {}".format(rf_r_coef[1]/rf_r_std_beta))
    print("Risk free R^2 with bias: {}".format(r_squared(rf_r_coef[0], rf_r_coef[1], rf_1, r[1:])))
    rf_r_errors = predict(rf_r_coef,rf[:-1]) - r[1:]

    
    cov_rf = np.cov(rf_errors, rf_r_errors)
    rf_unbias_beta = rf_r_coef[1] + cov_rf[1,0]/cov_rf[0,0] * (1+3*rf_coef[1])/np.size(rf_1)
    rf_r_std_unbias_beta = standard_error(predict(np.column_stack((rf_r_coef[0],rf_unbias_beta)),rf[:-1]), r[1:])
    print("Rf corrected beta for bias: {} {}".format(rf_r_coef[0],rf_unbias_beta))
    print("Rf and r corrected t-statistic: {}".format(rf_unbias_beta/rf_r_std_unbias_beta))
    print("Rf unbiased R^2: {}".format(r_squared(rf_r_coef[0], rf_unbias_beta, rf[:-1], r[1:])))

    
    
def regression(X,y):
    const = np.ones((np.size(X),1))
    X1 = np.column_stack((const, X))
    Xt = np.transpose(X1)
    XtX_inv = inv(np.dot(Xt,X1))
    
    Xt_y = np.dot(np.transpose(X1),y)
    beta = np.dot(XtX_inv,Xt_y)
    return beta

def standard_error(predicted, actual):
    n = np.size(actual)
    sum_error = 0
    for i in range(n):
        sum_error += (predicted[i] - actual.iloc[i])**2
    return np.sqrt(sum_error/(n-2))

def standard_error_alpha(std_beta, X):
    n = np.size(X)
    sum_x = 0
    for i in range(n):
        sum_x += (X.iloc[i])**2
    return std_beta * np.sqrt(1/n * sum_x)
    

def predict(beta, X):
    n = np.size(X)
    const = np.ones((n,1))
    prediction = np.zeros(n)
    for i in range(n):
        prediction[i] = np.dot(np.matrix(beta), np.transpose(np.column_stack((1,X.iloc[i]))))
    return prediction

def r_squared(alpha,beta,X,y):
    y_bar = np.mean(y)
    sum_res = 0
    sum_tot = 0
    for i in range(len(X)):
        sum_res += (y.iloc[i] - alpha - np.dot(np.transpose(beta), X.iloc[i]))**2
        sum_tot += (y.iloc[i] - y_bar)**2
    return 1 - sum_res/sum_tot


if __name__ == "__main__":
    main("PredictorData2015_GoyalWelch.csv")
