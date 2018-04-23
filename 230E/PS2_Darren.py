import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pandas as pd
import math
from sklearn import linear_model
import matplotlib.mlab as mlab

def main(portfolio_file_name, factor_file_name):
    wr = [6] + [7 for i in range(48)] #.txt files aren't separated with tabs; used to split by spaces
    wf = [6] + [8 for i in range(4)]

    df_returns = pd.read_fwf(portfolio_file_name,widths = wr,header = 11)
    df_factors = pd.read_fwf(factor_file_name,widths = wf,header = 3)
    

    #1.
    returns = df_returns.iloc[:,1:].apply(lambda x: np.log(1+x/100))
    factors = df_factors.iloc[:,1:].apply(lambda x: np.log(1+x/100))
    #print(factors)
    #print(returns)

    #2.
    n = len(returns)
    port_n = len(returns.columns)
    col_names = list(returns.columns.values)

    #a.)
    reg = linear_model.LinearRegression()
    alphas = []
    betas = []
    
    const = np.ones((n,1))
    X_1 = np.column_stack((const, factors[["Mkt-RF"]]))#factors.iloc[:,0])) #matrix of 1's and mkt - rf
    Xt = np.transpose(X_1)
    XtX_inv = inv(np.dot(Xt,X_1)) #(X'X)^(-1)
    
    for i in range(port_n):
        coef = regression_given_X(X_1, XtX_inv, returns.iloc[:,i] - factors.iloc[:,3]) #y = return_i - rf
        alphas.append(coef[0])
        betas.append(coef[1])

        
        reg.fit(X_1, returns.iloc[:,i] - factors.iloc[:,3])
        yhat = reg.predict(X_1)
        SS_Residual = sum((returns.iloc[:,i] - factors.iloc[:,3]-yhat)**2)
        SS_Total = sum((returns.iloc[:,i] - factors.iloc[:,3]-np.mean(returns.iloc[:,i] - factors.iloc[:,3]))**2)
        r_squared1 = 1 - (float(SS_Residual))/SS_Total
        #print(r_squared1)
        #print(reg.coef_)
        #print(reg.intercept_)

    #print(alphas)
    #print(betas)
    #for i in range(48):
    #    print("{0} & {1:.6f} & {2:.6f}\\\\ \\hline".format(col_names[i], alphas[i],betas[i]))
        
    #b.)
    plt.hist(alphas, normed=True)
    #plt.show()
    mu = np.mean(alphas)    
    variance = np.var(alphas)
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x,mlab.normpdf(x, mu, sigma))
    plt.title("Histogram and distribution of Alphas")
    plt.show()
    
    #c.)
    r_squares = []
    for i in range(port_n):
        r_squares.append(r_squared(alphas[i],betas[i],factors.iloc[:,0],returns.iloc[:,i] - factors.iloc[:,3]))
        
    #print(r_squares)
    #for i in range(48):
    #    print("{0} & {1:.6f}\\\\ \\hline".format(col_names[i], r_squares[i]))

    #d.)
    excess_returns = []
    
    for i in range(port_n):
        excess_returns.append(np.mean(returns.iloc[:,i]) - np.mean(factors.iloc[:,3]))
    plt.scatter(betas,excess_returns)
    plt.title("Beta Vs. Excess Returns")
    plt.xlabel("Beta")
    plt.ylabel("Excess Return")
    fit = np.polyfit(betas,excess_returns,deg=1)
    plt.plot(np.array(betas), fit[1] + fit[0] * np.array(betas))
    plt.show()


    #3.
    #a.)
    reg2 = linear_model.LinearRegression()
    alphas2 = []
    betas2 = []
    
    const2 = np.ones((n,1))
    X_12 = np.column_stack((const2, factors[["Mkt-RF"]], factors[["SMB"]], factors[["HML"]]))#factors.iloc[:,0])) #matrix of 1's, Mkt - RF, SMB, and HML
    Xt2 = np.transpose(X_12)
    XtX_inv2 = inv(np.dot(Xt2,X_12)) #(X'X)^(-1)
    
    for i in range(port_n):
        coef2 = regression_given_X(X_12, XtX_inv2, returns.iloc[:,i] - factors.iloc[:,3]) #y = return_i - rf
        alphas2.append(coef2[0])
        betas2.append(([coef2[1],coef2[2], coef2[3]]))

        reg2.fit(X_12, returns.iloc[:,i] - factors.iloc[:,3])
        yhat2 = reg2.predict(X_12)
        SS_Residual2 = sum((returns.iloc[:,i] - factors.iloc[:,3]-yhat2)**2)
        SS_Total2 = sum((returns.iloc[:,i] - factors.iloc[:,3]-np.mean(returns.iloc[:,i] - factors.iloc[:,3]))**2)
        r_squared12 = 1 - (float(SS_Residual2))/SS_Total2
        #print(r_squared12)
        #print(reg2.coef_)
        #print(reg2.intercept_)

    #print(alphas2)
    #print(betas2)
    for i in range(48):
        print("{0} & {1:.6f} & {2:.6f} & {3:.6f} & {4:.6f}\\\\ \\hline".format(col_names[i], alphas2[i],betas2[i][0],betas2[i][1],betas2[i][2]))

    #b.)
    plt.hist(alphas2, normed=True)
    #plt.show()
    mu = np.mean(alphas2)    
    variance = np.var(alphas2)
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x,mlab.normpdf(x, mu, sigma))
    plt.title("Histogram and distribution of Alphas")
    plt.show()
    

    #c.)
    r_squares2 = []
    
    for i in range(port_n):
        r_squares2.append(r_squared(alphas2[i],betas2[i],factors.iloc[:,0:3],returns.iloc[:,i] - factors.iloc[:,3]))
        
    #print(r_squares2)
    for i in range(48):
        print("{0} & {1:.6f}\\\\ \\hline".format(col_names[i], r_squares2[i]))

    #d.)
    excess_returns = []
    beta1 = []
    beta2 = []
    beta3 = []
    
    for i in range(port_n):
        excess_returns.append(np.mean(returns.iloc[:,i]) - np.mean(factors.iloc[:,3]))
        beta1.append(betas2[i][0])
        beta2.append(betas2[i][1])
        beta3.append(betas2[i][2])

    plt.scatter(beta1,excess_returns)
    plt.title("Mkt_RF Beta Vs. Excess Returns")
    plt.xlabel("Beta")
    plt.ylabel("Excess Return")
    fit = np.polyfit(beta1,excess_returns,deg=1)
    plt.plot(np.array(beta1), fit[1] + fit[0] * np.array(beta1))
    plt.show()
        
    
    plt.scatter(beta2,excess_returns)
    plt.title("SMB Beta Vs. Excess Returns")
    plt.xlabel("Beta")
    plt.ylabel("Excess Return")
    fit = np.polyfit(beta2,excess_returns,deg=1)
    plt.plot(np.array(beta2), fit[1] + fit[0] * np.array(beta2))
    plt.show()
    
    plt.scatter(beta3,excess_returns)
    plt.title("HML Beta Vs. Excess Returns")
    plt.xlabel("Beta")
    plt.ylabel("Excess Return")
    fit = np.polyfit(beta3,excess_returns,deg=1)
    plt.plot(np.array(beta3), fit[1] + fit[0] * np.array(beta3))
    plt.show()



def regression_given_X(X,XtX_inv,y):
    '''Linear regression with constant factors to save computation time'''
    Xt_y = np.dot(np.transpose(X),y)
    beta = np.dot(XtX_inv,Xt_y)
    return beta
    
def r_squared(alpha,beta,X,y):
    y_bar = np.mean(y)
    sum_res = 0
    sum_tot = 0
    for i in range(len(X)):
        sum_res += (y.iloc[i] - alpha - np.dot(np.transpose(beta), X.iloc[i]))**2
        sum_tot += (y.iloc[i] - y_bar)**2
    return 1 - sum_res/sum_tot




if __name__ == "__main__":
    portfolio_file_name = "48_Industry_Portfolios.txt"
    factor_file_name = "F-F_Research_Data_Factors.txt"
    #portfolio_file_name = "portfolio_returns_48.txt"
    #factor_file_name = "factors_ff.txt"
    main(portfolio_file_name, factor_file_name)
    

    
