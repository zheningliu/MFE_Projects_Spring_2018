# 230A PS 1
import numpy as np
from sympy import Symbol, solve
import math
n = 5
t = [i+1 for i in range(n)]
r = [4.0, 4.6, 5.0, 5.7, 6.1]
p = []

for i in range(5):
	p.append(1 / pow(1+r[i]/100,t[i]))

# part a
print (p)

# part c
coupon = [80 for i in range(n)]
coupon[n-1] = coupon[n-1] + 1000
curr_p = np.dot(p,coupon)
print (curr_p)

x = Symbol("x")
print (solve(math.exp(-0.05)*pow((1+x/2),2)-pow(1.046,2)))
