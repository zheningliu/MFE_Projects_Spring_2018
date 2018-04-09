# 230A PS 1
import numpy as np
from sympy import Symbol, solve
import math

# problem 1
n = 5
t = [i+1 for i in range(n)]
r = [4.0, 4.6, 5.0, 5.7, 6.1]
p = []

for i in range(5):
	p.append(1000 / math.exp(r[i]/100*t[i]))

# part a
print (p)

# part c
coupon = [80 for i in range(n)]
coupon[n-1] = coupon[n-1] + 1000
curr_p = np.dot(p,coupon)
print (curr_p)

# part b
r_22 = Symbol("r_22")
print ("The semi-annually compounded bond rate is: ", solve(math.exp(-0.05)*pow((1+r_22/2),2)-math.exp(2*0.046)))

# problem 2
# part a
t = 30
gg = 0.1
r = 0.035
cf = []
for i in range(t):
	if i < 6:
		cf.append(1000*pow(1+gg,i))
	else:
		cf.append(cf[len(cf)-1])
discount = [1/pow(1+r,i+1) for i in range(t)]
pv = np.dot(cf,discount)
print (pv)

# part b
print (10/(0.12-0.03))

# problem 3
# part a
t = [0.5, 1, 1.5, 2]
price = [98, 95, 101, 104]
coupon = [0, 0, 6.2, 8]
fv = 100
r = []
x = Symbol("x")
for i in range(4):
	if i < 2:
		r.append(math.log(fv/price[i])/t[i])
	elif i == 2:
		r3 = solve(coupon[i]/2/x+coupon[i]/2/pow(x,2)+(fv+coupon[i]/2)/pow(x,3)-price[i])
		r.append(math.log(r3[0])*2)
	elif i == 3:
		r4 = solve(coupon[i]/2/x+coupon[i]/2/pow(x,2)+coupon[i]/2/pow(x,3)+(fv+coupon[i]/2)/pow(x,4)-price[i])
		r.append(math.log(r4[1])*2)
print ("The spot rates are: ", r)

# part b
fr = []
for j in range(3):
	fr.append(2*(r[j+1]*t[j+1]-r[j]*t[j]))
print ("The forward rates are: ", fr)

# part c
df = []
for k in range(len(t)):
	df.append(1/math.exp(r[k]*t[k]))
cr = []
for time in range(len(t)):
	cr.append(2*(fv-fv*df[time])/sum(df[:time+1]))
print ("The coupon rates are: ", cr)

# part e
payment = [3.5, 3.5, 3.5, 103.5]
bond_price = np.dot(payment, df)
YTM = solve(3.5/x+3.5/pow(x,2)+3.5/pow(x,3)+103.5/pow(x,4)-bond_price)
print ("The YTM is: ", math.log(YTM[1])*2)
