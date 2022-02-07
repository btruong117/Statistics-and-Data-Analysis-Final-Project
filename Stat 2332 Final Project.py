#!/usr/bin/env python
# coding: utf-8

#Brian Truong
#Stat 2332 Final Project
#Dr. Chowdhury

import numpy as np
import pandas as pd
#q1
d1 = pd.read_csv("final.csv")
np.shape(d1)


# In[101]:


#q2
d1.columns
d1.head(10)


# In[102]:


#q3
d1.info()


# In[103]:


#q4
d1.dtypes
del d1['ID']
d1.info()


# In[104]:


#q5
d1.describe()


# In[105]:


#q5
temp = d1[['MOFB', 'YOB', 'AOR']].copy()
#print(temp.head(10))
num_nan = temp.isna().sum()
print(num_nan)


# In[106]:


#q6
d2 = d1[['RMOB', 'WI', 'RCA', 'Religion', 'Region',
        'AOR', 'HEL', 'DOBCMC', 'DOFBCMC', 'MTFBI', 'RW',
        'RH', 'RBMI']].copy()
d2.isna()


# In[107]:


#q7
d3 = d2.dropna()
d3.head(30).isna()


# In[108]:


#q8
d3.describe()


# In[109]:


#q9
x = ['DOBCMC', 'DOFBCMC', 'MTFBI']
d3["average"] = d3[x].mean(axis = 1)
d3['average']


# In[110]:


#q10
d3["NewReligion"] = d3['Religion']
d3.loc[d3['NewReligion'] != 1] = 2 
pd.crosstab(index = d3["NewReligion"], columns = 'count')


# In[111]:


#q11
pd.crosstab(index = d3["Region"], columns = "count")


# In[112]:


#q12
pd.crosstab(index = d3["Region"], columns = d3["Religion"])


# In[113]:


#q13
d3_byRegion = d3.groupby('Region')
d3_byRegion.mean().transpose()


# In[114]:


#q14
d3_byReligion = d3.groupby('Religion')
d3_byReligion.std().transpose()


# In[115]:


import matplotlib.pyplot as plt
y = d3['MTFBI']
plt.boxplot(y)
plt.title("Boxplot of MTFBI")


# In[116]:


#q16
r = d3['RCA']
plt.hist(r, color = "skyblue", ec = "red")
plt.title("Histogram of RCA")
plt.xlabel("Values of RCA")
plt.ylabel("Frequency")


# In[117]:


#q17
d3['Region'].value_counts(sort = False).plot.bar()


# In[118]:


#q18
labels = ['1', '2', '3', '4', '5', '6', '7']
cols = ['r', 'b', 'g', 'y', 'c', 'w', 'm']
sizes = ['1686', '4142', '2564', '2070', '2184', '1829', '1550']
plt.pie(sizes, explode = None, labels = labels, autopct = '%1.1f%%', colors = cols)
plt.title("Pie chart of Regions")
plt.show()


# In[119]:


#q19
b = d3['MTFBI']
h = d3['RCA']
reg = d3['Region']
ba = d3['Region'].value_counts(sort = False)
labels = ['1', '2', '3', '4', '5', '6', '7']
cols = ['r', 'b', 'g', 'y', 'c', 'w', 'm']
sizes = ['1686', '4142', '2564', '2070', '2184', '1829', '1550']

plt.subplot(2,2,2)
plt.hist(h)
plt.subplot(2,2,1)
plt.boxplot(b)
plt.subplot(2,2,3)
d3['Region'].value_counts(sort = False).plot.bar()
plt.subplot(2,2,4)
plt.pie(sizes, explode = None, labels = labels, autopct = '%1.1f%%', colors = cols)


# In[120]:


#q20
d4 = d3.groupby('WI', as_index = False)
np.shape(d4)
#d4.head(10)


# In[121]:


#q21
import statistics as ss
ds = [rows for _, rows in d4]
datalist = []
for i in range(len(ds)):
    datalist.append(pd.DataFrame({'WI' : [ss.mean(ds[i].WI)], 'MTFBI mean' : [ss.mean(ds[i].MTFBI)],
                                 'MTFBI min' : [min(ds[i].MTFBI)], 'MTFBI Max' : [max(ds[i].MTFBI)],
                                 'MTFBI Variance' : [ss.variance(ds[i].MTFBI)]
                                 ,'MTFBI Median' : [ss.median(ds[i].MTFBI)]}))
new_data = pd.concat(datalist)
print(new_data)


# In[122]:


#q22
import scipy.stats as st
st.stats.ttest_1samp(d3.MTFBI, 30)


# In[123]:


#q23
import scipy
scipy.stats.shapiro(d3.MTFBI)


# In[124]:


#q24
scipy.stats.ttest_ind(d3[d3.NewReligion == 1].MTFBI, d3[d3.NewReligion == 2].MTFBI)


# In[125]:


#q25
columns = ["DOBCMC","DOFBCMC","AOR","MTFBI","RW","RH", "RBMI"]
c1=d3[columns]
import matplotlib.pyplot as plt
plt.matshow(c1.corr())
# correltion matrix
c1.corr()


# In[126]:


#q27
import statsmodels.tools as sm
from statsmodels.api import OLS
y=d3.MTFBI
x=d3[['AOR','RW','Region']]
x1=sm.add_constant(x)
model = OLS(y, x1).fit()
model.summary()


# In[127]:


#q28 - q32
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
#q28
def sim(n):
    x = np.random.binomial(20,.70,n)
    u = np.random.normal(0,5,n)
    N = np.random.uniform(15,30,n)
    E = np.random.uniform(-1, 1, n)
    y= 50 + 10*x + 20 * u + 100 * N + E 
    y1=pd.DataFrame(y)
    return y1

a = sim(1000)

# mean and variance
np.mean(a)
np.var(a)
stats.ttest_1samp(a,1000)
# repeating 100 times
B=100
repeat=[sim(1000) for i in range(B)]
alldata=pd.concat(repeat)
# computing mean and variance from simulated data
simulated_mean=np.mean(alldata)
simulated_variance=np.var(alldata)
# Theoretical Mean and Variance
theoretical_mean= 640
theoretical_variance= 257920
# difference between them
#29
print("100 iterations mean: " , abs(simulated_mean-theoretical_mean))
#30
print("100 iterations variance: " , abs(simulated_variance-theoretical_variance))

C=500
repeat=[sim(1000) for i in range(C)]
alldata=pd.concat(repeat)
# computing mean and variance from simulated data
simulated_mean=np.mean(alldata)
simulated_variance=np.var(alldata)
# Theoretical Mean and Variance
theoretical_mean= 640
theoretical_variance= 257920
# difference between them
#31
print("500 iterations mean: " , abs(simulated_mean-theoretical_mean))
#32
print("500 iterations variance: ", abs(simulated_variance-theoretical_variance))


# In[128]:


#q33
import math
def fun(x, y, z):
    return (math.exp(x) - math.log(z**2))/(5+y)

x = 1
y = 2
z = 3
for i in range(5):
    print(fun(x,y,z))
    x += 1
    y += 1
    z += 1

    


# In[129]:


#q34
a=np.array([[70,100,40],[120,450,340],[230,230,1230]])
b=np.array([900,1000,3000])
x=np.linalg.solve(a,b)
print(x)


# In[130]:


#q35
a = np.array([[20, 30, 30],
            [20, 80, 120],
            [40, 90, 360]])
print(np.linalg.inv(a))


# In[131]:


#q36
a_2 = a.transpose()
b = np.array([[10],
             [20],
             [30]])
v1 = np.linalg.inv(a_2.dot(a))
v2 = a_2.dot(b)
print(v1.dot(v2))


# In[132]:


#q37
import numpy as np
from matplotlib import pyplot as plt
def f(x):
    return math.exp(x) / math.factorial(x)

x = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
y = []
for i in range(len(x)):
    y.append(f(x[i]))
    
plt.plot(x,y)
plt.show()
print(y)


# In[133]:


#q38
def g(x):
    out = 0
    if x < 0:
        out = 2 * (x**2) + math.exp(x) + 3
    elif x >= 0 and x < 10:
        out = 9 * x + math.log(20)
    else:
        out = 7 * (x**2) + 5 * x - 17
    return out

x = [i for i in range(-10, 20)]
y = []
for i in range(len(x)):
    y.append(g(x[i]))

plt.plot(x, y)
plt.show()

        


# In[134]:


#q39
def circ_area(r):
    return math.pi * r**2

for i in range(10, 20):
    print("Area of circle with radius " + str(i))
    print(round(circ_area(i),2))


# In[135]:


#q40
def h(x):
    return 1/(math.log(x))

sum = 0
for i in range(2, 10000):
    sum += h(i)

print(sum)


# In[136]:


#q41
sum = 0
for i in range(30):
    for j in range(10):
        sum += (i ** 10)/ (3 + j)
print(sum)


# In[137]:


#q42
from scipy.integrate import quad
def f(x):
    return (x ** 15) * (math.exp(-(40 * x)))

res, err = quad(f, 0, math.inf)
print(res, err)


# In[138]:


#q43
def g(x):
    return (x ** 150) * (math.pow(1-x,30))

res,err = quad(g, 0 ,1)
print(res, err)


# In[139]:


#q44
def fun(x, y, z):
    return (math.exp(x) - math.log(z**2))/(5+y)

x = 1
y = 2
z = 3
for i in range(5):
    print(fun(x,y,z))
    x += 1
    y += 1
    z += 1


# In[140]:


#q45
import sympy as sym
solution = sym.solve('x**2 - 33 * x + 1','x')
solution[0]


# In[141]:


#q47
def interest(p, t, r):
    return p * ((1 + r) ** t)

print(interest(40, 50, 0.1))


# In[142]:


#q48
import statsmodels.api as sm
model = sm.OLS(d3.MTFBI, d3.AOR).fit()
predictions = model.predict(d3.AOR)
model.summary()


# In[143]:


#q49
import scipy
scipy.stats.pearsonr(d3.AOR, d3.MTFBI)


# In[144]:


#q50
def chi_sq_test_for_variance(variable, h0):
    sample_variance = variable.var()
    n = variable.notnull().sum()
    degrees_of_freedom = n - 1
    x_sq_stat = (n-1) * sample_variance / h0
    p = stats.chi2.cdf(x_sq_stat, degrees_of_freedom)
    
    if p > 0.05:
        p = 1 - p
    return (x_sq_stat,p,degrees_of_freedom)

aor_variance = round(d3["AOR"].var(),2)
x_sq_stat,pval,dof = chi_sq_test_for_variance(d3["AOR"], h0 = 10)
print(round(x_sq_stat, 2),pval,dof)


# In[ ]:





# In[ ]:




