#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# Array method and atribute:-
#     
# dtype = check data type
# 
# ndim = find array dimension
# 
# shape = check array shape dimention
# 
# reshape = array dimention change
# 
# nditer() = iterate the advance method ( for x in np.nditer(arr) print(x)) any of dimention array
# 
# concatenate() = join the array ( np.concatenate(arr + arr1), axis=1)
# 
# stack() =  join the array ( np.stack(arr + arr1), axis=1)
# 
# hstack()  = join the array to stack along rows ( np.hstack(arr + arr1))
# 
# vstack()  = join the array to stack along columns ( np.vtack(arr + arr1))
# 
# dstack() = join the array to stack along columns ( np.dtack(arr + arr1))
# 
# array_split() = np.array_split(arr, 3)
# 
# where() = seraching ( np.where(arr == 4))
# 
# searchsorted() = (np.searchsorted(arr, 7))
# 
# sort() = (np.sort(arr))

# Array random module:--
#     
#     random.randint(100)
#     random.rand()
#     random.randint(100, size=(5))
#     random.randint(100, size=(3, 5)) 
#     random.rand(3, 5)
#     random.choice([3, 5, 7, 9])
#     random.choice([3, 5, 7, 9], size=(3, 5))
#     random.shuffle(arr) = changing the array index
#     random.permutation(arr))
#     sns.distplot([0, 1, 2, 3, 4, 5],hist=False) plt.show()
#     random.normal(size=(2, 3))
#     random.normal(loc=1, scale=2, size=(2, 3))  
#     random.binomial(n=10, p=0.5, size=10)
#     sns.distplot(random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False) plt.show()
#     random.poisson(lam=2, size=10)
#     sns.distplot(random.poisson(lam=2, size=1000), kde=False) plt.show()
#     
#     sns.distplot(random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')
#     sns.distplot(random.poisson(lam=50, size=1000), hist=False, label='poisson') plt.show()
#     
#     
# Difference Between Normal and Binomial Distribution:-
#     
# The main difference is that normal distribution is continous whereas binomial is discrete, 
# 
# but if there are enough data points it will be quite similar to normal distribution with certain loc and scale.  
# 
# sns.distplot(random.normal(loc=50, scale=5, size=1000), hist=False, label='normal')
# 
# sns.distplot(random.binomial(n=100, p=0.5, size=1000), hist=False, label='binomial')
# 
# plt.show()
# 
# 
# Difference Between Binomial and Poisson Distribution:-
# 
# Binomial distribution only has two possible outcomes, whereas poisson distribution can have unlimited possible outcomes.
# 
# But for very large n and near-zero p binomial distribution is near identical to poisson distribution 
# such that n * p is nearly equal to lam.
# 
# sns.distplot(random.binomial(n=1000, p=0.01, size=1000), hist=False, label='binomial')
# 
# sns.distplot(random.poisson(lam=10, size=1000), hist=False, label='poisson')
# 
# plt.show()
# 
# 
# x = random.uniform(size=(2, 3))
# 
# sns.distplot(random.uniform(size=1000), hist=False)
# 
# x = random.logistic(loc=1, scale=2, size=(2, 3))
# 
# sns.distplot(random.logistic(size=1000), hist=False)
# 
# plt.show()
# 
# Difference Between Logistic and Normal Distribution:-
#     
# Both distributions are near identical, but logistic distribution has more area under the tails,
# 
# meaning it represents more possibility of occurrence of an event further away from mean.
# 
# sns.distplot(random.normal(scale=2, size=1000), hist=False, label='normal')
# 
# sns.distplot(random.logistic(size=1000), hist=False, label='logistic')
# 
# plt.show()
# 
# 
# x = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
# 
# Relation Between Poisson and Exponential Distribution:-
#     
# Poisson distribution deals with number of occurences of an event in a time period 
# 
# whereas exponential distribution deals with the time between these events.
# 
# sns.distplot(random.exponential(size=1000), hist=False)
# 
# plt.show()
# 
# 
# Chi Square Distribution:-
#     
# Chi Square distribution is used as a basis to verify the hypothesis.
# 
# It has two parameters:
# 
# df - (degree of freedom).
# 
# size - The shape of the returned array.
# 
# x = random.chisquare(df=2, size=(2, 3))
# 
# sns.distplot(random.chisquare(df=1, size=1000), hist=False)
# 
# plt.show()
# 
# 
# Rayleigh Distribution:-
#     
# Rayleigh distribution is used in signal processing.
# 
# It has two parameters:
# 
# scale - (standard deviation) decides how flat the distribution will be default 1.0).
# 
# size - The shape of the returned array.
# 
# x = random.rayleigh(scale=2, size=(2, 3))
# 
# sns.distplot(random.rayleigh(size=1000), hist=False)
# 
# plt.show()
# 
# Pareto Distribution:-
#     
# A distribution following Pareto's law i.e. 80-20 distribution (20% factors cause 80% outcome).
# 
# x = random.pareto(a=2, size=(2, 3))
# 
# sns.distplot(random.pareto(a=2, size=1000), kde=False)
# 
# plt.show()
# 
# Zipf distributions are used to sample data based on zipf's law.
# 
# x = random.zipf(a=2, size=(2, 3))
# 
# x = random.zipf(a=2, size=1000)
# 
# sns.distplot(x[x<10], kde=False)
# 
# plt.show()

# In[2]:


from numpy import random


# What are ufuncs:-
#     
#     ufuncs stands for "Universal Functions" and they are NumPy functions that operate on the ndarray object.
#     
# Why use ufuncs:-
#     
#     ufuncs are used to implement vectorization in NumPy which is way faster than iterating over elements.
# 
#     They also provide broadcasting and additional methods like reduce, accumulate etc. that are very helpful for computation.
# 
#     ufuncs also take additional arguments, like:
# 
#     where =  boolean array or condition defining where the operations should take place.
# 
#     dtype = defining the return type of elements.
# 
#     out = output array where the return value should be copied.
# 
# 
# What is Vectorization:-
#     
#     Converting iterative statements into a vector based operation is called vectorization.
# 
#     It is faster as modern CPUs are optimized for such operations.
#     
#     example without unfun method:-
#         
#         x = [1, 2, 3, 4]
#         y = [4, 5, 6, 7]
#         z = []
# 
#         for i, j in zip(x, y):
#           z.append(i + j)
#         print(z)
#         
#         x = [1, 2, 3, 4]
#         y = [4, 5, 6, 7]
#         z = np.add(x, y)
#         
#         
# How To Create Your Own ufunc:-
#     
#     To create your own ufunc, you have to define a function, 
#     
#     like you do with normal functions in Python, then you add it to your NumPy ufunc library with the frompyfunc() method.
# 
#     The frompyfunc() method takes the following arguments:
# 
#     function - the name of the function.
# 
#     inputs - the number of input arguments (arrays).
# 
#     outputs - the number of output arrays.  
#     
#     
#     create:-
#         
#         def myadd(x, y):
#             
#              return x+y
# 
#         myadd = np.frompyfunc(myadd, 2, 1)
# 
#         print(myadd([1, 2, 3, 4], [5, 6, 7, 8]))
#         
#         
# Check if a function is a ufunc:-
#     
#     print(type(np.add))
#     
#     print(type(np.concatenate))
#     
#     print(type(np.blahblah))
#     
#     if type(np.add) == np.ufunc:
#         
#          print('add is ufunc')
#             
#     else:
#         
#         print('add is not ufunc')
#         
#         
# math func :-
#     
#     newarr = np.add(arr1, arr2)
#     
#     newarr = np.sum([arr1, arr2])
#     
#     np.subtract(arr1, arr2)
#     
#     newarr = np.multiply(arr1, arr2)
#     
#     newarr = np.divide(arr1, arr2)
#     
#     newarr = np.power(arr1, arr2)
#     
#     newarr = np.mod(arr1, arr2)
#     
#     newarr = np.remainder(arr1, arr2)
#     
#     newarr = np.divmod(arr1, arr2)
#     
#     newarr = np.absolute(arr)
#     
#     arr = np.trunc([-3.1666, 3.6667])
#     
#     arr = np.fix([-3.1666, 3.6667])
#     
#     arr = np.around(3.1666, 2)
#     
#     arr = np.floor([-3.1666, 3.6667])
#     
#     arr = np.ceil([-3.1666, 3.6667])
#     
#     
# Log method:-
#     
#     arr = np.arange(1, 10)
# 
#     print(np.log2(arr))
#     
#     arr = np.arange(1, 10)
# 
#     print(np.log10(arr))
#     
#     arr = np.arange(1, 10)
# 
#     print(np.log(arr))
#     
#     nplog = np.frompyfunc(log, 2, 1)
# 
#     print(nplog(100, 15))
#     
#     x = np.prod(arr)
#     
#     x = np.prod([arr1, arr2])
#     
#     newarr = np.prod([arr1, arr2], axis=1)
#     
#     newarr = np.cumprod(arr)
#     
#     newarr = np.diff(arr)
#     
#     newarr = np.diff(arr, n=2)
#     
#     num1 = 4
#     
#     num2 = 6
# 
#     x = np.lcm(num1, num2)
#     
#     arr = np.array([3, 6, 9])
#     
#     x = np.lcm.reduce(arr)
#     
#     x = np.gcd(num1, num2)
#     
#     
#     x = np.sin(np.pi/2)
#     
#     
#     arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
# 
#     x = np.sin(arr)
#     
#     arr = np.array([90, 180, 270, 360])
# 
#     x = np.deg2rad(arr)
#     
#     arr = np.array([np.pi/2, np.pi, 1.5*np.pi, 2*np.pi])
# 
#     x = np.rad2deg(arr)
#     
#     x = np.arcsin(1.0)
#     
#     x = np.arcsin(arr)
#     
#     base = 3
#     perp = 4
# 
#     x = np.hypot(base, perp)
#     
#     x = np.sinh(np.pi/2)
#     
#     
#     arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
# 
#     x = np.cosh(arr)
#     
#     x = np.arcsinh(1.0)
#     
#     arr = np.array([0.1, 0.2, 0.5])
# 
#     x = np.arctanh(arr)
#     
#     arr = np.array([1, 1, 1, 2, 3, 4, 5, 5, 6, 7])
# 
#     x = np.unique(arr)
#     
#     
#     arr1 = np.array([1, 2, 3, 4])
#     
#     arr2 = np.array([3, 4, 5, 6])
# 
#     newarr = np.union1d(arr1, arr2)
#     
#     
#     newarr = np.intersect1d(arr1, arr2, assume_unique=True)
#     
#     newarr = np.setdiff1d(set1, set2, assume_unique=True)
#     
#     newarr = np.setxor1d(set1, set2, assume_unique=True)
# 
#     

# In[3]:


arr = np.array([1,2,3,4,5,6])


# In[4]:


arr


# In[5]:


arr1 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])


# In[6]:


arr1


# In[7]:


arr2 = np.array([[[1,2,3],[4,5,6],[7,8,9]]])


# In[8]:


arr2


# # Array slicing:-

# In[9]:


arr[1]


# In[10]:


arr1[1,3]


# In[11]:


arr2[0,1,2]


# In[12]:


arr1[1, -1]


# In[13]:


arr[:3]


# In[14]:


arr1[2, 1:4]


# In[15]:


arr1[0:2, 2]


# In[16]:


arr2[0:2, 1:4]


# In[17]:


for i in arr:
    print(i)


# In[18]:


for x in arr1:
    for y in x:
        print(y)


# In[19]:


for j in arr2:
    for k in j:
        for c in k:
            print(c)


# In[20]:


#advanced searching:- any dimention array iter this method ( nditer() )

for p in np.nditer(arr1):
    print(p)
    
    
    


# In[21]:


for h  in np.nditer(arr2):
    print(h)


# In[22]:


# Searching array and sorting array:- 

'''
where() = seraching ( np.where(arr == 4))

searchsorted() = (np.searchsorted(arr, 7))

sort() = (np.sort(arr))

'''

res = np.where(arr==4)

res


# In[23]:


ob = np.searchsorted(arr,5)
ob


# In[24]:


np.sort(arr)


# In[25]:


for l in np.nditer(arr1 % 2 == 0):
    print(l)


# In[26]:


# arange method

a = np.arange(1,10)
a


# In[27]:


#reshape
a = np.arange(1,17).reshape(4,4) # 2d array 4*4 
a


# In[28]:


# ones and zeros

np.ones((2,3))


# In[29]:


np.zeros((3,4))


# In[30]:


# random

np.random.random((3,4))


# In[31]:


# linespace
np.linspace(-10,10,10) # lower item -10 number of item 10 range of item 10


# In[32]:


# identity

np.identity(3)


# In[33]:


# ndim

arr.ndim


# In[34]:


# shape

arr2.shape


# In[35]:


# size
arr.size


# In[36]:


# itemsize

arr.itemsize


# In[37]:


# dtype

arr.dtype


# In[38]:


# astype change data type

arr2.astype(np.int64)



# In[39]:


# aerthmetics 

arr * 2


# In[40]:


# relation

arr == 6


# In[41]:


# vector
arr3 = np.arange(16,32).reshape(4,4)
b = arr3 + a
b


# In[42]:


# sum, mim, max,prod

np.sum(a)


# In[43]:


np.min(a)


# In[44]:


np.max(a)


# In[45]:


np.prod(a)


# In[46]:


np.max(a,axis=1)


# In[47]:


np.min(a, axis=0)


# In[48]:


# mean, median, std, var

np.mean(a,axis=1)


# In[49]:


# dot product

c = np.arange(0,12).reshape(3,4)
d = np.arange(12,24).reshape(4,3)

np.dot(c,d)


# In[50]:


# log and exponetian

np.exp(c)


# In[51]:


np.log(d)


# In[52]:


# round , floor, ceil

np.round(np.random.random((2,3))*100)


# In[53]:


np.floor(np.random.random((2,3))*100)


# In[54]:


np.ceil(np.random.random((2,3))*100)


# In[55]:


arr


# In[56]:


arr1


# In[57]:


c


# In[58]:


# indexing and slicing

e = np.array([[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]]])
e


# In[59]:


c


# In[60]:


c[0,:]


# In[61]:


c[:,2]


# In[62]:


c[1:,1:3] 


# In[63]:


c[0::2,0::3]


# In[64]:


c[0::2,1::2]


# In[65]:


c[1, ::3]


# In[66]:


c[:2,1:]


# In[67]:


t = np.arange(27).reshape(3,3,3)
t


# In[68]:


t[1]


# In[69]:


t[2]


# In[70]:


t[0::2]


# In[71]:


t[0,1, :]


# In[72]:


t[1,0:,1]


# In[73]:


t[2,1:,1:]


# In[74]:


t[0::2, 0, ::2]


# In[75]:


# transpose row to column or column to row

np.transpose(c) # c.T


# In[76]:


# ravel convert any dimenstion to one array

t.ravel()


# In[77]:


# horizental stack ( zorana )

k = np.arange(12).reshape(3,4)
p = np.arange(12,24).reshape(3,4)




# In[78]:


np.hstack((k,p))


# In[79]:


# vertical stacking

np.vstack((k,p))


# In[80]:


# horizental split

np.hsplit(k,2)


# In[81]:


# vertical split

np.vsplit(k,3)


# # advanced indexing

# In[82]:


# fancy indexing

b


# In[83]:


b[[0,2,3]]


# In[84]:


b[:,[0,2,3]]


# In[85]:


# boolean indexing
a = np.random.randint(1,100,24).reshape(6,4)


# In[86]:


a


# In[87]:


# find the even number
a[a%2==0]


# In[88]:


# fint the greter then 50
a[a > 50]


# In[89]:


# fint the greter then 50 and even

a[(a > 50) & (a % 2 == 0)]


# In[90]:


# find all number not divide in 7

a[a % 7 != 0]


# In[91]:


# broadcasting array to diffrence to shape

# diff shape

a = np.arange(6).reshape(2,3)


# In[92]:


b = np.arange(3).reshape(1,3)


# In[93]:


print(a+b)


# In[94]:


# sigmoid

# categorical cross entropy

def sigmoid(arr):
    return 1/(1 + np.exp(-(arr)))

a = np.arange(10)

sigmoid(a)


# In[95]:


# mean squared error
def mse(actual, predictive):
    
    return np.mean((actual - predictive)**2)

actual = np.random.randint(1,50,25)

predictive = np.random.randint(1,50,25)

mse(actual, predictive)


# In[96]:


# missing value
a = np.array([1,2,3,4,5,np.nan,6,7])
a


# In[97]:


a[~np.isnan(a)]


# # Ploting

# In[99]:


# x, y

x = np.linspace(-10,10,100)

y = x

plt.plot(x,y)


# In[100]:


# y = x^2

x = np.linspace(-10,10,100)

y = x**2

plt.plot(x,y)


# In[101]:


# y = sin(x)

x = np.linspace(-10,10,100)

y = np.sin(x)

plt.plot(x,y)


# In[103]:


# y= xlog(x)
x = np.linspace(-10,10,100)

y = x * np.log(x)

plt.plot(x,y)


# In[107]:


#sort

np.sort(arr1,axis=0)


# In[108]:


# append

arr


# In[109]:


np.append(arr,200)


# In[110]:


# unique

x = np.array([1,2,3,4,5,1,2,3,6,7])

np.unique(x)


# In[111]:


# argmax

np.argmax(arr)


# In[112]:


# cumsum

np.cumsum(arr)


# In[113]:


# cumprod

np.cumprod(arr)


# In[114]:


# percentile

np.percentile(arr,100) # max value


# In[115]:


np.percentile(arr,0) # min value


# In[116]:


np.percentile(arr,50) # 50% mid


# In[117]:


# histogram (frequecy count)

np.histogram(arr, bins=[0,2,4,6,8,10])


# In[119]:


# isin

arr[np.isin(arr,[2,4,7])]


# In[120]:


# flip reverse the item

np.flip(arr)



# In[121]:


# put
np.put(arr,[0,3],[12,16])


# In[122]:


arr


# In[123]:


# delete
np.delete(arr,-1)


# In[124]:


np.delete(arr,[0,4])


# In[126]:


# clip

np.clip(arr,a_min=2,a_max=5)


# In[ ]:




