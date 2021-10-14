import numpy as np
import matplotlib.pyplot as plt
import math

from numpy.core.numeric import ones

def sl(x,y,z,shape,l):#likelyhood funtion
    sum = 0.0
    for i in range(0,shape):
        sum+=math.log2(h(x[i],z))+math.log2(1-h(y[i],z))
    return sum
def h(x,z):
    return (1/(1+pow(math.e,-(z.T@x))))
#dataset:z the parameter;f dataset;l regularization;n the dimension of zeta
def gd_with_r(x,y,z,l,n,times,shape):
    a = 0.01 #learning rate at beginning
    temp = np.zeros(n+1)

    for k in range(1,times+1): #batch for k times
        for i in range(0,shape):
            for j in range(0,n+1):
                temp[j] = a*(1-h(x[i],z))*x[i][j]+l*z[j]
            for j in range(0,n+1):
                temp[j] += a*(0-h(y[i],z))*y[i][j]+l*z[j]
            z = temp+z
        
        if k%100 == 0:
            print(z)
            print("like:",sl(x,y,z,shape,l))
    #g is loss and g is gradient.    
    return z
n = 2
mean1 = np.ones(n)
mean0 = np.ones(n)*2
cov = np.identity(n)/20
shape = 100
l = 0
#form:[1,x1,...,xn,tag]
x = np.c_[np.ones(shape),np.random.multivariate_normal(mean1, cov, (shape,), 'raise')]   # nx2

y = np.c_[np.ones(shape),np.random.multivariate_normal(mean0, cov, (shape,), 'raise')]

z = np.zeros(n+1)
z = gd_with_r(x,y,z,l,n,200,shape)
p = [[0,-z[0]/z[1]],[-z[0]/z[2],0]]
plt.plot(p[0], p[1], color='r')
plt.scatter(x[:,1],x[:,2])
plt.scatter(y[:,1],y[:,2])
print(z)
plt.show()
