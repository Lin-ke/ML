import numpy as np
import matplotlib.pyplot as plt
import math
import os
import csv
from numpy.core.numeric import ones
from numpy.lib.function_base import blackman
def str2int(list):
    list1 = []
    for i in range(0,len(list)):
        list1.append(float(list[i]))
    return list1
def sl(x,y,z,shape,l):#likelyhood function
    sum = 0.0
    for i in range(0,shape):
        sum+=math.log2(h(x[i],z))+math.log2(1-h(y[i],z))
    return sum
def h(x,z):
    b = (z@x.T).min()
    m = pow(math.e,b)
    return (m/(m+pow(math.e,-(z.T@x)+b)))
def hessian(x,z):
    #1. 矩阵化计算。2.不使用Hessian矩阵。
    n = z.shape[0]
    H = np.empty((n,n))
    shape = x.shape[0]
    for i in range(0,shape):
        for j in range(0,n):
            for k in range(0,n):
                H[j,k] = x[i,j]*x[i,k]*(h(x[i],z)-1)*h(x[i],z)+y[i,j]*y[i,k]*(h(y[i],z)-1)*h(y[i],z)
    #print(H)
    return H
def gradient(x,y,z,l):
    n = z.shape[0]
    temp = np.zeros(n)
    for i in range(0,shape):
        for j in range(0,n):
            temp[j] = (1-h(x[i],z))*x[i][j]-l*np.sign(z[j])
        for j in range(0,n):
            temp[j] += (0-h(y[i],z))*y[i][j]-l*np.sign(z[j])
    return temp
    
def Newton_method(x,y,z,l,times,shape):
    p = 0
    a =0.01
    for k in range(0,times+1):
        z = z-a*(np.linalg.pinv(hessian(x,z))@(gradient(x,y,z,l)))
        if k%100 == 0:
            temp1 = sl(x,y,z,shape,l)
            print("like:",temp1)
            if(abs(temp1-p)<a/100 ):
                if(abs(temp1)>1e-1):
                    a = a/10
                    if(a<1e-5):
                        break
            p = temp1
            
    return z 

def draw():
    y_min, y_max =x[:,2].min(),y[:, 2].max()
    x_min, x_max =x[:,1].min(),y[:, 1].max()
    p = [[(-z[0]-z[2]*y_min)/z[1],(-z[0]-y_max*z[2])/z[1]],[y_min,y_max]]
    #x-y

    plt.axis([x_min-0.5, x_max+0.5, y_min-0.5, y_max+0.5])
    plt.plot(p[0], p[1], color='r')
    plt.scatter(x[:,1],x[:,2])
    plt.scatter(y[:,1],y[:,2])
    d = 0
    while 1:
        if os.path.isfile(str(d)+'.jpg'):
            d = d+1
        else:
            plt.savefig('./'+str(d)+'.jpg')
            break
    plt.show()

#dataset:z the parameter;f dataset;l regularization;n the dimension of zeta
def gd_with_r(x,y,z,l,n,times,shape):
    a = 0.1 #learning rate at beginning
    temp = np.zeros(n+1)
    p = 0
    for k in range(1,times+1): #batch for k times
        for i in range(0,shape):
            for j in range(0,n+1):
                temp[j] = a*(1-h(x[i],z))*x[i][j]-l*np.sign(z[j]) + a*(0-h(y[i],z))*y[i][j]-l*np.sign(z[j])
            z = temp+z
            '''
        if k%100 == 0:
            temp1 = sl(x,y,z,shape,l)
            if(abs(temp1-p)<1e-10 or np.linalg.norm(temp,ord=1)<1e-6):
                break
            p = temp1
            '''
    #g is loss and g is gradient.    
    return z
def gd_with_l1(x,y,z,l,n,times,shape):
    a = 0.1 #learning rate at beginning
    temp = np.zeros(n+1)
    p = 0
    for k in range(1,times+1): #batch for k times
        for i in range(0,shape):
            for j in range(0,n+1):
                temp[j] = a*(1-h(x[i],z))*x[i][j]-l*np.sign(z[j]) + a*(0-h(y[i],z))*y[i][j]-l*np.sign(z[j])
            z = temp+z
            '''
        if k%100 == 0:
            temp1 = sl(x,y,z,shape,l)
            if(abs(temp1-p)<1e-10 or np.linalg.norm(temp,ord=1)<1e-6):
                break
            p = temp1
            '''
    #g is loss and g is gradient.   
    return z
n = 2
mean1 = np.ones(n)/2
mean0 = np.ones(n)*2
shape = 50

list = []
list1 = []
list2 = []
s = 30
'''
for i in range(0,s):
    l = 1e-5
    A = np.random.random(size=(n,n))
    B = np.random.random(size=(n,n))
    cov = (A@A.T)/5
    cov2 = (B@B.T)/2
#form:[1,x1,...,xn,tag]
    x = np.c_[np.ones(shape),np.random.multivariate_normal(mean1, cov, (shape,), 'raise')]   # nx2
    y = np.c_[np.ones(shape),np.random.multivariate_normal(mean0, cov2, (shape,), 'raise')]
    z = np.zeros(n+1)
    z = gd_with_l1(x,y,z,l,n,1000,shape)
    list.append(sl(x,y,z,shape,l))
    z = np.zeros(n+1)
    z = gd_with_r(x,y,z,l,n,1000,shape)
    list2.append(sl(x,y,z,shape,l))
    l = 0
    z = np.zeros(n+1)
    z = gd_with_r(x,y,z,l,n,1000,shape)
    list1.append(sl(x,y,z,shape,l))
    print("done")
'''
list = [-0.8495791759924363, -3.7295178424326783, -0.88399540282923, -9.170296787703617, -0.21113637549532002, -14.650762351945506, -8.949750272569565, -16.347900944228382, -0.8006175829318126, -0.47229778856212007, -4.373080978878049, -6.515087219127119, -0.5202983710515248, -1.190771092417166, -1.1940710665347143, -21.828778325610884, -8.907749786701169, -0.8071009616097021, -9.828874725641656, -2.149423575040869, -1.3264926331925004, -15.302838457335476, -0.19408327739030343, -7.497074456006026, -1.0510335084808502, -0.2334317804178645, -13.704782476832328, -20.739816985183108, -23.11413680171414, -0.27256844983770956]
list1 =[-0.7680885637451177, -3.5912273488719695, -0.8033393202770482, -9.161104183327268, -0.1729197238301286, -14.640920735285407, -8.93245943985171, -16.330647056482018, -0.7261613054143291, -0.40825099637565365, -4.209956672435191, -6.498905579941202, -0.46056187013536126, -1.0974485593570882, -1.1096079795119684, -21.834369216173307, -8.906625655998322, -0.7279648690190214, -9.75670751142896, -2.0689432289144296, -1.2267429838387272, -15.30269619016054, -0.15899953039783166, -7.490210963345212, -0.9751594118214967, -0.19049593662109013, -13.676574483424162, -20.734953631459092, -23.11469058181052, -0.2233461501252542]  
list2 =[-0.8495791759924363, -3.7295178424326783, -0.88399540282923, -9.170296787703617, -0.21113637549532002, -14.650762351945506, -8.949750272569565, -16.347900944228382, -0.8006175829318126, -0.47229778856212007, -4.373080978878049, -6.515087219127119, -0.5202983710515248, -1.190771092417166, -1.1940710665347143, -21.828778325610884, -8.907749786701169, -0.8071009616097021, -9.828874725641656, -2.149423575040869, -1.3264926331925004, -15.302838457335476, -0.19408327739030343, -7.497074456006026, -1.0510335084808502, -0.2334317804178645, -13.704782476832328, -20.739816985183108, -23.11413680171414, -0.27256844983770956]   
    
plt.scatter(np.arange(0,s),list,zorder = 10,s = 20,color = 'white')
plt.scatter(np.arange(0,s),list1,zorder = 8,s = 40,color = 'red')
plt.scatter(np.arange(0,s),list2,zorder = 20,s = 5)
plt.savefig('./123.jpg')

