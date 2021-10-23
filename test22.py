import numpy as np
from numpy.core.fromnumeric import argmax
import numpy.matlib
import matplotlib.pyplot as plt
import string
import math
import os
import csv
from numpy.core.numeric import ones
def P(n,x,z,m):
    p = np.zeros((n,m))
    for i in range(0,m):
        p[:,i]=softmax(x[i]@z.T).T.squeeze()
    return p
def Y(n,m,tags):
    #construct y:
    Y = np.zeros((n,m))
    Y[tags,np.arange(m)] = 1
    return Y
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def sl(x,y,z,m,n,tags):
    b = np.zeros((m,1))
    for i in range(0,m):
        b[i,0] = P(n,x,z,m)[tags[i],i]
    #print(b)
    return np.log2(b).sum()

#z:parameter,x:data,l:,m:max_num*n,f:features
def gd(f,n,x,l,m,tags,times):
    a = 1e-3
    y = Y(n,m,tags)
    z = np.zeros((n,f+1))
    temp = np.zeros(z.shape)#gradient
    temp1 = 0
    p = 0
    for k in range(0,times+1):
        temp = a*((y-P(n,x,z,m))@x-l*z)
        temp[0] += a*l*z[0]
        z+=temp
        if k%100 == 0:
            temp1 = sl(x,y,z,m,n,tags)
            if(abs(temp1-p)<1e-6 or np.linalg.norm(temp,ord=1)<1e-6):
                break
            p = temp1
            print("like:",temp1) 

    return z
def clf(x,z):
    return np.argmax(softmax(x@z.T))
def pre(x,y,z):
    #x,y are xs and ys
    Z = np.empty(x.shape)
    for i in range(0,min(x.shape[0],y.shape[0])):
        for j in range(0,min(x.shape[1],y.shape[1])):
            Z[i][j] = clf([1,x[i][j],y[i][j]],z)
    return Z
            

def read_data(max_num,n,tags,f):
    #4: demension of features
    x = np.empty((0,f))
    for i in range(1,n+1):
        x = np.r_[x,np.loadtxt(fname = "./data1/"+str(i)+".csv",max_rows=max_num)]
        tags+=[i-1]*max_num
    x = np.c_[np.ones(n*max_num),x]
    return x

def create(x,max_num,n):
    for i in range(0,n):
        np.savetxt(str(i+1)+".csv", x[max_num*i:max_num*(i+1)], fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)       
def pred_walk():
    max_num = 25
    n = 3
    l = 1e-5
    tags = []
    f = 4
    x = read(n,0,max_num,tags)
    z = gd(f,n,x,l,max_num*n,tags,10000)
    x = read(n,max_num,50,tags)
    count = 0
    for i in range(0,x.shape[0]):
        print(clf(x[i],z),tags[i])
        if(clf(x[i],z)==(tags[i])):
            count = count+1
    print(z)
    print(count)
def str2int(list):
    list1 = []
    for i in range(0,len(list)-1):
        list1.append(float(list[i]))
    return list1
def read(n,begin,max_num,tags):
    count = 0
    temp = np.empty(shape = (0,4))
    for i in range(1,n+1):  
        count1 = 0
        csv_reader = csv.reader(open("./data1/"+str(i)+".csv"))
        for line in csv_reader:
            line = line[0].split(",")
            if count1>=begin:
                temp = np.r_[temp,np.mat(str2int(line))]
                count = count+1
                count1 = count1+1
                tags.append(i-1)
                if count1>=max_num:
                    break
            else:
                count1 = count1+1
    temp = np.c_[np.ones(count),temp]
    return temp
def test():
    max_num = 100
    n = 5
    l = 1e-5
    tags = []
    count = 0
    #
    f= 2
    A = np.random.random(size=(f,f))
    cov = (A@A.T)/2
    x = np.c_[np.random.multivariate_normal((2,2), cov, (max_num,), 'raise')]   # nx2
    y = np.c_[np.random.multivariate_normal((4,6), cov, (max_num,), 'raise')]
    z = np.c_[np.random.multivariate_normal((3,5), cov, (max_num,), 'raise')]
    m = np.c_[np.random.multivariate_normal((3,3), cov, (max_num,), 'raise')]
    s = np.c_[np.random.multivariate_normal((3,2), cov, (max_num,), 'raise')]
    plt.scatter(x[:,0],x[:,1],zorder = 10)
    plt.scatter(y[:,0],y[:,1],zorder = 10)
    plt.scatter(z[:,0],z[:,1],zorder = 10)
    plt.scatter(m[:,0],m[:,1],zorder = 10)
    plt.scatter(s[:,0],s[:,1],zorder = 10)
    x_min, x_max = min(x[:, 0].min(),y[:, 0].min(),z[:, 0].min()) -.5,max(x[:, 0].max(),y[:, 0].max(),z[:, 0].max()) + .5
    y_min, y_max = min(x[:, 1].min(),y[:, 1].min(),z[:, 1].min()) -.5,max(x[:, 1].max(),y[:, 1].max(),z[:, 1].max()) + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x = np.r_[x,y,z,m,s]
    create(x,max_num,n)
   
    
    max_num = 50
    x = read_data(max_num,n,tags,f)
    total = max_num*n
    z = gd(f,n,x,l,total,tags,10000)
    tags = []
    '''
    print(tags)
    for i in range(0,100*3):
        if(clf(x[i],z)==(tags[i])):
            count = count+1
    print(count)
    '''
    Z = pre(xx,yy,z)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral,zorder = 1)    
    
    plt.savefig('./123/'+str(1)+'.jpg')


pred_walk()