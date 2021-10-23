import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import string
import math
import csv
from numpy.core.numeric import ones
def sl(f,z,n,shape,l):#likelyhood funtion
    sum = 0.0
    print(z)
    for i in range(0,n):
        for j in range(0,shape):

            sum+=math.log2(h(f,z,i,j,n)[i])
    return sum
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def dot(a,b):
    sum = 0
    for m in range(0,5):
        sum+=a[0,m]*b[m,0]
    return sum
def h(f,z,i,j,n):
    list=[]
    for d in range(0,n):
        temp = dot(f[i][j],z[d].T)
        list.append(temp)
    re = softmax(list)
    return re
#由于我已经知道有个样本点了，
def I(tag,z):
    if(z==tag):
        return 1
    else:
        return 0
#dataset:z the parameter;f dataset;l regularization;n the dimension of zeta
#f:[[data1],[data2],...]shape = 100
def gd_with_r(f,z,l,n,times,shape):
    a = 0.1  #learning rate at beginning
    temp = np.matlib.zeros((n,5))
    p = 0
    for k in range(1,times+1): #batch for k times
        for i in range(0,n):#the i-th class
            for j in range(0,shape): #i-th batch's j-th data
                for m in range(0,n):
                    temp[m] += a*(I(i,m)-h(f,z,i,j,n)[m])*f[i][j]-l*np.sign(z[m])
        z = temp+z
        if k%100 == 0:
            temp1 = sl(f,z,n,shape,l)
            print(temp1)
            #if(abs(temp1-p)<1e-6 or np.linalg.norm(temp,ord=1)<1e-6):
            #    break
            p = temp1
            
    #g is loss and g is gradient.    
    return z   
def str2int(list):
    list1 = []
    for i in range(0,len(list)):
        list1.append(float(list[i]))
    return list1
max_num = 10
n = 4 #number of tag
#4: demension of features
f = []
for i in range(1,n+1):
    temp = np.empty(shape = (0,4))
    count = 0
    csv_reader = csv.reader(open("./data1/"+str(i)+".csv"))
    for line in csv_reader:
        line = np.mat(str2int(line))
        temp = np.r_[temp,line]
        count = count+1
        if count==max_num:
            break
    if count!=max_num:
        print(i,"not enough",max_num)
    temp = np.c_[np.ones(count),temp]
    f.append(temp)
l = 1e-5
shape = max_num*n
z = np.matlib.zeros((n,5))
plt.scatter([f[0][:,1]],[f[0][:,2]])
plt.scatter([f[1][:,1]],[f[1][:,2]])
#plt.show()
z = gd_with_r(f,z,l,n,1000,max_num)
