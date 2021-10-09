from matplotlib.markers import MarkerStyle
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.core.defchararray import count
from numpy.core.numeric import identity
from numpy.testing._private.utils import print_assert_equal

#loss function:J(z) = sigma(h(x)-y)^2/N
def sl(f,z,y,shape):
    sum = 0.0
    for i in range(0,shape):
        sum+=pow((y[i]-np.dot(z,f[:,i])),2)
    return sum/shape

def sl_with_r(f,z,y,shape,l):
    l = l/2
    sum = sl(f,z,y,shape)
    for i in range(0,z.shape[0]):
        sum+=l*pow(z[i],2)
    return sum
#normal equation
def ne(f,z,y):
    z = np.dot(np.dot(np.linalg.pinv(np.dot(f,f.T)),f),y)
    return z

def ne_with_r(f,z,y,l):
    z = np.dot(np.dot(np.linalg.pinv(np.dot(f,f.T)+l*identity(z.shape[0])),f),y)
    return z
def sign(z):
    if z > 0:
        return 1
    if z == 0:
        return 0
    else:
        return -1
def gd(f,z,y,shape,n,times,l):
    #gradient descent
    a = 0.01 #learning rate at beginning
    temp = np.zeros(n+1)
    d = 0 #the curracy
    r1 = 0.000001
    r2 = 0.01
    g = 0
    p = 0
    count = 0
    for k in range(1,times+1): #batch for k times
        for i in range(0,shape):
            for j in range(0,n+1):
                temp[j] = a*((y[i]-z@f[:,i])*f[j][i]+l*(z[j]))
            z = temp+z
    #g is loss and g is gradient.    
        if k%100 == 0: #每隔100次调整参数
            temp2 = sl(f,z,y,shape)
            temp3 = np.linalg.norm(temp/a,ord = 1)
            if g!=0 and (temp2-g>0) : #如果学习反向了
                a = a*0.8
                temp2 = g
                z = temp4
                flag = 1
            if p!=0 and (temp3-p>0): #如果梯度反向了
                if p<r2:
                    a = a*0.8
                    z = temp4
                    temp3 = p
                    flag = 1

            if abs(temp2-g)<r1 and abs(temp3-p)<r1 :
                if flag == 0 and (temp2>r2 or temp3 >r2) :
                    a = a*1.25
                else:
                    break
            g = temp2
            p = temp3
            flag = 0
            print("loss",g)
            print("grad:",temp3)
            print("a:",a)
            if int(math.log(g,10))!=d:
                d = int(math.log(g,10))
                a = a*0.5
            count = count+1
            if count==10:
                a = a*1.2
                count = 0
                if a<r1:
                    break
            temp4 = z #保存此轮参数
    return z
def cong_gra(f,y,z,n,l):
    r = f@y
    p = r
    A = f@f.T+l*np.identity(n+1)
    z = 0
    d = 1e-7
    for i in range(0,n):
        a = (r.T@r)/(p.T@A@p)
        z = z+a*p
        temp = r-a*A@p
        b = (temp.T@temp)/(r.T@r)
        p = temp+b*p
        r = temp
        if (p.T@A@p)<d :
            print("now,break")
            
            break  
    return z
def draw(f,z):
    y1 = np.zeros(shape)
    Y1 = np.zeros(256)
    l = np.empty([n+1,256],dtype = float)
    for i in range(0,shape):
        y1[i] = np.dot(z,f[:,i])
    X = np.linspace(0,0.9,256)
    Y = np.sin(2*math.pi*X)
    for i in range(0,256):
        l[0,i] = 1
        for j in range(1,n+1):
            l[j,i] = l[j-1,i]*X[i]
    for i in range(0,256):
        Y1[i] = np.dot(z,l[:,i])
    plt.plot(X,Y,color="red", linewidth=1.0, linestyle="-",label = "sin2πx")
    plt.plot(X,Y1,color="black",linewidth=1.0, linestyle="-")
    plt.scatter(x,y ,s = 10,label = "sin2πx+e")
    plt.scatter(f[1],y1,s=10,label = "P(x)")
    plt.legend()
    plt.savefig('./test2.jpg')
    return
#samples,(x,sin(x)+e,e~N(0,1)&e<=0.1)
shape = 10
x = np.arange(0,1,1/shape)
y = np.sin(2*math.pi*x)
y+=np.random.normal(0,0.1,shape)
#the polynomial,(z.T*x)
n = 50
l = 0
ed = np.zeros(100)
z = np.zeros(n+1)
f = np.empty([n+1,shape],dtype = float)
for i in range(0,shape):
    f[0,i] = 1
    for j in range(1,n+1):
        f[j,i] = f[j-1,i]*x[i]

z = cong_gra(f,y,z,n,l)
draw(f,z)