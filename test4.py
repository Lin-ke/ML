import numpy as np
import matplotlib.pyplot as plt
import math
import os
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from numpy.core.defchararray import decode
from numpy.core.numeric import ones
from numpy.lib.npyio import loadtxt, savetxt
from numpy.linalg.linalg import eig, eigvals
def mean0(x):
    meanValue = np.mean(x,axis=0)
    return x-meanValue,meanValue
#x:[x1T//x2T//xnT]
def rotate():
    a = 40                              #旋转矩阵
    b = 30
    c = 60
    rotate = np.array([[np.cos(a)*np.cos(c)-np.cos(b)*np.sin(a)*np.sin(c), -np.cos(b)*np.cos(c)*np.sin(a)-np.cos(a)*np.sin(c), np.sin(a)*np.sin(b)],
                    [np.cos(c)*np.sin(a)+np.cos(a)*np.cos(b)*np.sin(c), np.cos(a)* np.cos(b)*np.cos(c)-np.sin(a)*np.sin(c), -np.cos(a)*np.sin(b)],
                    [np.sin(b)*np.sin(c), np.cos(c)*np.sin(b), np.cos(b)]])
    return rotate
def read(k):
    f = np.zeros((0,2500))
    for i in range(k):
        im = Image.open("C:\\Users\\yungaroo\\Documents\\Tencent Files\\2643690387\\FileRecv\\50x50\\data\\{}_2019.jpg".format(str(i+61854)))
        x = np.array(im)
        f = np.r_[f,x.reshape((1,2500))]
    print(f.shape)
    return f
def contribute(value):
    sum = value.sum()
    temp = np.empty(value.shape)
    for i in range(0,1000):
        temp[i] = value[range(i)].sum()/sum
    print(temp)
    return
#x:num,2500;vec:k,2500 y:num,k
def importA():
    return
def pca(x):#对数据进行变换
    k = 1000
    meanMrix, meanValue = mean0(x)
    x1 = np.cov(meanMrix,rowvar=0)
    eigValue,eigVec = np.linalg.eigh(x1)
    eigValue = np.real(eigValue)
    index = bigK(eigValue,k)
    eigValue = eigValue[index]
    eigVec = eigVec[:,index]
    y = meanMrix@eigVec@eigVec.T+meanValue
    savefig(y,1500)
    importA()
    return y,eigValue,eigVec
def pca2(x):#对数据进行变换
    k = 2
    meanMrix, meanValue = mean0(x)
    x1 = np.cov(meanMrix,rowvar=0)
    eigValue,eigVec = np.linalg.eigh(x1)
    eigValue = np.real(eigValue)
    index = bigK(eigValue,k)
    eigValue = eigValue[index]
    eigVec = eigVec[:,index]
    y = meanMrix@eigVec@eigVec.T+meanValue
    return y
def pca_with(x,k):
    meanMrix, meanValue = mean0(x)
    A = importA()
    A = A[:,range(k)]#取前k个
    y = meanMrix@A@A.T+meanValue
    savefig(y,y.shape[0])
def importA():
    A = np.loadtxt("a.txt")
    return A
def bigK(eigs,k):
    return np.argsort(-eigs)[range(0,k)]
def savefig(temp2,k):
    for i in range(k):
        temp = temp2[i].reshape(50,50)
        im = Image.fromarray(temp)
        im = im.convert('L')
        im.save("C:\\Users\\yungaroo\\Documents\\pythoncodes\\savefig\\{}.jpg".format(str(i)))
def makeData():
    x_low = 0                           #普通测试样本范围
    x_up = 5
    y_low = 0
    y_up = 5

    a = 40                              #旋转矩阵
    b = 30
    c = 60
    rotate = np.array([[np.cos(a)*np.cos(c)-np.cos(b)*np.sin(a)*np.sin(c), -np.cos(b)*np.cos(c)*np.sin(a)-np.cos(a)*np.sin(c), np.sin(a)*np.sin(b)],
                    [np.cos(c)*np.sin(a)+np.cos(a)*np.cos(b)*np.sin(c), np.cos(a)* np.cos(b)*np.cos(c)-np.sin(a)*np.sin(c), -np.cos(a)*np.sin(b)],
                    [np.sin(b)*np.sin(c), np.cos(c)*np.sin(b), np.cos(b)]])


    size = 50                       #普通测试样本数

    x = np.random.uniform(high = x_up, low = x_low, size = (size,))
    y = np.random.uniform(high = y_up, low = y_low, size = (size,))
    z = np.random.uniform(high = y_up/5, low = y_low/5, size = (size,))

    xyz = np.vstack((x, y))
    xyz = np.vstack((xyz, z))
    xyz = xyz.T

    for i in range(size):
        xyz[i,:] = np.dot(rotate, xyz[i,:])

    return xyz
def draw2(f,f1):
    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(f[:,0], f[:,1], f[:,2], c='black',)
    ax.scatter(f1[:,0], f1[:,1], f1[:,2], c='red',)
    
    # 绘制图例
    ax.legend(loc='best')
    
    
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    
    
    # 展示
    plt.show()
f = read(10)
pca_with(f,300)
