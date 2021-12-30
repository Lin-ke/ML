import numpy as np
import matplotlib.pyplot as plt
import math
import os
from numpy.core.numeric import ones
#mixture guassian model:n model,m dimension,shape:[n1,n2...nn],range
def make_gaussian(n,m,shape,var,r):
    tags = np.empty(shape=(0,1))
    temp = np.empty(shape=(0,m))
    for i in range(0,n):
        A = np.random.random((m,m))
        cov = var*A@A.T
        temp = np.r_[temp,np.random.multivariate_normal(np.random.rand(m)*range, cov, (shape,), 'raise')]
        tag = np.c_[tags,i*np.ones(shape[i])] 
    print(temp)
make_gaussian(2,2,(5,5),0.1,5)
