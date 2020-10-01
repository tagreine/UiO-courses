# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:27:43 2020

@author: tlgreiner
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from Compfys_functions import Functions

N       = [5,10,15,20]
rho_max = 10

for i in N:
    
    rho_min = 0.0
    h = (rho_max-rho_min)/i
    
    lmbd_an = np.zeros(i)
    for j in range(i):
        lmbd_an[j] = 4*(j+1)-1
        
    # quantum matrix
    A = np.zeros([i,i])
    V = np.zeros(i)
    for j in range(i):
        rho    = (j+1*h)
        V[j]   = rho**2
        d      = 2.0/h**2 + V[j]
        e      = -1.0/h**2
        A[j,j] = d
        if j>0:
            A[j,j-1] = e
        if j<i-1:
            A[j,j+1] = e
    
    ############## 
    W = A.copy()
    for j in range(i):
        W[:,j] = W[:,j]/np.linalg.norm(W[:,j])
    
    eigJ,eigJv = Functions.jacobi(A,W,n=i,conv=10**-8)
    eigJ,eigJv = Functions.sort_eig(eigJ,eigJv)  
    eigJ = Functions.sort_eig_val(eigJ)

    eigNP,eigNPv = np.linalg.eig(A)	
    eigNP = Functions.sort_eig_val(eigNP)    
    
    #plt.figure()
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('N={}'.format(i))
    plt.scatter(np.arange(4),lmbd_an[0:4], marker='x')
    plt.scatter(np.arange(4),eigJ[i-4:][::-1],  marker='v')
    plt.scatter(np.arange(4),eigNP[i-4:][::-1], marker='o')
    plt.legend(['Analytical','Jacobi','Numpy'])
    plt.ylabel('Eigenvalue')
    plt.xlabel('$\lambda$')
    plt.grid()
    plt.show()    
    