# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:18:19 2020

@author: tlgreiner
"""

import matplotlib.pyplot as plt
import numpy as np
from Compfys_functions import Functions
import time
from scipy.linalg import toeplitz


# Compute relative error and computation time

plt.figure(figsize=(7,7))
plt.subplots_adjust(hspace=.4,wspace=0.4)
n = [10,10**2,10**3,10**4]
err_th = []
err_lu = []
legend = []
th     = []
lu     = []
for i,j in enumerate(n):
    
    h   = 1/(j+1)
    xx  = np.linspace(0,1,j)
    
    # The forcing function
    fx = 100*np.exp(-10*xx)

    # Analytic/closed form solution
    ux = 1 - (1 - np.exp(-10))*xx - np.exp(-10*xx)

    # Special case setup
    d = (h**2)*fx.reshape(len(fx),1)
    
    time_start_th = time.clock()
    # special case
    th_ux = Functions.special_case(d)
    time_elapsed_th = (time.clock() - time_start_th)
    th.append(time_elapsed_th)
    
    # Tridiagonal matrix setup
    a1 = np.array([2,-1])
    a2 = np.zeros([j-2])
    a  = np.concatenate([a1,a2])
    
    # Define the Toeplitz matrix        
    A    = toeplitz(a, a)
    
    time_start_lu = time.clock()
    # LU inverse
    lu_ux = Functions.LU_decomp_inverse(A,d).T
    time_elapsed_lu = (time.clock() - time_start_lu)
    lu.append(time_elapsed_lu)
    
    err_th.append(np.log10( np.sum(np.abs(th_ux-ux))/np.sum(np.abs(ux)))) 
    err_lu.append(np.log10( np.sum(np.abs(lu_ux-ux))/np.sum(np.abs(ux))))
    
    plt.subplot(2,2,1)
    plt.title('Gaussian elimination')
    plt.scatter(np.log10(h),err_th[i])
    plt.ylabel('Relative error')
    plt.xlabel('log(h)')

    plt.subplot(2,2,2)
    plt.title('LU decomposition')
    plt.scatter(np.log10(h),err_lu[i])
    plt.ylabel('Relative error')
    plt.xlabel('log(h)')
  
    plt.subplot(2,2,3)
    plt.title('Computation time Gauss')
    plt.scatter(np.log10(j),time_elapsed_th*1000)
    plt.ylabel('Time (ms)')
    plt.xlabel('log(n)')
   
    plt.subplot(2,2,4)
    plt.title('Computation time LU')
    plt.scatter(np.log10(j),time_elapsed_lu*1000)
    plt.ylabel('Time (ms)')
    plt.xlabel('log(n)')


plt.grid()
plt.show()
