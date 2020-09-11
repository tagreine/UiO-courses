# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:11:47 2019

@author: tlgreiner
"""
import matplotlib.pyplot as plt
import numpy as np
from Compfys_functions import Functions
from scipy.linalg import toeplitz


# Testing Jacobi iteration

n = 100

h   = 1/(n+1)
xx  = np.linspace(0,1,n)
    
# The forcing function
fx = 100*np.exp(-10*xx)

# Analytic solution
ux = 1 - (1 - np.exp(-10))*xx - np.exp(-10*xx)

# Tridiagonal matrix setup
a1 = np.array([2,-1])
a2 = np.zeros([n-2])
a  = np.concatenate([a1,a2])
    
# Define the Toeplitz matrix        
A    = toeplitz(a, a)

d = (h**2)*fx.reshape(len(fx),1)

x = np.random.randn(n).reshape(n,1)

v = Functions.Jacobi_it(A,x,d,eps=0.001)    

plt.figure()
plt.plot(xx,ux)
plt.plot(xx,v)
plt.ylabel('u(x)')
plt.xlabel('x')
plt.legend(['Analytic','Jacobi'],loc='upper right')
plt.grid()



