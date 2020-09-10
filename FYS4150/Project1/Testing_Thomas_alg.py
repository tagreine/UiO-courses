# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:05:36 2020

@author: tlgreiner
"""

# Testing Thomas algorithm, Gauss elimination 

import numpy as np
import matplotlib.pyplot as plt
from Compfys_functions import Functions

plt.figure(figsize=(5,10))
plt.subplots_adjust(hspace=.8)
n = [10,100,1000]

for i,j in enumerate(n):
    

    h   = 1/(j+1)
    xx  = np.linspace(0,1,j)
    
    # The forcing function
    fx = 100*np.exp(-10*xx)

    # Analytic solution
    ux = 1 - (1 - np.exp(-10))*xx - np.exp(-10*xx)

    # Thomas algorithm setup
    a = -1*np.ones([j,1])
    b = 2*np.ones([j,1])
    c = -1*np.ones([j,1])

    d = (h**2)*fx.reshape(len(fx),1)
    
    # Thomas algorithm for matrix inverse
    th_ux = Functions.Thomas_algorithm_solution(a,b,c,d)

    plt.subplot(len(n),1,i+1)
    plt.plot(xx,th_ux)
    plt.plot(xx,ux)
    plt.title('n={}'.format(j))
    plt.ylabel('u(x)')
    plt.xlabel('x')
    plt.legend(['Analytic','Thomas Algorithm'],loc='upper right')
    plt.grid()