# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:10:31 2020

@author: tlgreiner
"""

# Check difference in time
    
import numpy as np
import matplotlib.pyplot as plt
from Compfys_functions import Functions
import time


plt.figure(figsize=(5,10))
plt.subplots_adjust(hspace=.8)
n = [10**6]

for i,j in enumerate(n):
    

    h   = 1/(j+1)
    xx  = np.linspace(0,1,j)
    
    # The forcing function
    fx = 100*np.exp(-10*xx)

    # Analytic/closed form solution
    ux = 1 - (1 - np.exp(-10))*xx - np.exp(-10*xx)

    # Gauss algorithm setup
    a = -1*np.ones([j,1])
    b = 2*np.ones([j,1])

    d = (h**2)*fx.reshape(len(fx),1)
    
    time_start = time.clock()
    
    # Gaus elimination for matrix inverse
    #th_ux = Functions.gaus_elim(b,a,d)
    # special case
    th_ux = Functions.special_case(d)
    time_elapsed = (time.clock() - time_start)
    
    plt.subplot(len(n),1,i+1)
    plt.plot(xx,th_ux)
    plt.plot(xx,ux)
    plt.title('n={}'.format(j))
    plt.ylabel('u(x)')
    plt.xlabel('x')
    plt.legend(['Analytic','Gauss elimination'],loc='upper right')
    plt.grid()