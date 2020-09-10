# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:14:17 2020

@author: tlgreiner
"""
import matplotlib.pyplot as plt
import numpy as np
from Compfys_functions import Functions


# Compute relative error

plt.figure(figsize=(5,5))
n = [10,100,10**3,10**4,10**5,10**6,10**7]
err = []
legend = []
for i,j in enumerate(n):
    
    h   = 1/(j+1)
    xx  = np.linspace(0,1,j)
    
    # The forcing function
    fx = 100*np.exp(-10*xx)

    # Analytic/closed form solution
    ux = 1 - (1 - np.exp(-10))*xx - np.exp(-10*xx)
    
    # Special case setup
    d = (h**2)*fx.reshape(len(fx),1)
    
    # special case
    th_ux = Functions.special_case(d)

    err.append(np.log10( np.sum(np.abs(th_ux-ux))/np.sum(np.abs(ux)))) 
    legend.append('n={}'.format(j))
    
    plt.scatter(np.log10(h),err[i])
    plt.ylabel('Relative error')
    plt.xlabel('log(h)')
    plt.grid()

plt.legend(legend,loc='upper left')
plt.show()
