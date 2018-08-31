# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 13:08:09 2018

@author: tlgreiner
"""

'''
Ex 1,2,3 in FYS-STK4155

'''


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model



def least_square(x,y,basis=0):
    
    # exctract the shape of the model
    shape   = x.shape
    n       = shape[0]
        
    # include bias within the data
    x0      = np.ones([n,1])
        
    # basis expansion on x, with basis = 0 for straight line fit
        
    X       = np.zeros([n,basis+2])
    X [:,0] = x0[:,0]
    
    
    for i in range(basis+1):
        X[:,(i+1)] = x[:,0]**(i+1)
        
    Xt   = np.transpose(X)
    
    Hat  = np.dot(np.linalg.inv(np.dot(Xt,X)),Xt)
    beta = np.dot(Hat,y)
    y_   = np.dot(X,beta)
    
    return y_


# Example using least squares estimate
# Define the training data set (x,y)
x = np.random.rand(100,1)
x.sort(axis=0)
y = 5*x*x + 0.1*np.random.randn(100,1)

shape = x.shape
n = shape[0]

# plot the data
plt.scatter(x,y)

# Self-written least squares estimate
y_0 = least_square(x,y,0) # straight line
y_1 = least_square(x,y,1) # second order poly
y_2 = least_square(x,y,2) # third order poly


plt.plot(x, y, 'g^', x, y_0,'r', x, y_1,'k', x, y_2, 'b', linewidth=3.0)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least squares model estimation')
plt.legend(('y', 'Linear fit', 'poly2 fit', 'poly3 fit'),loc='upper right')
plt.show()


# Scikit-learn least squares estimate
reg     = linear_model.LinearRegression(fit_intercept=True)
x0      = np.ones([n,1])
X       = np.zeros([n,3])
X[:,0]  = x0[:,0]
X[:,1]  = x[:,0]
X[:,2]  = x[:,0]**2
reg.fit(X,y)
y_sci = reg.predict(X)


plt.plot(x, y, 'g^', x, y_sci,'r', x, y_0,'b--', linewidth=3.0)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least squares model estimation')
plt.legend(('y', 'Linear fit sci','Linear fit self'),loc='upper right')
plt.show()




