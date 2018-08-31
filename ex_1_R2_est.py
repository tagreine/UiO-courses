# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:24:03 2018

@author: tlgreiner
"""

'''
Ex 1,2,3 in FYS-STK4155

'''

import matplotlib.pyplot as plt
import numpy as np
from ex_1_least_sqr import least_square
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error



# R2 score
def R2_metric(y ,y_):
    
    # extract the shape (dimensions) of the model 
    shape = y.shape
    n     = shape[0] 
    
    # compute the mean and store it as a vector
    y_m  = np.mean(y_)
    y_mu = y_m*np.ones([n,1])
    
    A = np.dot(np.transpose((y - y_)),(y-y_))
    B = np.dot(np.transpose((y - y_mu)),(y-y_mu))
    
    # compute the R2 score
    R2 = 1 - A/B
    
    return R2


def MSE_metric(y, y_):
    
    # extract the shape (dimensions) of the model 
    shape = y.shape
    n     = shape[0] 
    
    # compute the MSE score
    Err   = np.dot(np.transpose((y - y_)),(y - y_))/n
    Err   = np.squeeze(Err)
    
    return Err
    



# Example using R2 and MSE
# Define the training data set (x,y)
x = np.random.rand(100,1)
x.sort(axis=0)
y = 5*x*x + 0.5*np.random.randn(100,1)

y_ = least_square(x,y,2)

R2_err = R2_metric(y, y_)
R2_err_sci = r2_score(y, y_)

MSE     = MSE_metric(y, y_)
MSE_sci = mean_squared_error(y, y_)

print('The MSE error is ', MSE) 
print('The MSE error from scikit learn is ', MSE_sci)


plt.plot(x, y, 'r', x, y_, 'b')
plt.show()





