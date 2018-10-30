# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:17:38 2018

@author: tlgreiner
"""

import numpy as np
from NeuralNetwork import initialize_params, forwardProp, costGrad, gradientDescentOptimizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

a = np.arange(400*1600).reshape(400,1600)

y = np.arange(400).reshape(400,1).T
y = y/np.max(y)

a = a.T

# Define hyperparameters
inputLayerSize   = 1600
outputLayerSize  = 1
hiddenLayerSize1 = 20
hiddenLayerSize2 = 20


W1,W2,W3,bias1,bias2,bias3 = initialize_params(inputLayerSize,hiddenLayerSize1,hiddenLayerSize2,outputLayerSize)


# Testing forward propagation

y_pred,z4,z3,z2,a3,a2,a1 = forwardProp(a,W1,W2,W3,bias1,bias2,bias3,activation='sigmoid')

#plt.plot(a,y,'r',a,y_pred,'k')

# Testing backward propagation
dCdW3, dCdW2, dCdW1, dCdb3, dCdb2, dCdb1 = costGrad(a,y,W1,W2,W3,bias1,bias2,bias3,activation='sigmoid')

for i in range(100000):
    
    # Testing gradient optimization
    dCdW3, dCdW2, dCdW1, dCdb3, dCdb2, dCdb1 = costGrad(a,y,W1,W2,W3,bias1,bias2,bias3,activation='sigmoid')

    params = [W1, W2, W3, 
              bias1, bias2, bias3,
              ]

    grads = [dCdW1, dCdW2, dCdW3,
             dCdb1, dCdb2, dCdb3, 
             ]

    W1,W2,W3,bias1,bias2,bias3 = gradientDescentOptimizer(a,params, grads,eta=0.01,lmbda=0.01)
        
y_pred,z4,z3,z2,a3,a2,a1 = forwardProp(a,W1,W2,W3,bias1,bias2,bias3)
plt.plot(a,y,'r',a,y_pred,'k')


