# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:17:38 2018

@author: tlgreiner
"""

'''
This modelling work, which comprise phase determination of the 2D Ising model, 
follows the work done by Pankaj Mehta, Marin Bukov, Ching-Hao Wang, Alexandre Day, 
Clint Richardson, Charles Fisher, David Schwab in Notebook 6:

    https://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB_CVII-logreg_ising.html

'''

import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from NN_functions import sigmoid, gradientDescent, predict, predict_nn, logistic_regression_batch
from NeuralNetwork import neural_net_batch, initialize_params, forwardProp, costGrad, gradientDescentOptimizer
from tools import R2_metric, MSE_metric,bootstrap_resampling_regression
import numpy as np
import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

np.random.seed() # shuffle random seed generator

# Ising model parameters
L   = 40 # Number of spins in linear system size
J   = -1.0 # Ising interaction strength
T   = np.linspace(0.25,4.0,16) # set of temperatures (including critical temperatures)
T_c = 2.26 # Onsager critical temperature in the TD limit

###### define ML parameters
num_classes         = 2
train_to_test_ratio = 0.5 # training samples

# Loading the 2D Ising model data : downloaded from https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/
data = pickle.load(open('Ising2DFM_reSample_L40_T=All.pkl','rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)


labels = pickle.load(open('Ising2DFM_reSample_L40_T=All_labels.pkl','rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

#X_critical=data[70000:100000,:]
#Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

del data,labels

# define training and test data sets
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

# pick random data points from ordered and disordered states 
# to create the training and test sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio)

# full data set
#X=np.concatenate((X_critical,X))
#Y=np.concatenate((Y_critical,Y))


Y_train = Y_train.reshape(Y_train.shape[0],1)

m = X_train.shape[0]
n = X_train.shape[1]

# Testing neural network=================================================================

# Extract a smaller sample to reduce computational time
x   = X_train[0:20,:].reshape(20,n).T
y   = Y_train[0:20,:].reshape(20,1).T


# Define hyperparameters
inputLayerSize   = 1600
outputLayerSize  = 1
hiddenLayerSize1 = 50
hiddenLayerSize2 = 50


# Testing Sigmoid===========================================================================================================================

W1,W2,W3,bias1,bias2,bias3 = initialize_params(inputLayerSize,hiddenLayerSize1,hiddenLayerSize2,outputLayerSize)

for i in range(100000):
    
    # Testing gradient optimization
    dCdW3, dCdW2, dCdW1, dCdb3, dCdb2, dCdb1 = costGrad(x,y,W1,W2,W3,bias1,bias2,bias3,activation='sigmoid')
    
    
    params = [W1, W2, W3, 
              bias1, bias2, bias3,
              ]

    grads = [dCdW1, dCdW2, dCdW3,
             dCdb1, dCdb2, dCdb3, 
             ]

    W1,W2,W3,bias1,bias2,bias3 = gradientDescentOptimizer(x,params, grads,eta=0.01,lmbda=0.01)
        
y_pred_sigmoid,z4,z3,z2,a3,a2,a1 = forwardProp(x,W1,W2,W3,bias1,bias2,bias3)

# Testing tanh===========================================================================================================================

W1,W2,W3,bias1,bias2,bias3 = initialize_params(inputLayerSize,hiddenLayerSize1,hiddenLayerSize2,outputLayerSize)

for i in range(100000):
    
    # Testing gradient optimization
    dCdW3, dCdW2, dCdW1, dCdb3, dCdb2, dCdb1 = costGrad(x,y,W1,W2,W3,bias1,bias2,bias3,activation='tanh')
    
    
    params = [W1, W2, W3, 
              bias1, bias2, bias3,
              ]

    grads = [dCdW1, dCdW2, dCdW3,
             dCdb1, dCdb2, dCdb3, 
             ]

    W1,W2,W3,bias1,bias2,bias3 = gradientDescentOptimizer(x,params, grads,eta=0.01,lmbda=0.01)
        
y_pred_tanh,z4,z3,z2,a3,a2,a1 = forwardProp(x,W1,W2,W3,bias1,bias2,bias3)

# Testing Relu===========================================================================================================================

W1,W2,W3,bias1,bias2,bias3 = initialize_params(inputLayerSize,hiddenLayerSize1,hiddenLayerSize2,outputLayerSize)


for i in range(100000):
    
    # Testing gradient optimization
    dCdW3, dCdW2, dCdW1, dCdb3, dCdb2, dCdb1 = costGrad(x,y,W1,W2,W3,bias1,bias2,bias3,activation='Relu')
    
    
    params = [W1, W2, W3, 
              bias1, bias2, bias3,
              ]

    grads = [dCdW1, dCdW2, dCdW3,
             dCdb1, dCdb2, dCdb3, 
             ]

    W1,W2,W3,bias1,bias2,bias3 = gradientDescentOptimizer(x,params, grads,eta=0.01,lmbda=0.01)
        
y_pred_relu,z4,z3,z2,a3,a2,a1 = forwardProp(x,W1,W2,W3,bias1,bias2,bias3)

# Plot results===========================================================================================================================

plt.plot(np.arange(20).T, y.T, 'Xr', np.arange(20).T, y_pred_sigmoid.T, 'ob', np.arange(20).T,y_pred_tanh.T, 'Dk', np.arange(20).T, y_pred_relu.T, 'py', lw=2)
plt.legend(('Truth','Sigmoid','Tanh','Relu'))
plt.xlabel('$\\mathrm{y^i}$')
plt.grid()


plt.savefig('Results_Neural_Net_Classification/Classification_neural_net_test.png', dpi=600,bbox_inches = 'tight')

plt.show()
