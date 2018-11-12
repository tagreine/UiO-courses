# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 08:49:56 2018

@author: tagreine
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
from NN_functions import sigmoid, gradientDescent, predict, predict_nn, logistic_regression_batch, logistic_reg_cost
from NeuralNetwork import neural_net_batch, initialize_params, forwardProp, costGrad, gradientDescentOptimizer
from tools import R2_metric, MSE_metric
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


# We will train the estimators using stochastic gradient descent

Y_train = Y_train.reshape(Y_train.shape[0],1)
Y_test  = Y_test.reshape(Y_test.shape[0],1)


lmbdas = np.logspace(-5,2,8)

m = X_train.shape[0]
n = X_train.shape[1]

# Extract a smaller sample to reduce computational time
X   = X_train[0:800,:].reshape(800,n).T
Y   = Y_train[0:800,:].reshape(800,1).T
X_t = X_test[0:200,:].reshape(200,n).T
Y_t = Y_test[0:200,:].reshape(200,1).T

# Test transformation for tanh

Y_tan = Y + Y - 1 
Y_t_tan  = Y_t + Y_t - 1 

#

# preallocate data
train_accuracy_SGD = np.zeros(lmbdas.shape,np.float64)
test_accuracy_SGD  = np.zeros(lmbdas.shape,np.float64)
train_R2_SGD       = np.zeros(lmbdas.shape,np.float64)
test_R2_SGD        = np.zeros(lmbdas.shape,np.float64)
train_MSE_SGD      = np.zeros(lmbdas.shape,np.float64)
test_MSE_SGD       = np.zeros(lmbdas.shape,np.float64)

# Testing neural network=================================================================

# Define hyperparameters
inputLayerSize   = X.shape[0]
outputLayerSize  = 1
hiddenLayerSize1 = 20
hiddenLayerSize2 = 20

# =======================================================================================

for i,lm in enumerate(lmbdas):
      
                                                
    cost_SGD, W1, W2, W3, bias1, bias2, bias3 = neural_net_batch( X, Y_tan, method='Classification', hiddenLayerSize1=hiddenLayerSize1,hiddenLayerSize2=hiddenLayerSize2,epochs=1300,batch_size=50,alpha=0.05,lmbda = lm,activation='tanh',intercept ='False')

    Y_train_pred,z4,z3,z2,a3,a2,a1 = forwardProp(X,W1,W2,W3,bias1,bias2,bias3,activation='tanh')
    Y_test_pred,z4,z3,z2,a3,a2,a1  = forwardProp(X_t,W1,W2,W3,bias1,bias2,bias3,activation='tanh')
    
    Y_train_pred_SGD = predict_nn(Y_train_pred.T,0.50).reshape(1,800)
    Y_test_pred_SGD  = predict_nn(Y_test_pred.T,0.50).reshape(1,200)
      
    train_accuracy_SGD[i] = accuracy_score(Y.T,Y_train_pred_SGD.T)
    test_accuracy_SGD[i]  = accuracy_score(Y_t.T,Y_test_pred_SGD.T)
    
    train_R2_SGD[i]       = R2_metric(Y.T,Y_train_pred_SGD.T)
    test_R2_SGD[i]        = R2_metric(Y_t.T,Y_test_pred_SGD.T)

    train_MSE_SGD[i]      = MSE_metric(Y.T,Y_train_pred_SGD.T)
    test_MSE_SGD[i]       = MSE_metric(Y_t.T,Y_test_pred_SGD.T)
  
    print( 'Next regularization parameter' )
    

# Testing gave ok results for 
    
    #Sigmoid: alpha = 1, epochs = 500, batch_size = 50, 20 neurons
    #tanh:    alpha = 1, epochs = 500, batch_size = 50, 20 neurons
    #Relu:    alpha = 0.1, epochs = 500, batch_size = 50, 20 neurons

fig, axarr = plt.subplots(nrows=1, ncols=3)

axarr[0].semilogx(lmbdas,train_accuracy_SGD,'*--b',lmbdas,test_accuracy_SGD,'*--r', lmbdas[np.argmax(test_accuracy_SGD)], np.max(test_accuracy_SGD), 'x--k',lw=2)
axarr[0].legend(('Training set','Test set'))
axarr[0].set_ylabel('$\\mathrm{Accuracy}$')
axarr[0].set_xlabel('$\\lambda$')
axarr[0].grid()

axarr[1].semilogx(lmbdas,train_R2_SGD,'*--y',lmbdas,test_R2_SGD,'*--c',lmbdas[np.argmax(test_R2_SGD)], np.max(test_R2_SGD), 'x--k',lw=2)
axarr[1].legend(('Training set','Test set'))
axarr[1].set_ylabel('$\\mathrm{R2 score}$')
axarr[1].set_xlabel('$\\lambda$')
axarr[1].grid()

axarr[2].semilogx(lmbdas,train_MSE_SGD,'*--g',lmbdas,test_MSE_SGD,'*--m',lmbdas[np.argmin(test_MSE_SGD)], np.min(test_MSE_SGD), 'x--k',lw=2)
axarr[2].legend(('Training set','Test set'))
axarr[2].set_ylabel('$\\mathrm{MSE}$')
axarr[2].set_xlabel('$\\lambda$')
axarr[2].grid()


fig.subplots_adjust(right=2.5)

#plt.savefig('Results_Neural_Net_Classification/Classification_accuracy_R2_MSE_train_test_NN_Relu.png', dpi=600,bbox_inches = 'tight')

plt.show()
