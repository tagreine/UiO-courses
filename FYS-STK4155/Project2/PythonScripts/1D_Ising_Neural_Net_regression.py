# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:48:46 2018

@author: tagreine
"""

'''
This modelling work, by using the Hamiltonian from the classical Ising model for 
learning the coupling constant using regression methods, follows the work
done by Pankaj Mehta, Marin Bukov, Ching-Hao Wang, Alexandre Day, 
Clint Richardson, Charles Fisher, David Schwab in Notebook 4:

    https://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB_CVI-linreg_ising.html

'''

import numpy as np
#import scipy.sparse as sp
np.random.seed(12)
import warnings
import matplotlib.pyplot as plt
from tools import MSE_metric, R2_metric
from NeuralNetwork import neural_net_batch, forwardProp

#Comment this to turn on warnings
warnings.filterwarnings('ignore')
### define Ising model params
# system size
L=40
# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(10000,L))
def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E, J
# calculate Ising energies
energies, coupling_constant =ising_energies(states,L)


# Reshaping Ising states into: S_iS_j --> X_p, this means we redefine
# the system where all pairwise interactions of every variable are taken into
# consideration. This is done by taking the Einstein summation of the state 
# interactions:
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))

# Now we have a design matrix of size 10000x1600 (1600 is all pairwise (40x40) interactions)
# and from this we will be able to train a 1600x1 coupling constant vector

#==============================================================================
# Now we will train and fit the data produced by the 1D Ising model

# Defining the training data
Data=[states,energies]

n_samples = 400
# Define the size of train set for training the model
X_train=Data[0][:n_samples]
Y_train=Data[1][:n_samples].reshape(n_samples,1)
# and test set for fitting the model
X_test=Data[0][n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2].reshape(3*n_samples//2 - n_samples,1)


#========================================= Training neural network=================================================================

# Need to transpose the input and output bcs of the structure of the written network 

Y_train = Y_train.T
X_train = X_train.T
Y_test  = Y_test.T
X_test  = X_test.T

# Normalize for relu and sigmoid before training

Y_train_sig_rel = (Y_train/np.max(abs(Y_train)) + 1)/2 
Y_test_sig_rel  = (Y_test/np.max(abs(Y_test))   + 1)/2

# Normalize for tanh
Y_train_tan = Y_train/np.max(abs(Y_train))
Y_test_tan  = Y_test/np.max(abs(Y_test))


# Define hyperparameters
inputLayerSize   = X_train.shape[0]
outputLayerSize  = 1
hiddenLayerSize1 = 20
hiddenLayerSize2 = 20

# Define penalty parameters we want to compute for
lmbdas = np.logspace(-5, 5, 11)

# Define error for the different solutions on the training set and test set
train_err_nn_r = np.zeros([len(lmbdas),1])
test_err_nn_r  = np.zeros([len(lmbdas),1])
train_r2_nn_r  = np.zeros([len(lmbdas),1])
test_r2_nn_r   = np.zeros([len(lmbdas),1])

train_err_nn_s = np.zeros([len(lmbdas),1])
test_err_nn_s  = np.zeros([len(lmbdas),1])
train_r2_nn_s  = np.zeros([len(lmbdas),1])
test_r2_nn_s   = np.zeros([len(lmbdas),1])

train_err_nn_t   = np.zeros([len(lmbdas),1])
test_err_nn_t    = np.zeros([len(lmbdas),1])
train_r2_nn_t    = np.zeros([len(lmbdas),1])
test_r2_nn_t     = np.zeros([len(lmbdas),1])


i = 0

for lm in lmbdas:
     
    #===========================================================================================================================================================================================================
    # Train relu
    cost_SGD, W1_r, W2_r, W3_r, bias1_r, bias2_r, bias3_r = neural_net_batch( X_train, Y_train_sig_rel, method='Regression', hiddenLayerSize1=hiddenLayerSize1,hiddenLayerSize2=hiddenLayerSize2, epochs = 5000, batch_size = 50,alpha = 0.5, lmbda = lm, activation='Relu', intercept = 'False')
    Y_train_pred_r,z4,z3,z2,a3,a2,a1 = forwardProp(X_train,W1_r,W2_r,W3_r,bias1_r,bias2_r,bias3_r,activation='Relu')
    Y_test_pred_r,z4,z3,z2,a3,a2,a1  = forwardProp(X_test,W1_r,W2_r,W3_r,bias1_r,bias2_r,bias3_r,activation='Relu')
    # Train sigmoid
    cost_SGD, W1_s, W2_s, W3_s, bias1_s, bias2_s, bias3_s = neural_net_batch( X_train, Y_train_sig_rel, method='Regression', hiddenLayerSize1=hiddenLayerSize1,hiddenLayerSize2=hiddenLayerSize2, epochs = 5000, batch_size = 50,alpha = 10, lmbda = lm, activation='sigmoid', intercept = 'False')
    Y_train_pred_s,z4,z3,z2,a3,a2,a1 = forwardProp(X_train,W1_s,W2_s,W3_s,bias1_s,bias2_s,bias3_s,activation='sigmoid')
    Y_test_pred_s,z4,z3,z2,a3,a2,a1  = forwardProp(X_test,W1_s,W2_s,W3_s,bias1_s,bias2_s,bias3_s,activation='sigmoid')
    # Train tanh
    cost_SGD, W1_t, W2_t, W3_t, bias1_t, bias2_t, bias3_t = neural_net_batch( X_train, Y_train_tan, method='Regression', hiddenLayerSize1=hiddenLayerSize1,hiddenLayerSize2=hiddenLayerSize2, epochs = 5000, batch_size = 50,alpha = 1, lmbda = lm, activation='tanh', intercept = 'False')
    Y_train_pred_t,z4,z3,z2,a3,a2,a1 = forwardProp(X_train,W1_t,W2_t,W3_t,bias1_t,bias2_t,bias3_t,activation='tanh')
    Y_test_pred_t,z4,z3,z2,a3,a2,a1  = forwardProp(X_test,W1_s,W2_s,W3_s,bias1_s,bias2_s,bias3_s,activation='tanh')
    
    
    # Compute the metrics: MSE and R^2 to assess performance of prediction
    #Relu
    train_err_nn_r[i] = MSE_metric(Y_train_sig_rel.T, Y_train_pred_r.T)
    test_err_nn_r[i]  = MSE_metric(Y_test_sig_rel.T, Y_test_pred_r.T)  
    train_r2_nn_r[i]  = R2_metric(Y_train_sig_rel.T, Y_train_pred_r.T)
    test_r2_nn_r[i]   = R2_metric(Y_test_sig_rel.T, Y_test_pred_r.T)
    # Sigmoid
    train_err_nn_s[i] = MSE_metric(Y_train_sig_rel.T, Y_train_pred_s.T)
    test_err_nn_s[i]  = MSE_metric(Y_test_sig_rel.T, Y_test_pred_s.T)  
    train_r2_nn_s[i]  = R2_metric(Y_train_sig_rel.T, Y_train_pred_s.T)
    test_r2_nn_s[i]   = R2_metric(Y_test_sig_rel.T, Y_test_pred_s.T)
    # Tanh
    train_err_nn_t[i] = MSE_metric(Y_train_tan.T, Y_train_pred_t.T)
    test_err_nn_t[i]  = MSE_metric(Y_test_tan.T, Y_test_pred_t.T)  
    train_r2_nn_t[i]  = R2_metric(Y_train_tan.T, Y_train_pred_t.T)
    test_r2_nn_t[i]   = R2_metric(Y_test_tan.T, Y_test_pred_t.T)

    i += 1
    
    print( 'Next regularization parameter' )


fig, axarr = plt.subplots(nrows=1, ncols=3)

axarr[0].semilogx(lmbdas,train_err_nn_r,'*--b',lmbdas,test_err_nn_r,'*--r', lmbdas[np.argmin(test_err_nn_r[0:8])], np.min(test_err_nn_r[0:8]), 'x--k',lw=2)
axarr[0].legend(('Training set: Relu','Test set: Relu'))
axarr[0].set_ylabel('$\\mathrm{Prediction error}$')
axarr[0].set_xlabel('$\\lambda$')
axarr[0].grid()

axarr[1].semilogx(lmbdas,train_err_nn_s,'*--b',lmbdas,test_err_nn_s,'*--r', lmbdas[np.argmin(test_err_nn_s[0:8])], np.min(test_err_nn_s[0:8]), 'x--k',lw=2)
axarr[1].legend(('Training set: Sigmoid','Test set: Sigmoid'))
axarr[1].set_ylabel('$\\mathrm{Prediction error}$')
axarr[1].set_xlabel('$\\lambda$')
axarr[1].grid()

axarr[2].semilogx(lmbdas,train_err_nn_t,'*--b',lmbdas,test_err_nn_t,'*--r', lmbdas[np.argmin(test_err_nn_t[0:8])], np.min(test_err_nn_t[0:8]), 'x--k',lw=2)
axarr[2].legend(('Training set: Tanh','Test set: Tanh'))
axarr[2].set_ylabel('$\\mathrm{Prediction error}$')
axarr[2].set_xlabel('$\\lambda$')
axarr[2].grid()

fig.subplots_adjust(right=2.5)

plt.savefig('Results_Regression/Neural_Network_regression_MSE_train_test.png', dpi=600,bbox_inches = 'tight')

plt.show()


fig, axarr = plt.subplots(nrows=1, ncols=3)

axarr[0].semilogx(lmbdas,train_r2_nn_r,'*--b',lmbdas,test_r2_nn_r,'*--r', lmbdas[np.argmax(test_r2_nn_r[0:8])], np.max(test_r2_nn_r[0:8]), 'x--k',lw=2)
axarr[0].legend(('Training set: Relu','Test set: Relu'))
axarr[0].set_ylabel('$\\mathrm{R^2}$')
axarr[0].set_xlabel('$\\lambda$')
axarr[0].grid()

axarr[1].semilogx(lmbdas,train_r2_nn_s,'*--b',lmbdas,test_r2_nn_s,'*--r', lmbdas[np.argmax(test_r2_nn_s[0:8])], np.max(test_r2_nn_s[0:8]), 'x--k',lw=2)
axarr[1].legend(('Training set: Sigmoid','Test set: Sigmoid'))
axarr[1].set_ylabel('$\\mathrm{R^2}$')
axarr[1].set_xlabel('$\\lambda$')
axarr[1].grid()

axarr[2].semilogx(lmbdas,train_r2_nn_t,'*--b',lmbdas,test_r2_nn_t,'*--r', lmbdas[np.argmax(test_r2_nn_t[0:8])], np.max(test_r2_nn_t[0:8]), 'x--k',lw=2)
axarr[2].legend(('Training set: Tanh','Test set: Tanh'))
axarr[2].set_ylabel('$\\mathrm{R^2}$')
axarr[2].set_xlabel('$\\lambda$')
axarr[2].grid()

fig.subplots_adjust(right=2.5)

#plt.savefig('Results_Regression/Neural_Network_regression_R2_train_test.png', dpi=600,bbox_inches = 'tight')

plt.show()

# Testing best fit for relu

cost_SGD, W1, W2, W3, bias1, bias2, bias3 = neural_net_batch( X_train, Y_train_sig_rel, method='Regression', hiddenLayerSize1=hiddenLayerSize1,hiddenLayerSize2=hiddenLayerSize2, epochs = 2000, batch_size = 50,alpha = 0.5, lmbda = 0.00001, activation='Relu', intercept = 'False')
Y_train_pred,z4,z3,z2,a3,a2,a1 = forwardProp(X_train,W1,W2,W3,bias1,bias2,bias3,activation='Relu')
Y_test_pred,z4,z3,z2,a3,a2,a1  = forwardProp(X_test,W1,W2,W3,bias1,bias2,bias3,activation='Relu')

xrange =np.arange(400)
x1 = xrange[0:50].reshape(1,50)
y1 = Y_train_sig_rel[0,0:50].reshape(1,50)
y1p= Y_train_pred[0,0:50].reshape(1,50)

x2 = xrange[350:400].reshape(1,50)
y2 = Y_train_sig_rel[0,350:400].reshape(1,50)
y2p= Y_train_pred[0,350:400].reshape(1,50)


x1t = xrange[0:50].reshape(1,50)
y1t = Y_test_sig_rel[0,0:50].reshape(1,50)
y1tp= Y_test_pred[0,0:50].reshape(1,50)

x2t = xrange[100:150].reshape(1,50)
y2t = Y_test_sig_rel[0,100:150].reshape(1,50)
y2tp = Y_test_pred[0,100:150].reshape(1,50)

fig, axarr = plt.subplots(nrows=1, ncols=3)

axarr[0].plot(xrange,Y_train_sig_rel.T, '*--r',xrange,Y_train_pred.T, '*--k', lw=1)
axarr[0].legend(('Truth: Relu','Predicted train: Relu'))
axarr[0].set_ylabel('$\\mathrm{E[s]}$')
#axarr[0].set_xlabel('$\\$')
axarr[0].grid()

axarr[1].plot(x1.T,y1.T, '*--r',x1.T,y1p.T, 'k',lw=1)
axarr[1].legend(('Truth: Relu','Predicted train: Relu'))
axarr[1].set_ylabel('$\\mathrm{E[s]}$')
#axarr[1].set_xlabel('$\\$')
axarr[1].grid()

axarr[2].plot(x2.T,y2.T, '*--r',x2.T,y2p.T, 'k',lw=1)
axarr[2].legend(('Truth: Relu','Predicted train: Relu'))
axarr[2].set_ylabel('$\\mathrm{E[s]}$')
#axarr[2].set_xlabel('$\\$')
axarr[2].grid()

fig.subplots_adjust(right=2.5)

plt.savefig('Results_Regression/Neural_Network_train_fit_2kepoch_10.png', dpi=600,bbox_inches = 'tight')

#plt.show()



fig, axarr = plt.subplots(nrows=1, ncols=3)

axarr[0].plot(xrange[0:200].reshape(200,1),Y_test_sig_rel.T, '*--b',xrange[0:200].reshape(200,1),Y_test_pred.T, '*--g', lw=1)
axarr[0].legend(('Truth: Relu','Predicted test: Relu'))
axarr[0].set_ylabel('$\\mathrm{E[s]}$')
#axarr[0].set_xlabel('$\\$')
axarr[0].grid()

axarr[1].plot(x1t.T,y1t.T, '*--b',x1t.T,y1tp.T, 'g',lw=1)
axarr[1].legend(('Truth: Relu','Predicted test: Relu'))
axarr[1].set_ylabel('$\\mathrm{E[s]}$')
#axarr[1].set_xlabel('$\\$')
axarr[1].grid()

axarr[2].plot(x2t.T,y2t.T, '*--b',x2t.T,y2tp.T, 'g',lw=1)
axarr[2].legend(('Truth: Relu','Predicted test: Relu'))
axarr[2].set_ylabel('$\\mathrm{E[s]}$')
#axarr[2].set_xlabel('$\\$')
axarr[2].grid()

fig.subplots_adjust(right=2.5)

plt.savefig('Results_Regression/Neural_Network_test_fit_2kepoch_10.png', dpi=600,bbox_inches = 'tight')

#plt.show()

mse = MSE_metric(Y_test_sig_rel.T,Y_test_pred.T)