# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 13:13:22 2018

@author: tlgreiner
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt
from NeuralNetwork import bootstrap_resampling_neural_net_regressor
from sklearn.model_selection import train_test_split


#===================================================================================================
# Model assesment using bootstrap

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

lmbdas = np.logspace(-5, 2, 8)

n_samples = 400

Data_vector    = Data[1][:n_samples]
Design_matrix  = Data[0][:n_samples]


x_train, x_test, y_train, y_test = train_test_split(Design_matrix, Data_vector, test_size=0.2)
y_train = y_train.reshape(y_train.shape[0],1) 
y_test  = y_test.reshape(y_test.shape[0],1)

# Sigmoid and relu 
y_train_sig_rel = (y_train/np.max(abs(y_train)) + 1)/2
y_test_sig_rel = (y_test/np.max(abs(y_test)) + 1)/2

#tanh
y_train_tan = y_train/np.max(abs(y_train))
y_test_tan  = y_test/np.max(abs(y_test))



error_l, error_t_l, bias_l, bias_t_l, variance_l, variance_t_l             = bootstrap_resampling_neural_net_regressor(y_train_sig_rel, y_test_sig_rel, x_train, x_test, lmbdas, batch=50 ,hidden_layers = (20,20), activation='logistic', n_boostraps = 200)

error_tan, error_t_tan, bias_tan, bias_t_tan, variance_tan, variance_t_tan = bootstrap_resampling_neural_net_regressor(y_train_tan, y_test_tan, x_train, x_test, lmbdas, batch=50 ,hidden_layers = (20,20), activation='tanh', n_boostraps = 200)

error_r, error_t_r, bias_r, bias_t_r, variance_r, variance_t_r             = bootstrap_resampling_neural_net_regressor(y_train_sig_rel, y_test_sig_rel, x_train, x_test, lmbdas, batch=50 ,hidden_layers = (20,20), activation='relu', n_boostraps = 200)



fig, axarr = plt.subplots(nrows=1, ncols=3)

axarr[0].semilogx(lmbdas, error_r, '*--b', lmbdas, error_t_r, '*--r', lmbdas[np.argmin(error_t_r)], np.min(error_t_r), 'x--k', lw=2)
axarr[0].tick_params(labelsize=16)
axarr[0].legend(('Training set: Relu','Test set: Relu'))
axarr[0].set_ylabel('$\\mathrm{Prediction error}$')
axarr[0].set_xlabel('$\\lambda$')
# set title
axarr[0].grid()

axarr[1].semilogx(lmbdas, error_l, '*--b', lmbdas, error_t_l, '*--r', lmbdas[np.argmin(error_t_l)], np.min(error_t_l), 'x--k', lw=2)
axarr[1].tick_params(labelsize=16)
axarr[1].legend(('Training set: Sigmoid','Test set: Sigmoid'))
axarr[1].set_ylabel('$\\mathrm{Prediction error}$')
axarr[1].set_xlabel('$\\lambda$')
axarr[1].grid()

axarr[2].semilogx(lmbdas, error_tan, '*--b', lmbdas, error_t_tan, '*--r', lmbdas[np.argmin(error_t_tan)], np.min(error_t_tan), 'x--k', lw=2)
axarr[2].tick_params(labelsize=16)
axarr[2].legend(('Training set: Tanh','Test set: Tanh'))
axarr[2].set_ylabel('$\\mathrm{Prediction error}$')
axarr[2].set_xlabel('$\\lambda$')
axarr[2].grid()

fig.subplots_adjust(right=2.5)
plt.savefig('Results_Neural_Network/Neural_Net_Regression_Bootstrap_activations.png', dpi=600,bbox_inches = 'tight')

#plt.show()

fig, axarr = plt.subplots(nrows=1, ncols=3)

axarr[0].semilogx(lmbdas, variance_r, '*--b', lmbdas, variance_t_r, '*--g',lmbdas, bias_r, '*--m', lmbdas, bias_t_r, '*--c', lw=2)
axarr[0].tick_params(labelsize=16)
axarr[0].legend(('Training variance','Test variance','Training bias','Test bias'))
axarr[0].set_ylabel('$\\mathrm{Prediction error}$')
axarr[0].set_xlabel('$\\lambda$')
# set title
axarr[0].grid()

axarr[1].semilogx(lmbdas, variance_l, '*--b', lmbdas, variance_t_l, '*--g',lmbdas, bias_l, '*--m', lmbdas, bias_t_l, '*--c', lw=2)
axarr[1].tick_params(labelsize=16)
axarr[1].legend(('Training variance','Test variance','Training bias','Test bias'))
axarr[1].set_ylabel('$\\mathrm{Prediction error}$')
axarr[1].set_xlabel('$\\lambda$')
axarr[1].grid()

axarr[2].semilogx(lmbdas, variance_tan, '*--b', lmbdas, variance_t_tan, '*--g',lmbdas, bias_tan, '*--m', lmbdas, bias_t_tan, '*--c', lw=2)
axarr[2].tick_params(labelsize=16)
axarr[2].legend(('Training variance','Test variance','Training bias','Test bias'))
axarr[2].set_ylabel('$\\mathrm{Prediction error}$')
axarr[2].set_xlabel('$\\lambda$')
axarr[2].grid()

fig.subplots_adjust(right=2.5)
plt.savefig('Results_Neural_Network/Neural_Net_Regression_Bootstrap_activations_var_bias.png', dpi=600,bbox_inches = 'tight')

#plt.show()











