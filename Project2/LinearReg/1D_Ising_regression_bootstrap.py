# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:36:27 2018

@author: tagreine
"""


import numpy as np
#import scipy.sparse as sp
np.random.seed(12)
import warnings
import matplotlib.pyplot as plt
from tools import bootstrap_resampling_regression
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

lmbdas = np.logspace(-5, 5, 11)

n_samples = 400

Data_vector    = Data[1][:n_samples]
Design_matrix  = Data[0][:n_samples]


x_train, x_test, y_train, y_test = train_test_split(Design_matrix, Data_vector, test_size=0.2)
y_train = y_train.reshape(y_train.shape[0],1) 
y_test  = y_test.reshape(y_test.shape[0],1)

error, bias, variance, error_t, bias_t, variance_t = bootstrap_resampling_regression(y_train, y_test, x_train, x_test, lmbdas, method='Lasso', n_boostraps=200, intercept=True)

plt.subplot(1, 2, 1)
plt.semilogx(lmbdas, error, '*--b', lmbdas, error_t, '*--r', lmbdas[np.argmin(error_t)], np.min(error_t), 'x--k', lw=2)
plt.xlabel(r'$\lambda$')
plt.ylabel('Prediction Error')
plt.legend(('Training set','Test set'))
plt.grid(True)


plt.subplot(1, 2, 2)
plt.semilogx(lmbdas, variance, 'k', lmbdas, variance_t, 'c', lmbdas, bias, 'y', lmbdas, bias_t, 'm', lw=2)
plt.xlabel(r'$\lambda$')
#plt.ylabel('Variance')
plt.legend(('Training variance','Test variance','Training bias','Test bias'))
plt.grid(True)

plt.tight_layout()
plt.savefig('Results_Neural_Net_Classificaion/Prediction_erro_bias_variance_lasso.png', dpi=600)
plt.show()