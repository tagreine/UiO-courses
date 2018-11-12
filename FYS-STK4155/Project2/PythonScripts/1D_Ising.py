# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:00:22 2018

@author: tlgreiner
"""

'''
This modelling work using the Hamiltonian from the classical Ising model for 
learning the coupling constant in linear regression follows the modelling work
done by Pankaj Mehta, Marin Bukov, Ching-Hao Wang, Alexandre Day, 
Clint Richardson, Charles Fisher, David Schwab in Notebook 4:

    https://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB_CVI-linreg_ising.html

'''

import numpy as np
import scipy.sparse as sp
np.random.seed(12)
import warnings
import matplotlib.pyplot as plt
from tools import least_square




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
# consideration. This is done by taking the Einstein summation of the states 
# interactions:
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))

# Now we have a design matrix of size 10000x1600 (1600 is all pairwise (40x40) interactions)
# and from this we will be able to train a 1600x1 coupling constant vector

#==============================================================================

# Defining the training data
Data=[states,energies]


n_samples = 400
# Define the size of train set for training the model
X_train=Data[0][:n_samples]
Y_train=Data[1][:n_samples]
# and test set for fitting the model
X_test=Data[0][n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2]

# Note that the design matrix is singular, in this case we will turn towards
# using OLS and Ridge regression using singular value decomposition


def least_square_ridge_svd(x, y, lamb = 0.0001):
    # Algorithm for solving the least squares solution using ridge regression
    # and singular value decomposition (svd). Setting lamb = 0.0 will give the OLS 
    # solution with svd
            
    # Computing the svd of the design matrix svd(x) = U D V.T
    
    U, D, V = np.linalg.svd(x, full_matrices=False)
    
    # The identity matrix for ridge parameter
    I = np.eye(D.shape[0])
    
    D = D*I
    D2  = np.dot(D,D)
    
    INV = np.linalg.inv(D2 + lamb*I)
    
    # Define hat matrix with svd
    Hat = np.linalg.multi_dot([V.T, INV, D, U.T])
    
    beta = np.dot(Hat,y)
    y_   = np.dot(x,beta)
    
    return y_, beta

y_ridge, beta_ridge = least_square(X_train,Y_train,method='Ridge',lamb=0.001)
beta_ridge_new = beta_ridge[1:]
plt.imshow(beta_ridge_new.reshape(40,40))


y_ridge_svd, beta_ridge_svd = least_square_ridge_svd(X_train,Y_train,lamb=0.0001)

plt.imshow(beta_ridge_svd.reshape(40,40))



