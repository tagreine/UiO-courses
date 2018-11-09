# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:00:22 2018

@author: tlgreiner
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
from tools import least_square, least_square_ridge_svd, MSE_metric, R2_metric
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# Note that the design matrix is singular, in this case we will turn towards
# using OLS and Ridge regression using singular value decomposition

# Define penalty parameters we want to compute for
lmbdas = np.logspace(-5, 5, 11)

# Define error for the different solutions on the training set and test set
train_err_ols   = np.zeros([len(lmbdas),1])
test_err_ols    = np.zeros([len(lmbdas),1])
train_r2_ols    = np.zeros([len(lmbdas),1])
test_r2_ols     = np.zeros([len(lmbdas),1])

train_err_ridge = np.zeros([len(lmbdas),1])
test_err_ridge  = np.zeros([len(lmbdas),1])
train_r2_ridge  = np.zeros([len(lmbdas),1])
test_r2_ridge   = np.zeros([len(lmbdas),1])

train_err_lasso = np.zeros([len(lmbdas),1])
test_err_lasso  = np.zeros([len(lmbdas),1])
train_r2_lasso  = np.zeros([len(lmbdas),1])
test_r2_lasso   = np.zeros([len(lmbdas),1])

# Define estimators for the different solutions
beta_ols   = []
beta_ridge = []
beta_lasso = []


i = 0

for lmbda in lmbdas:
     
    #==============================================================================================================
    # Ordinary least squares with svd
    Y_pred_train_ols, beta_ols = least_square_ridge_svd(X_train,Y_train,lamb=0.0) # train model, and store weigths 
    Y_pred_test_ols = X_test.dot(beta_ols) # fit model

    
    # Compute the OLS metrics: MSE and R^2 to assess performance of prediction.
    train_err_ols[i] = MSE_metric(Y_train, Y_pred_train_ols)
    test_err_ols[i]  = MSE_metric(Y_test, Y_pred_test_ols)
    
    train_r2_ols[i] = R2_metric(Y_train, Y_pred_train_ols)
    test_r2_ols[i]  = R2_metric(Y_test, Y_pred_test_ols)

    #==============================================================================================================
    # Ridge regression with svd
    Y_pred_train_ridge, beta_ridge = least_square_ridge_svd(X_train,Y_train,lamb=lmbda) # train model, and store weigths 
    Y_pred_test_ridge = X_test.dot(beta_ridge) # fit model 

    
    # Compute the OLS metrics: MSE and R^2 to assess performance of prediction.
    train_err_ridge[i] = MSE_metric(Y_train, Y_pred_train_ridge)
    test_err_ridge[i]  = MSE_metric(Y_test, Y_pred_test_ridge)
    
    train_r2_ridge[i] = R2_metric(Y_train, Y_pred_train_ridge)
    test_r2_ridge[i]  = R2_metric(Y_test, Y_pred_test_ridge)
    
    
    #==============================================================================================================
    # Lasso regression
    Y_pred_train_lasso, beta_lasso = least_square(X_train,Y_train, method='Lasso',lamb=lmbda,intercept='False') # train model, and store weigths 
    Y_pred_test_lasso  = X_test.dot(beta_lasso) # fit model 
    Y_pred_train_lasso = Y_pred_train_lasso.reshape(Y_train.shape[0],1)
    Y_pred_test_lasso  = Y_pred_test_lasso.reshape(Y_test.shape[0],1)
    
    # Compute the OLS metrics: MSE and R^2 to assess performance of prediction.
    train_err_lasso[i] = MSE_metric(Y_train, Y_pred_train_lasso)
    test_err_lasso[i]  = MSE_metric(Y_test, Y_pred_test_lasso)
    
    train_r2_lasso[i] = R2_metric(Y_train, Y_pred_train_lasso)
    test_r2_lasso[i]  = R2_metric(Y_test, Y_pred_test_lasso)
    
    
    i += 1
    #===============================================================================================================
    # Plotting data
    
    cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

    fig, axarr = plt.subplots(nrows=1, ncols=4)
    
    axarr[0].imshow(coupling_constant,**cmap_args)
    axarr[0].set_title('$\\mathrm{Truth}$',fontsize=16)
    axarr[0].tick_params(labelsize=16)
    
    axarr[1].imshow(beta_ols.reshape(L,L),**cmap_args)
    axarr[1].set_title('$\\mathrm{OLS}$',fontsize=16)
    axarr[1].tick_params(labelsize=16)
    
    axarr[2].imshow(beta_ridge.reshape(L,L),**cmap_args)
    axarr[2].set_title('$\\mathrm{Ridge},\ \\lambda=%.4f$' %(lmbda),fontsize=16)
    axarr[2].tick_params(labelsize=16)
    
    im=axarr[3].imshow(beta_lasso.reshape(L,L),**cmap_args)
    axarr[3].set_title('$\\mathrm{LASSO},\ \\lambda=%.4f$' %(lmbda),fontsize=16)
    axarr[3].tick_params(labelsize=16)
    
    divider = make_axes_locatable(axarr[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=fig.colorbar(im, cax=cax)
    
    cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
    cbar.set_label('$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)
    
    fig.subplots_adjust(right=2.0)
    #plt.savefig('Results/' + 'Ising_Regression' + str(i) + '.png', dpi=600,bbox_inches = 'tight')

    plt.show()



# Plot our performance R^2 score on both the training and test data

plt.semilogx(lmbdas.reshape(len(lmbdas),1), train_r2_ols, 'b',label='Train (OLS)')
plt.semilogx(lmbdas.reshape(len(lmbdas),1), test_r2_ols,'--b',label='Test (OLS)')
plt.semilogx(lmbdas.reshape(len(lmbdas),1), train_r2_ridge,'r',label='Train (Ridge)',linewidth=1)
plt.semilogx(lmbdas.reshape(len(lmbdas),1), test_r2_ridge,'--r',label='Test (Ridge)',linewidth=1)
plt.semilogx(lmbdas.reshape(len(lmbdas),1), train_r2_lasso, 'g',label='Train (LASSO)')
plt.semilogx(lmbdas.reshape(len(lmbdas),1), test_r2_lasso, '--g',label='Test (LASSO)')

fig = plt.gcf()
fig.set_size_inches(10.0, 6.0)

#plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
#           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left',fontsize=16)
plt.ylim([-0.01, 1.01])
plt.xlim([min(lmbdas), max(lmbdas)])
plt.xlabel(r'$\lambda$',fontsize=16)
plt.ylabel(r'$R^2$',fontsize=16)
plt.tick_params(labelsize=16)
#plt.savefig('Results/' + 'Ising_Regression_R2.png', dpi=600,bbox_inches = 'tight')
plt.show()

# Plot our performance MSE on both the training and test data

plt.semilogx(lmbdas.reshape(len(lmbdas),1), train_err_ols, 'b',label='Train (OLS)')
plt.semilogx(lmbdas.reshape(len(lmbdas),1), test_err_ols,'--b',label='Test (OLS)')
plt.semilogx(lmbdas.reshape(len(lmbdas),1), train_err_ridge,'r',label='Train (Ridge)',linewidth=1)
plt.semilogx(lmbdas.reshape(len(lmbdas),1), test_err_ridge,'--r',label='Test (Ridge)',linewidth=1)
plt.semilogx(lmbdas.reshape(len(lmbdas),1), train_err_lasso, 'g',label='Train (LASSO)')
plt.semilogx(lmbdas.reshape(len(lmbdas),1), test_err_lasso, '--g',label='Test (LASSO)')

fig = plt.gcf()
fig.set_size_inches(10.0, 6.0)

#plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
#           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left',fontsize=16)
plt.ylim([-0.01, 20])
plt.xlim([min(lmbdas), max(lmbdas)])
plt.xlabel(r'$\lambda$',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.tick_params(labelsize=16)
#plt.savefig('Results/' + 'Ising_Regression_MSE.png', dpi=600,bbox_inches = 'tight')
plt.show()









