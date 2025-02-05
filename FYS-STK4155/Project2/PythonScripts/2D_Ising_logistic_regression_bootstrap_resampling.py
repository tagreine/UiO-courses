# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:25:17 2018

@author: tagreine
"""

'''
This modelling work, which comprise phase determination of the 2D Ising model, and 
follows the work done by Pankaj Mehta, Marin Bukov, Ching-Hao Wang, Alexandre Day, 
Clint Richardson, Charles Fisher, David Schwab in Notebook 6:

    https://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB_CVII-logreg_ising.html

'''

import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from tools import bootstrap_resampling_logistic
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



# Loading the 2D Ising model data : downloaded from https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/
data = pickle.load(open('Ising2DFM_reSample_L40_T=All.pkl','rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data = data.astype('int')
data[np.where(data==0)] = -1 # map 0 state to -1 (Ising variable can take values +/-1)


labels = pickle.load(open('Ising2DFM_reSample_L40_T=All_labels.pkl','rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

# Divide data into ordered, critical and disordered
# Training data is structured as follows
#     
#       
#   X = [ x1.T, x2.T, ... , xm.T]
#
#   where x1, ... , xm = column vector of size (LxL)x1, containing LxL images  
#
#   y = [y1.T, y2.T, ... , yn.T,]
#
#   where y1, ... , yn = scalars, containing binary classes, i.e. 0 or 1 

X_ordered = data[:70000,:]
Y_ordered = labels[:70000]

#X_critical=data[70000:100000,:]
#Y_critical=labels[70000:100000]

X_disordered = data[100000:,:]
Y_disordered = labels[100000:]

del data,labels

# define training and test data sets
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

# pick random data points from ordered and disordered states 
# to create the training and test sets

###### define ML parameters
train_to_test_ratio           = 0.8 # training samples
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=train_to_test_ratio)



############################### Model assessment ########################################


# We will train the estimators and assess the model using stochastic gradient descent and bootstrap

Y_train = Y_train.reshape(Y_train.shape[0],1)
Y_test  = Y_test.reshape(Y_test.shape[0],1)

lmbdas = np.logspace(-5,5,11)

# Extract a smaller sample to reduce computational time

m = X_train.shape[0]
n = X_train.shape[1]

X   = X_train[0:800,:].reshape(800,n)
Y   = Y_train[0:800,:].reshape(800,1)
X_t = X_test[0:200,:].reshape(200,n)
Y_t = Y_test[0:200,:].reshape(200,1)


error_l2, error_t_l2 = bootstrap_resampling_logistic(Y, Y_t, X, X_t, lmbdas, method='l2',n_boostraps = 100)
error_l1, error_t_l1 = bootstrap_resampling_logistic(Y, Y_t, X, X_t, lmbdas, method='l1',n_boostraps = 100)


# plot states
fig, axarr = plt.subplots(nrows=1, ncols=2)

axarr[0].semilogx(lmbdas, error_l2, '*--b', lmbdas, error_t_l2, '*--r', lmbdas[np.argmin(error_t_l2)], np.min(error_t_l2), 'x--k',lw=2)
axarr[0].tick_params(labelsize=16)
axarr[0].legend(('Training set L2','Test set L2'))
axarr[0].set_ylabel('$\\mathrm{Loss}$')
axarr[0].set_xlabel('$\\lambda$')
axarr[0].grid()

axarr[1].semilogx(lmbdas, error_l1, '*--g', lmbdas, error_t_l1, '*--c', lmbdas[np.argmin(error_t_l1)], np.min(error_t_l1), 'x--k', lw=2)
axarr[1].tick_params(labelsize=16)
axarr[1].legend(('Training set L1','Test set L1'))
axarr[1].set_ylabel('$\\mathrm{Loss}$')
axarr[1].set_xlabel('$\\lambda$')
axarr[1].grid()

fig.subplots_adjust(right=2.0)
plt.savefig('Results_Classification/Bootstrap_resampling_l2_l1.png', dpi=600,bbox_inches = 'tight')

plt.show()

# Plotting best fit accuracy for Lasso

LassoEst = linear_model.SGDClassifier(loss='log', penalty='l1', alpha=0.001, max_iter=100, shuffle=False, random_state=1, learning_rate='optimal')

y_pred   = np.ravel(LassoEst.fit(X,Y).predict(X_t))

BestScore    = accuracy_score(Y_t,y_pred)
