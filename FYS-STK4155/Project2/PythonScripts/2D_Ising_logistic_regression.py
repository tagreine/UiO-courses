# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 09:32:32 2018

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
from NN_functions import predict, logistic_regression_batch, log_likelihood, logistic_reg_cost
from tools import R2_metric
import numpy as np
import warnings
from time import sleep
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


print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
#print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')


cmap_args=dict(cmap='plasma_r')

# plot states
fig, axarr = plt.subplots(nrows=1, ncols=2)

axarr[0].imshow(X_ordered[20001].reshape(L,L),**cmap_args)
axarr[0].set_title('$\\mathrm{Ordered\\ phase}$',fontsize=16)
axarr[0].tick_params(labelsize=16)

im=axarr[1].imshow(X_disordered[50001].reshape(L,L),**cmap_args)
axarr[1].set_title('$\\mathrm{Disordered\\ phase}$',fontsize=16)
axarr[1].tick_params(labelsize=16)


fig.subplots_adjust(right=1.0)
#plt.savefig('Results_Logistic_Classification/Ordered_Disordered.png', dpi=600,bbox_inches = 'tight')

plt.show()

############################### Training ########################################

# We will train the estimators using stochastic gradient descent with mini batches

Y_train = Y_train.reshape(Y_train.shape[0],1)
Y_test  = Y_test.reshape(Y_test.shape[0],1)

lmbdas = np.logspace(-5,5,11)

# preallocate data
train_accuracy_SGD = np.zeros(lmbdas.shape,np.float64)
test_accuracy_SGD  = np.zeros(lmbdas.shape,np.float64)
train_R2_SGD       = np.zeros(lmbdas.shape,np.float64)
test_R2_SGD        = np.zeros(lmbdas.shape,np.float64)
train_MSE_SGD      = np.zeros(lmbdas.shape,np.float64)
test_MSE_SGD       = np.zeros(lmbdas.shape,np.float64)


m = X_train.shape[0]
n = X_train.shape[1]

# Extract a smaller sample to reduce computational time
X   = X_train[0:800,:].reshape(800,n)
Y   = Y_train[0:800,:].reshape(800,1)
X_t = X_test[0:200,:].reshape(200,n)
Y_t = Y_test[0:200,:].reshape(200,1)

for i,lm in enumerate(lmbdas):
    sleep(2)
    
    np.random.seed(1)
    theta_train = np.zeros(X.shape[1]).reshape(X.shape[1],1)

    cost_SGD, theta_train = logistic_regression_batch( X, Y, theta_train, epochs = 500, batch_size = 50,alpha = 0.05, lmbda = lm, intercept = 'False')
            
    Y_train_pred_SGD      = predict(X.dot(theta_train),0.5)
    Y_test_pred_SGD       = predict(X_t.dot(theta_train),0.5)
      
    train_accuracy_SGD[i] = accuracy_score(Y,Y_train_pred_SGD)
    test_accuracy_SGD[i]  = accuracy_score(Y_t,Y_test_pred_SGD)
    
    train_R2_SGD[i]       = R2_metric(Y,Y_train_pred_SGD)
    test_R2_SGD[i]        = R2_metric(Y_t,Y_test_pred_SGD)

    train_MSE_SGD[i]      = logistic_reg_cost(X, Y_train_pred_SGD, theta_train, lmbda = lm, intercept = 'False')#log_likelihood(X,Y_train_pred_SGD,theta_train)
    test_MSE_SGD[i]       = logistic_reg_cost(X_t, Y_test_pred_SGD, theta_train, lmbda = lm, intercept = 'False')#log_likelihood(X_t,Y_test_pred_SGD,theta_train)
    
    
    print( 'Next regularization parameter' )
    

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
axarr[2].set_ylabel('$\\mathrm{Loss}$')
axarr[2].set_xlabel('$\\lambda$')
axarr[2].grid()


fig.subplots_adjust(right=2.5)

plt.savefig('Results_Classification/Classification_accuracy_R2_loss_train_test.png', dpi=600,bbox_inches = 'tight')

plt.show()

  













