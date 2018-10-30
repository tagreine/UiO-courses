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
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from NN_functions import sigmoid, gradientDescent, predict,
from tools import R2_metric, MSE_metric
import numpy as np
import warnings
from sklearn import linear_model
from sklearn.utils import resample
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
train_to_test_ratio = 0.8 # training samples

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
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=train_to_test_ratio)

# full data set
#X=np.concatenate((X_critical,X))
#Y=np.concatenate((Y_critical,Y))

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

#axarr[1].imshow(X_critical[10001].reshape(L,L),**cmap_args)
#axarr[1].set_title('$\\mathrm{critical\\ region}$',fontsize=16)
#axarr[1].tick_params(labelsize=16)

fig.subplots_adjust(right=1.0)
#plt.savefig('Results_Logistic_Classification/Ordered_Disordered.png', dpi=600,bbox_inches = 'tight')

plt.show()

############################### Training ########################################

# We will train the estimators using stochastic gradient descent

m = X_train.shape[0]
n = X_train.shape[1]

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

#train_accuracy        = np.zeros(lmbdas.shape,np.float64)
#test_accuracy         = np.zeros(lmbdas.shape,np.float64)


num_iter = m
import time
t = time.time()
for i,lm in enumerate(lmbdas):
    
    np.random.seed(1)
    theta_train_SGD = 0.01*np.random.random(1600).reshape(X_train.shape[1],1)
    epochs = 10
    
    for k in range(epochs):

        for j in range(num_iter):
    
            cost_SGD, theta_train_SGD = gradientDescent(X_train[j,:].reshape(1,n), Y_train[j,:].reshape(1,1), theta_train_SGD, method = 'Logistic', alpha = 0.1, lmbda=lm , num_iters = 1, intercept = 'False')

    
    # Predict the magnetic phases. For ordered phase the prediction has been set to threshold >=0.5 
    Y_train_pred_SGD      = predict(X_train.dot(theta_train_SGD))
    Y_test_pred_SGD       = predict(X_test.dot(theta_train_SGD))
      
    train_accuracy_SGD[i] = accuracy_score(Y_train,Y_train_pred_SGD)
    test_accuracy_SGD[i]  = accuracy_score(Y_test,Y_test_pred_SGD)
    
    train_R2_SGD[i]       = r2_score(Y_train,Y_train_pred_SGD)
    test_R2_SGD[i]        = r2_score(Y_test,Y_test_pred_SGD)

    train_MSE_SGD[i]      = MSE_metric(Y_train,Y_train_pred_SGD)
    test_MSE_SGD[i]       = MSE_metric(Y_test,Y_test_pred_SGD)

elapsed = time.time() - t
  
fig, axarr = plt.subplots(nrows=1, ncols=3)

axarr[0].semilogx(lmbdas,train_accuracy_SGD,'*--b',lmbdas,test_accuracy_SGD,'*--r',lw=2)
axarr[0].legend(('Training set','Test set'))
axarr[0].set_ylabel('$\\mathrm{Accuracy}$')
axarr[0].set_xlabel('$\\lambda$')
axarr[0].grid()

axarr[1].semilogx(lmbdas,train_R2_SGD,'*--k',lmbdas,test_R2_SGD,'*--c',lw=2)
axarr[1].legend(('Training set','Test set'))
axarr[1].set_ylabel('$\\mathrm{R2 score}$')
axarr[1].set_xlabel('$\\lambda$')
axarr[1].grid()

axarr[2].semilogx(lmbdas,train_MSE_SGD,'*--y',lmbdas,test_MSE_SGD,'*--m',lw=2)
axarr[2].legend(('Training set','Test set'))
axarr[2].set_ylabel('$\\mathrm{MSE}$')
axarr[2].set_xlabel('$\\lambda$')
axarr[2].grid()


fig.subplots_adjust(right=2.5)

#plt.savefig('Results_Logistic_Classification/Ordered_Disordered.png', dpi=600,bbox_inches = 'tight')
plt.show()
#===================================================================================================

# We will train the estimators and assess the model using stochastic gradient descent and bootstrap

model_comp = lmbdas.copy()
n_boostraps = 100

def bootstrap_resampling_logistic(y_train, y_test, x_train, x_test, model_complx, n_boostraps = 500):
    
    # Bootstrap algorithm for model assessment
    
    
    mc = len(model_complx)
    
    error    = np.zeros([mc])
    bias     = np.zeros([mc])
    variance = np.zeros([mc])

    error_t    = np.zeros([mc])
    bias_t     = np.zeros([mc])
    variance_t = np.zeros([mc])
    
    for j in range(mc):   
        
        y_pred   = np.empty((y_train.shape[0], n_boostraps))
        y_pred_t = np.empty((y_test.shape[0], n_boostraps))
        
        logreg_SGD = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=model_complx[j], max_iter=100, 
                                           shuffle=False, random_state=1, learning_rate='optimal')
        
        for i in range(n_boostraps):
            x_, y_ = resample(x_train, y_train)
            
            # Predict training and test data for each bootstrap
            y_pred[:, i]   = np.ravel(logreg_SGD.fit(x_,y_).predict(x_train))
            y_pred_t[:, i] = np.ravel(logreg_SGD.fit(x_,y_).predict(x_test))
            
            # Compute the error, variance and bias squared at eah point in the model    
        error[j]    = np.mean( np.mean((y_train - y_pred)**2, axis=1, keepdims=True) )
        bias[j]     = np.mean( (y_train - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance[j] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
            
        error_t[j]    = np.mean( np.mean((y_test - y_pred_t)**2, axis=1, keepdims=True) )
        bias_t[j]     = np.mean( (y_test - np.mean(y_pred_t, axis=1, keepdims=True))**2 )
        variance_t[j] = np.mean( np.var(y_pred_t, axis=1, keepdims=True) )
            
    return error, bias, variance, error_t, bias_t, variance_t



x_train = X_train[0:5000,:]
y_train = Y_train[0:5000,:]
x_test  = X_test[0:5000,:]
y_test  = Y_test[0:5000,:]

error, bias, variance, error_t, bias_t, variance_t = bootstrap_resampling_logistic(y_train, y_test, x_train, x_test, model_comp, n_boostraps)


# plot states
fig, axarr = plt.subplots(nrows=1, ncols=2)

axarr[0].semilogx(model_comp, error, '*--b', model_comp, error_t, '*--r', lw=2)
#axarr[0].tick_params(labelsize=16)
axarr[0].legend(('Training set','Test set'))
axarr[0].set_ylabel('$\\mathrm{Prediction error}$')
axarr[0].set_xlabel('$\\lambda$')
axarr[0].grid()

axarr[1].semilogx(model_comp, variance, '--k', model_comp, variance_t, '--c', model_comp, bias, 'y', model_comp, bias_t, 'm', lw=2)
#axarr[1].tick_params(labelsize=16)
axarr[1].legend(('Training variance','Test variance','Training bias','Test bias'))
axarr[1].set_xlabel('$\\lambda$')
axarr[1].grid()
fig.subplots_adjust(right=1.0)
#plt.savefig('Results_Logistic_Classification/Ordered_Disordered.png', dpi=600,bbox_inches = 'tight')

plt.show()







