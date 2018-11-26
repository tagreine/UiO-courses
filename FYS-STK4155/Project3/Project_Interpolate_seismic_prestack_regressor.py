# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 19:48:07 2018

@author: tagreine
"""

import numpy as np
from sklearn.neural_network_network import MLPClassifier
import matplotlib.pyplot as plt
import os
os.chdir('M:\FYS-STK4155\Project3\LoadSegyData')
from LoadTrainingData import load_dat
# Import seismic data for training
data = load_dat('Training_data.mat')


os.chdir('M:\FYS-STK4155\Project3')
# Split data to training set and test set

X_train = data[0:15]
X_test  = data[15:20]

X_train = X_train[:,0:200,:]
X_test  = X_test[:,0:200,:]

#============================= Plotting train and test sets ======================================    
# plot the training and test images
n = 5
plt.figure(figsize=(10, 10))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(X_train[i].reshape(200, 200))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#============================= Define target data and input data for training set and test set ======================================   
ratio        = 4         
target_      = X_train
target_      = np.expand_dims(target_,axis=3)
shape_target = target_.shape

target_test  = X_test
target_test  = np.expand_dims(target_test,axis=3)
shape_target_test = target_test.shape

# Define input values
input_       = target_[:,:,::ratio,:]
shape_input  = input_.shape

input_test   = target_test[:,:,::ratio,:]
shape_input_test  = input_test.shape

#============================= Plotting targets and inputs ======================================    
# plot the training and test images
n = 5  # how many digits we will display
plt.figure(figsize=(10, 4))
for i in range(n):
    
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(target_[i].reshape(200, 200),aspect="auto")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display input
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(input_[i].reshape(200, 50),aspect="auto")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
     
#plt.savefig('\Results\Targets_Inputs.png', dpi=600)    
plt.show()

# Define training parameters
epoch_loss = 0
epochs     = 1000
samp_hr    = shape_target[1]
samp_lr    = np.int(shape_target[2]/ratio)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 10, 5, 10, 20), random_state=1)















