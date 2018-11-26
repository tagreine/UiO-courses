# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 18:44:29 2018

@author: tagreine
"""

#=================================================================================================
#========================= Simple test of the functions ==========================================

import numpy as np
import tensorflow as tf
#from scipy.misc import imresize
from skimage.transform import resize
import matplotlib.pyplot as plt
from CNN_Interpolate_tf import create_CNN_int,create_loss,create_optimizer,create_placeholders, Interpolation_model

#logs_path    = "./logs"  # path to the folder that we want to save the logs for Tensorboard

# ================================= Get training data ============================================
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
train_loss = np.zeros([epochs,1])
test_loss  = np.zeros([epochs,1])

samp_hr    = shape_target[1]
samp_lr    = np.int(shape_target[2]/ratio)
 

#summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

#=============================== Start tensorflow session =========================================
with tf.Session() as sess:
    x,xt,y,yt          = create_placeholders(input_,input_test,samp_hr,samp_lr)
    pred_train, params = create_CNN_int(x,  shape_target, ratio, samp_hr, samp_lr)
    pred_test          = Interpolation_model(xt, shape_target_test, ratio, params, samp_hr, samp_lr)

    loss  = create_loss(y,  pred_train)
    losst = create_loss(yt, pred_test)
    
    training   = create_optimizer(loss)
    tf.global_variables_initializer().run() 
    
    for epoch in range(epochs):
    
        train, train_loss[epoch] = sess.run([training, loss], feed_dict = {x: input_, y: target_})        
        test_loss[epoch]         = sess.run([losst], feed_dict = {xt: input_test, yt: target_test})
        
        print('Epoch', epoch, 'completed out of', epochs, 'Training loss:', train_loss[epoch],'Test loss:', test_loss[epoch])
               
                
    out_train, parameters = np.squeeze(sess.run([pred_train, params], feed_dict = {x:  input_}))





with tf.Session() as sess:
    x,xt,y,yt  = create_placeholders(input_,input_test,samp_hr,samp_lr)
    pred_test = Interpolation_model(xt, shape_target_test, ratio, parameters, samp_hr, samp_lr)
    out_test  = np.squeeze(sess.run(pred_test,  feed_dict = {xt: input_test}))




# Display interpolated training data
n = 1 # how many digits we will display
m = 0
a = 0.1
plt.figure(figsize=(20, 10))
for i in range(n):
    
    # display original
    ax = plt.subplot(n, 4, i + 1)
    plt.imshow(target_test[i+m].reshape(200, 200),aspect="auto")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Truth')
    plt.colorbar()
    plt.clim(-a,a)
    
    # display input
    ax = plt.subplot(n, 4, i + 1 + n)
    plt.imshow(input_test[i+m].reshape(200, 50),aspect="auto")
    #input_int = resize(input_test[i,:,:,0], (samp_hr,samp_hr), order=5)
    #plt.imshow(input_int,aspect="auto")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Input')
    plt.colorbar()
    plt.clim(-a,a)

    # display reconstruction
    ax = plt.subplot(n, 4, i + 1 + 2*n)
    plt.imshow(out_test[i+m].reshape(200, 200),aspect="auto")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Predicted')
    plt.colorbar()
    plt.clim(-a,a)
    
    # display reconstruction
    ax = plt.subplot(n, 4, i + 1 + 3*n)
    plt.imshow((out_test[i+m].reshape(200, 200) - target_test[i+m].reshape(200, 200)),aspect="auto")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Truth-Predicted')
    plt.colorbar()
    plt.clim(-a,a)
         
plt.savefig('Interpolated_shot_test_set5.png', dpi=600)    
plt.show()


























# Display interpolated training data
n = 10  # how many digits we will display
plt.figure(figsize=(10, 4))
for i in range(n):
    
    # display original
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(target_[i].reshape(28, 28),aspect="auto")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display input
    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(input_[i].reshape(28, 7),aspect="auto")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(4, n, i + 1 + 2*n)
    input_int = imresize(input_[i,:,:,0], (samp_hr,samp_hr,1), interp='bicubic', mode=None)
    plt.imshow(input_int,aspect="auto")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(4, n, i + 1 + 3*n)
    plt.imshow(test[i].reshape(28, 28)/np.max(test),aspect="auto")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)  
       
#plt.savefig('Interpolation_test.png', dpi=600)    
plt.show()
