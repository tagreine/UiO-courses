# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:30:47 2018

@author: tagreine
"""

import os
import numpy as np
import tensorflow as tf
#from scipy.misc import imresize
#from skimage.transform import resize
import matplotlib.pyplot as plt
os.chdir('M:\FYS-STK4155\Project3')
from CNN_Interpolate_transpose_tf import create_CNN_int,create_loss,create_optimizer,create_placeholders, Interpolation_model, create_loss_l1, create_huber_loss, create_Adadelta_optimizer, create_Adagrad_optimizer
from DisplayImages import display_training_im
from ImageMetrics import PSNR

#logs_path    = "./logs"  # path to the folder that we want to save the logs for Tensorboard

# ================================= Get training data ============================================

os.chdir('M:\FYS-STK4155\Project3\LoadSegyData')
from LoadTrainingData import load_dat
# Import seismic data for training
data = load_dat('Training_data.mat')


os.chdir('M:\FYS-STK4155\Project3')
# Split data to training set and test set

X_train = data[0:15]
X_test  = data[15:20]

a = 200
b = 100

X_train = X_train[:,0:a,0:b]
X_test  = X_test[:,0:a,0:b]


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

display_training_im(target_, input_, a=200, b=100, ratio = 4,n=4,c=0.2,save='False',name='Results\Targets_Inputs.png')

# Define training parameters
epoch_loss = 0
epochs     = 1000
train_loss_adam = np.zeros([epochs,1])
test_loss_adam  = np.zeros([epochs,1])
train_loss_adag = np.zeros([epochs,1])
test_loss_adag  = np.zeros([epochs,1])
train_loss_adad = np.zeros([epochs,1])
test_loss_adad  = np.zeros([epochs,1])

samp_hry   = shape_target[1]
samp_hrx   = shape_target[2]
samp_lr    = np.int(shape_target[2]/ratio)
 

#summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

#=============================== Start tensorflow session =========================================
with tf.Session() as sess:

    x,xt,y,yt = create_placeholders(input_,input_test,samp_hry, samp_hrx,samp_lr)
    
    # L2 loss 
    pred_train_adam, params_adam = create_CNN_int(x, shape_target, ratio, samp_hry, samp_hrx, samp_lr)
    pred_test_adam               = Interpolation_model(xt, shape_target_test, ratio, params_adam, samp_hry, samp_hrx, samp_lr)

    loss_adam     = create_loss(y,  pred_train_adam)
    losst_adam    = create_loss(yt, pred_test_adam)     
    training_adam = create_optimizer(loss_adam, eta=0.001)

    # L1 loss
    pred_train_adag, params_adag  = create_CNN_int(x, shape_target, ratio, samp_hry, samp_hrx, samp_lr)
    pred_test_adag              = Interpolation_model(xt, shape_target_test, ratio, params_adag, samp_hry, samp_hrx, samp_lr)
    
    loss_adag     = create_loss(y,  pred_train_adag)
    losst_adag    = create_loss(yt, pred_test_adag)     
    training_adag = create_Adagrad_optimizer(loss_adag, eta=0.01)

    # Huber loss
    pred_train_adad, params_adad = create_CNN_int(x, shape_target, ratio, samp_hry, samp_hrx, samp_lr)
    pred_test_adad               = Interpolation_model(xt, shape_target_test, ratio, params_adad, samp_hry, samp_hrx, samp_lr)
    
    loss_adad     = create_loss(y,  pred_train_adad)
    losst_adad    = create_loss(yt, pred_test_adad)     
    training_adad = create_Adadelta_optimizer(loss_adad, eta=0.1)  
    
#================================================================================================    
    tf.global_variables_initializer().run() 
    
    for epoch in range(epochs):
        
        # L2
        train2, train_loss_adam[epoch]   = sess.run([training_adam, loss_adam], feed_dict = {x: input_, y: target_})        
        test_loss_adam[epoch]            = sess.run([losst_adam], feed_dict = {xt: input_test, yt: target_test})
        
        # L1
        train1, train_loss_adag[epoch]   = sess.run([training_adag, loss_adag], feed_dict = {x: input_, y: target_})        
        test_loss_adag[epoch]            = sess.run([losst_adag], feed_dict = {xt: input_test, yt: target_test})        
        
         # Huber
        trainH, train_loss_adad[epoch] = sess.run([training_adad, loss_adad], feed_dict = {x: input_, y: target_})        
        test_loss_adad[epoch]          = sess.run([losst_adad], feed_dict = {xt: input_test, yt: target_test})         
        
        print('Epoch', epoch, 'completed out of', epochs, 'Loss Adam:', test_loss_adam[epoch],'Loss Adagrad:', test_loss_adag[epoch],'Loss Adadelta:', test_loss_adad[epoch])
     
    out_train_adam, parameters_adam = np.squeeze(sess.run([pred_train_adam, params_adam], feed_dict = {x:  input_}))
    out_train_adag, parameters_adag = np.squeeze(sess.run([pred_train_adag, params_adag], feed_dict = {x:  input_}))
    out_train_adad, parameters_adad = np.squeeze(sess.run([pred_train_adad, params_adad], feed_dict = {x:  input_}))
    

    
plt.figure(figsize=(7, 4))
plt.subplot(1,2,1)    
plt.semilogx(np.arange(epochs),test_loss_adam,'k',np.arange(epochs),test_loss_adag,'b',np.arange(epochs),test_loss_adad,'g',lw=2)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam','Adagrad','Adadelta'])
plt.ylim([0,0.009])
plt.grid()

plt.subplot(1,2,2)    
plt.plot(np.arange(epochs),test_loss_adam,'k',np.arange(epochs),test_loss_adag,'b',np.arange(epochs),test_loss_adad,'g',lw=2)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam','Adagrad','Adadelta'])
plt.ylim([0,0.004])
plt.xlim([400,500])
plt.grid()

plt.subplots_adjust(wspace=0.4) 
  
plt.savefig('Results\Loss_Adam_Adagrad_Adadelta', dpi=600)
 
plt.show()      
    

# Predict test set with trained function
with tf.Session() as sess:
    x,xt,y,yt  = create_placeholders(input_,input_test,samp_hry, samp_hrx,samp_lr)
    
    pred_test_adam = Interpolation_model(xt, shape_target_test, ratio, parameters_adam, samp_hry, samp_hrx, samp_lr)
    out_test_adam  = np.squeeze(sess.run(pred_test_adam,  feed_dict = {xt: input_test}))    
    
    pred_test_adag = Interpolation_model(xt, shape_target_test, ratio, parameters_adag, samp_hry, samp_hrx, samp_lr)
    out_test_adag  = np.squeeze(sess.run(pred_test_adag,  feed_dict = {xt: input_test}))    
        
    pred_test_adad = Interpolation_model(xt, shape_target_test, ratio, parameters_adad, samp_hry, samp_hrx, samp_lr)
    out_test_adad  = np.squeeze(sess.run(pred_test_adad,  feed_dict = {xt: input_test}))    
        
    
        
psnr_adam   = np.zeros([shape_input_test[0],1])
psnr_adag  = np.zeros([shape_input_test[0],1])
psnr_adad = np.zeros([shape_input_test[0],1])



for i in range(shape_input_test[0]):
    psnr_adam[i] = PSNR(target_test[i,:,:,0], out_test_adam[i,:,:])
    psnr_adag[i] = PSNR(target_test[i,:,:,0], out_test_adag[i,:,:])
    psnr_adad[i] = PSNR(target_test[i,:,:,0], out_test_adad[i,:,:])

    
mean_psnr_adam = np.mean(psnr_adam)
mean_psnr_adag = np.mean(psnr_adag)
mean_psnr_adad = np.mean(psnr_adad)    



