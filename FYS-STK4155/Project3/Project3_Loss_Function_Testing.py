# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:16:50 2018

@author: tagreine
"""

import os
import numpy as np
import tensorflow as tf
#from scipy.misc import imresize
#from skimage.transform import resize
import matplotlib.pyplot as plt
os.chdir('M:\FYS-STK4155\Project3')
from CNN_Interpolate_transpose_tf import create_CNN_int,create_loss,create_optimizer,create_placeholders, Interpolation_model, create_loss_l1, create_huber_loss
from DisplayImages import display_training_im, display_gathers
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
train_loss_l2 = np.zeros([epochs,1])
test_loss_l2  = np.zeros([epochs,1])
train_loss_l1 = np.zeros([epochs,1])
test_loss_l1  = np.zeros([epochs,1])
train_loss_l2l1 = np.zeros([epochs,1])
test_loss_l2l1  = np.zeros([epochs,1])

samp_hry   = shape_target[1]
samp_hrx   = shape_target[2]
samp_lr    = np.int(shape_target[2]/ratio)
 

#summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

#=============================== Start tensorflow session =========================================
with tf.Session() as sess:

    x,xt,y,yt = create_placeholders(input_,input_test,samp_hry, samp_hrx,samp_lr)
    
    # L2 loss 
    pred_train_l2, params_l2  = create_CNN_int(x, shape_target, ratio, samp_hry, samp_hrx, samp_lr)
    pred_test_l2              = Interpolation_model(xt, shape_target_test, ratio, params_l2, samp_hry, samp_hrx, samp_lr)

    loss_l2     = create_loss(y,  pred_train_l2)
    losst_l2    = create_loss(yt, pred_test_l2)     
    training_l2 = create_optimizer(loss_l2)

    # L1 loss
    pred_train_l1, params_l1  = create_CNN_int(x, shape_target, ratio, samp_hry, samp_hrx, samp_lr)
    pred_test_l1              = Interpolation_model(xt, shape_target_test, ratio, params_l1, samp_hry, samp_hrx, samp_lr)
    
    loss_l1     = create_loss_l1(y,  pred_train_l1)
    losst_l1    = create_loss_l1(yt, pred_test_l1)     
    training_l1 = create_optimizer(loss_l1)

    # Huber loss
    pred_train_l2l1, params_l2l1  = create_CNN_int(x, shape_target, ratio, samp_hry, samp_hrx, samp_lr)
    pred_test_l2l1              = Interpolation_model(xt, shape_target_test, ratio, params_l2l1, samp_hry, samp_hrx, samp_lr)
    
    loss_l2l1     = create_huber_loss(y,  pred_train_l2l1,Delta=1.0)
    losst_l2l1    = create_huber_loss(yt, pred_test_l2l1,Delta=1.0)     
    training_l2l1 = create_optimizer(loss_l2l1)  
    
#================================================================================================    
    tf.global_variables_initializer().run() 
    
    for epoch in range(epochs):
        
        # L2
        train2, train_loss_l2[epoch]   = sess.run([training_l2, loss_l2], feed_dict = {x: input_, y: target_})        
        test_loss_l2[epoch]            = sess.run([losst_l2], feed_dict = {xt: input_test, yt: target_test})
        
        # L1
        train1, train_loss_l1[epoch]   = sess.run([training_l1, loss_l1], feed_dict = {x: input_, y: target_})        
        test_loss_l1[epoch]            = sess.run([losst_l1], feed_dict = {xt: input_test, yt: target_test})        
        
         # Huber
        trainH, train_loss_l2l1[epoch] = sess.run([training_l2l1, loss_l2l1], feed_dict = {x: input_, y: target_})        
        test_loss_l2l1[epoch]          = sess.run([losst_l2l1], feed_dict = {xt: input_test, yt: target_test})         
        
        print('Epoch', epoch, 'completed out of', epochs, 'Loss L2:', test_loss_l2[epoch],'Loss L1:', test_loss_l1[epoch],'Loss Huber:', test_loss_l2l1[epoch])
     
    out_train_l2, parameters_l2 = np.squeeze(sess.run([pred_train_l2, params_l2], feed_dict = {x:  input_}))
    out_train_l1, parameters_l1 = np.squeeze(sess.run([pred_train_l1, params_l1], feed_dict = {x:  input_}))
    out_train_l2l1, parameters_l2l1 = np.squeeze(sess.run([pred_train_l2l1, params_l2l1], feed_dict = {x:  input_}))
                 
# plot loss vs epochs
plt.figure(figsize=(7, 4))
plt.subplot(1,2,1)    
plt.semilogx(np.arange(epochs),test_loss_l2,'r',np.arange(epochs),test_loss_l1,'b',np.arange(epochs),test_loss_l2l1,'g',lw=2)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['L2','L1','Huber'])
plt.ylim([0,0.06])
plt.grid()

plt.subplot(1,2,2)    
plt.semilogx(np.arange(epochs),test_loss_l2,'r',np.arange(epochs),test_loss_l2l1,'g',lw=2)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['L2','Huber'])
plt.ylim([0,0.005])
plt.grid()

plt.subplots_adjust(wspace=0.4)   
#plt.savefig('Results\LossL2_L1_Huber_vs_epoch_sub_adam', dpi=600)
 
plt.show()    


# Predict test set with trained function
with tf.Session() as sess:
    x,xt,y,yt  = create_placeholders(input_,input_test,samp_hry, samp_hrx,samp_lr)
    
    pred_test_l2 = Interpolation_model(xt, shape_target_test, ratio, parameters_l2, samp_hry, samp_hrx, samp_lr)
    out_test_l2  = np.squeeze(sess.run(pred_test_l2,  feed_dict = {xt: input_test}))    
    
    pred_test_l1 = Interpolation_model(xt, shape_target_test, ratio, parameters_l1, samp_hry, samp_hrx, samp_lr)
    out_test_l1  = np.squeeze(sess.run(pred_test_l1,  feed_dict = {xt: input_test}))    
        
    pred_test_l2l1 = Interpolation_model(xt, shape_target_test, ratio, parameters_l2l1, samp_hry, samp_hrx, samp_lr)
    out_test_l2l1  = np.squeeze(sess.run(pred_test_l2l1,  feed_dict = {xt: input_test}))    
        
    
        
psnr_l2   = np.zeros([shape_input_test[0],1])
psnr_l1   = np.zeros([shape_input_test[0],1])
psnr_l2l1 = np.zeros([shape_input_test[0],1])



for i in range(shape_input_test[0]):
    psnr_l2[i]   = PSNR(target_test[i,:,:,0], out_test_l2[i,:,:])
    psnr_l1[i]   = PSNR(target_test[i,:,:,0], out_test_l1[i,:,:])
    psnr_l2l1[i] = PSNR(target_test[i,:,:,0], out_test_l2l1[i,:,:])

    
mean_psnr_l2   = np.mean(psnr_l2)
mean_psnr_l1   = np.mean(psnr_l1)
mean_psnr_l2l1 = np.mean(psnr_l2l1)    



display_gathers(target_test,input_test, out_test_l2, a = 200, b = 100, ratio = 4,n=1,m=3,c=0.1,save='True',name='Results\Interpolated_l2.png')
display_gathers(target_test,input_test, out_test_l1, a = 200, b = 100, ratio = 4,n=1,m=3,c=0.1,save='True',name='Results\Interpolated_l1.png')
display_gathers(target_test,input_test, out_test_l2l1, a = 200, b = 100, ratio = 4,n=1,m=3,c=0.1,save='True',name='Results\Interpolated_l2l1.png')




