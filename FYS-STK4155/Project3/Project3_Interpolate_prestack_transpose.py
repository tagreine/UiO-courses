# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 18:44:29 2018

@author: tagreine
"""

#=================================================================================================
#========================= Simple test of the functions ==========================================
import os
import numpy as np
import tensorflow as tf
#from scipy.misc import imresize
#from skimage.transform import resize
import matplotlib.pyplot as plt
os.chdir('M:\FYS-STK4155\Project3')
from CNN_Interpolate_transpose_tf import create_CNN_int,create_loss,create_optimizer,create_placeholders, Interpolation_model, create_huber_loss
from DisplayImages import display_gathers, display_training_im, plotCNNFilter
from sklearn.metrics import mean_squared_error, r2_score
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
train_loss = np.zeros([epochs,1])
test_loss  = np.zeros([epochs,1])

samp_hry   = shape_target[1]
samp_hrx   = shape_target[2]
samp_lr    = np.int(shape_target[2]/ratio)
 

#summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

#=============================== Start tensorflow session =========================================

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        x,xt,y,yt = create_placeholders(input_,input_test,samp_hry, samp_hrx,samp_lr)
        pred_train, params  = create_CNN_int(x, shape_target, ratio, samp_hry, samp_hrx, samp_lr)
        pred_test           = Interpolation_model(xt, shape_target_test, ratio, params, samp_hry, samp_hrx, samp_lr)
        
        #loss  = create_huber_loss(y,  pred_train,1.0)
        #losst = create_huber_loss(yt, pred_test,1.0)
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
    x,xt,y,yt  = create_placeholders(input_,input_test,samp_hry, samp_hrx,samp_lr)
    pred_test = Interpolation_model(xt, shape_target_test, ratio, parameters, samp_hry, samp_hrx, samp_lr)
    out_test  = np.squeeze(sess.run(pred_test,  feed_dict = {xt: input_test}))


# Display interpolated training data
display_gathers(target_,input_,out_train, a = 200, b = 100, ratio = 4,n=1,m=4,c=0.1,save='True',name='Results\Interpolated_train_transp.png')
# Display interpolated test data
display_gathers(target_test,input_test,out_test, a = 200, b = 100, ratio = 4,n=1,m=3,c=0.1,save='True',name='Results\Interpolated_test_transp_huber.png')
# Display first layer weights  
plotCNNFilter(parameters['W6'],save='True',name='Results\Trained_weights_transp.png')

# plot loss vs epochs
plt.figure(figsize=(7,4))
plt.subplot(1,2,1)
plt.semilogx(np.arange(epochs),test_loss,np.arange(epochs),train_loss,lw=2)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test set','Training set'])
plt.grid()

plt.subplot(1,2,2)
plt.plot(np.arange(epochs),test_loss,np.arange(epochs),train_loss,lw=2)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test set','Training set'])
plt.ylim([0.00006,0.0002])
plt.xlim([400,1000])
plt.grid()
plt.subplots_adjust(wspace=0.4)

plt.savefig('Results\Loss_vs_epoch_transp_adam_huber', dpi=600)


MSE = mean_squared_error(target_test.reshape(shape_target_test[0]*shape_target_test[1]*shape_target_test[2]*shape_target_test[3],1), out_test.reshape(shape_target_test[0]*shape_target_test[1]*shape_target_test[2]*shape_target_test[3],1))
R2  = r2_score(target_test.reshape(shape_target_test[0]*shape_target_test[1]*shape_target_test[2]*shape_target_test[3],1), out_test.reshape(shape_target_test[0]*shape_target_test[1]*shape_target_test[2]*shape_target_test[3],1))


psnr = np.zeros([shape_input_test[0],1])
for i in range(shape_input_test[0]):
    psnr[i] = PSNR(target_test[i,:,:,0], out_test[i,:,:])
    
mean_psnr = np.mean(psnr)















