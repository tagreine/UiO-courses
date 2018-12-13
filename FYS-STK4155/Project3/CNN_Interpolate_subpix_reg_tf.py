# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:28:52 2018

@author: tagreine
"""

import tensorflow as tf


#======================================================================================================
# Define tensorflow placeholders for targets and inputs in relation to training set and test set 
def create_placeholders(X_train,X_test,samp_hry,samp_hrx,samp_lr):
    with tf.name_scope('data'):
        
        shape  = X_train.shape
        shapet = X_test.shape
        
        x  = tf.placeholder(tf.float32, [shape[0],  samp_hry, samp_lr, 1], name='inputs')
        xt = tf.placeholder(tf.float32, [shapet[0], samp_hry, samp_lr, 1], name='test_inputs')
        y  = tf.placeholder(tf.float32, [shape[0],  samp_hry, samp_hrx, 1], name='training_targets')
        yt = tf.placeholder(tf.float32, [shapet[0], samp_hry, samp_hrx, 1], name='test_targets')
        
        return x,xt,y,yt

#======================================================================================================        
# Define tensorflow variables
def weight_variables(filter_shape=[5,5,1,64],std=0.1,name='w1'):
    with tf.name_scope('weigths'):
        weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=std), name=name)          
        return weight

def bias_variables(filter_shape=64,name='b1'):
    with tf.name_scope('biases'):
        bias = tf.Variable(tf.zeros([filter_shape], name=name))      
        return bias
#======================================================================================================        
# Define tensorflow model    
def create_CNN_int(X, shape,ratio,samp_hry, samp_hrx,samp_lr):
    with tf.name_scope('CNN_int'):
    
        # Define standard deviation for the random initializaion
        std = 0.1
        # Define filter shapes for the conv net
        filter_shape1 = [9, 9, 1, 64]     #filter_shape1 = [11, 11, 1, 32]
        filter_shape2 = [3, 3, 64, 64]    #filter_shape2 = [3, 3, 32, 32]
        filter_shape3 = [3, 3, 64, ratio] #filter_shape3 = [3, 3, 32, ratio]
        filter_shape4 = [3, 3, 1, 128]    #filter_shape4 = [7, 7, 1, 32]
        filter_shape5 = [3, 3, 128,64]    #filter_shape5 = [3, 3, 32, 32]
        filter_shape6 = [3, 3, 64, 1]     #filter_shape6 = [5, 5, 32, 1]
               
        # Define weights and biases for the model. Fixed filter shapes
        weight1 = weight_variables(filter_shape1,std,name='w1')
        weight2 = weight_variables(filter_shape2,std,name='w2')
        weight3 = weight_variables(filter_shape3,std,name='w3')
        weight4 = weight_variables(filter_shape4,std,name='w4')
        weight5 = weight_variables(filter_shape5,std,name='w5')
        weight6 = weight_variables(filter_shape6,std,name='w6')
        # Define weights and biases for the model. Fixed filter shapes        
        bias1 = bias_variables(filter_shape=64,    name='b1')
        bias2 = bias_variables(filter_shape=64, name='b2')
        bias3 = bias_variables(filter_shape=1,     name='b3')
        bias4 = bias_variables(filter_shape=128,   name='b4')
        bias5 = bias_variables(filter_shape=64,    name='b5')
        bias6 = bias_variables(filter_shape=1,     name='b6')
        
        params = {'W1': weight1,
                  'W2': weight2,
                  'W3': weight3,
                  'W4': weight4,
                  'W5': weight5,
                  'W6': weight6,
                  'B1': bias1,
                  'B2': bias2,
                  'B3': bias3,
                  'B4': bias4,
                  'B5': bias5,
                  'B6': bias6
                }
    
        # Define convolutions/deconvolutions and activations
        a1    = X
        conv1 = tf.nn.conv2d(a1, weight1, strides=[1,1,1,1], padding='SAME') + bias1        
        #a2    = tf.nn.elu(conv1)
        a2    = tf.nn.leaky_relu(conv1,alpha=0.5)
        #a2    = tf.nn.tanh(conv1)
        conv2 = tf.nn.conv2d(a2, weight2, strides=[1,1,1,1], padding='SAME') + bias2        
        #a3    = tf.nn.elu(conv2)
        a3    = tf.nn.leaky_relu(conv2,alpha=0.5)
        #a3    = tf.nn.tanh(conv2)
        conv3 = tf.nn.conv2d(a3, weight3, strides=[1,1,1,1], padding='SAME') + bias3
        #a4    = tf.nn.elu(conv3)
        a4    = tf.nn.leaky_relu(conv3,alpha=0.5)
        #a4    = tf.nn.tanh(conv3)
        subpx = tf.contrib.periodic_resample.periodic_resample(a4, [shape[0],samp_hry, samp_hrx, None])
        #a5    = tf.nn.elu(conv4)
        a5    = tf.nn.leaky_relu(subpx,alpha=0.5)
        #a5    = tf.nn.tanh(conv4)
        conv4 = tf.nn.conv2d(a5, weight4, strides=[1,1,1,1], padding='SAME') + bias4
        #a6    = tf.nn.elu(conv5)
        a6    = tf.nn.leaky_relu(conv4,alpha=0.5)
        #a6    = tf.nn.tanh(conv5)
        conv5 = tf.nn.conv2d(a6, weight5, strides=[1,1,1,1], padding='SAME') + bias5
        #a6    = tf.nn.elu(conv5)
        a7    = tf.nn.leaky_relu(conv5,alpha=0.5)        
           
        pred  = tf.nn.conv2d(a7, weight6, strides=[1,1,1,1], padding='SAME') + bias6
        
        return pred, params
    
def Interpolation_model(x, shape, ratio, params, samp_hry, samp_hrx, samp_lr):
        
        # Define convolutions/deconvolutions and activations
        a1    = x
        conv1 = tf.nn.conv2d(a1, params['W1'], strides=[1,1,1,1], padding='SAME') + params['B1']        
        #a2    = tf.nn.elu(conv1)
        a2    = tf.nn.leaky_relu(conv1,alpha=0.5)
        #a2    = tf.nn.tanh(conv1)
        conv2 = tf.nn.conv2d(a2, params['W2'], strides=[1,1,1,1], padding='SAME') + params['B2']        
        #a3    = tf.nn.elu(conv2)
        a3    = tf.nn.leaky_relu(conv2,alpha=0.5)
        #a3    = tf.nn.tanh(conv2)
        conv3 = tf.nn.conv2d(a3, params['W3'], strides=[1,1,1,1], padding='SAME') + params['B3']
        #a4    = tf.nn.elu(conv3)
        a4    = tf.nn.leaky_relu(conv3,alpha=0.5)
        #a4    = tf.nn.tanh(conv3)
        subpx = tf.contrib.periodic_resample.periodic_resample(a4, [shape[0],samp_hry, samp_hrx, None])
        #a5    = tf.nn.elu(conv4)
        a5    = tf.nn.leaky_relu(subpx,alpha=0.5)
        #a5    = tf.nn.tanh(conv4)
        conv4 = tf.nn.conv2d(a5, params['W4'], strides=[1,1,1,1], padding='SAME') + params['B4']
        #a6    = tf.nn.elu(conv5)
        a6    = tf.nn.leaky_relu(conv4,alpha=0.5)
        #a6    = tf.nn.tanh(conv5)
        conv5 = tf.nn.conv2d(a6, params['W5'], strides=[1,1,1,1], padding='SAME') + params['B5']
        #a6    = tf.nn.elu(conv5)
        a7    = tf.nn.leaky_relu(conv5,alpha=0.5)        
           
        pred  = tf.nn.conv2d(a7, params['W6'], strides=[1,1,1,1], padding='SAME') + params['B6']
    
        return pred
    
#======================================================================================================        
# Define tensorflow loss functions   
def create_loss(X,pred):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(X - pred))
        return loss
    
def create_loss_l1(X,pred):
    with tf.name_scope('loss'):
        loss = tf.losses.absolute_difference(X,pred)
        return loss


def create_huber_loss(X,pred,Delta=1.0):
    with tf.name_scope('loss'):
        loss = tf.losses.huber_loss(X,pred,delta=Delta)
        return loss
    
#======================================================================================================        
# Define tensorflow optimizers    
def create_optimizer(loss,eta = 0.001):
    with tf.name_scope('optimizer'):
        training = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)
        return training
        
def create_Adagrad_optimizer(loss, eta = 0.001):
    with tf.name_scope('optimizer'):
        training = tf.train.AdagradOptimizer(learning_rate=eta).minimize(loss)
        return training
    
def create_Adadelta_optimizer(loss, eta = 0.001):
    with tf.name_scope('optimizer'):
        training = tf.train.AdadeltaOptimizer(learning_rate=eta).minimize(loss)
        return training    