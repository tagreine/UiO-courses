# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:21:01 2019

@author: tlgreiner
"""

import numpy as np

def Mexican_Hat(t,sigma):
    # t     = time vector
    # sigma = standard deviation 
    
    a  = np.divide(2,np.sqrt(np.multiply(3,sigma)))
    b  = (1 - (t/sigma)**2)
    c  = -np.divide(t**2,2*sigma**2)
    
    MH = a*b*np.exp(c)
    
    return MH

def sigmoid(z):
    # Sigmoid activation function
    return np.divide(1,1 + np.exp(-z))

def Relu(z):
    # Rectified linear unit
    relu = np.maximum(0,z)
    return relu

def tanh(z):
    # The tangens hyperbolicus activation function
    tan = np.tanh(z)
    return tan 


def leakyRelu(z,eta):
    # Rectified linear unit
    lrelu = np.maximum(eta*z,z)
    return lrelu

def nn_model(X, num_features=1, num_nodes=3, std=0.1, activation='sigmoid'):
   tf.random.set_random_seed(1)
   w1 = tf.Variable(tf.truncated_normal(shape=(num_nodes,num_features), stddev=std))
   b1 = tf.Variable(tf.zeros(1,1)) 
   
   w2 = tf.Variable(tf.random_normal(shape=(1,num_nodes), stddev=std))
   b2 = tf.Variable(tf.zeros(1,1))

   a1  = X
   
   if activation == 'sigmoid':
       a2  = tf.sigmoid(tf.add(tf.matmul(w1,a1),b1))   
   if activation == 'tanh':
       a2  = tf.math.tanh(tf.add(tf.matmul(w1,a1),b1))
   if activation == 'Relu':
       a2  = tf.nn.relu(tf.add(tf.matmul(w1,a1),b1))
   if activation == 'leakyRelu':
       a2  = tf.nn.leaky_relu(tf.add(tf.matmul(w1,a1),b1))   
    
    
   out = tf.add(tf.matmul(w2,a2),b2)
   
   return out, w1, w2, b1, b2, a2

def Function_fitting(X, Y, num_nodes=100,num_epoch = 10000, std=0.01, activation = 'Relu'):
    with tf.Session() as sess:
    
        x = tf.placeholder(dtype=tf.float32,shape=(X.shape[1],len(X)))
        y = tf.placeholder(dtype=tf.float32,shape=(1,len(X)))
    
        pred, w1, w2, b1, b2, a2 = nn_model(x, X.shape[1] , num_nodes=num_nodes,std=std, activation = activation)
    
        loss  = tf.reduce_mean(tf.square(y - pred))
    
        training = tf.train.AdamOptimizer(0.001).minimize(loss)
    
        tf.global_variables_initializer().run()

        for epoch in range(num_epoch):
        
            train = sess.run([training], feed_dict = {x: X.T, y: Y.T})
    
        out_train, w1, w2, b1, b2, a2 = np.squeeze(sess.run([pred, w1, w2, b1, b2, a2], feed_dict = {x: X.T}))

    return out_train, w1, w2, b1, b2, a2


def Fourier_basis(X,basis=2):
    
    fb = np.zeros((len(X),basis))

    for i in range(basis):
        fb[:,i] = np.sqrt(2)*np.sin(2*math.pi*(i+1)*X)
    
    return fb   
