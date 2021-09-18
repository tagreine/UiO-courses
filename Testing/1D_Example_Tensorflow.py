# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 08:31:18 2019

@author: tlgreiner
"""

'Mexican hat wavelet'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Functions import Mexican_Hat, sigmoid, tanh, Relu, leakyRelu
from sklearn.preprocessing import normalize

sigma = 1.5

t   = np.expand_dims(np.arange(-8,8,0.1),axis=1)

# Function to approximate
Ricker = Mexican_Hat(t,sigma)

fig, ax = plt.subplots()
ax.plot(t,Ricker, 'k',lw=2)
ax.set(xlabel='time (s)', ylabel='Amplitude', title='Function approximation', ylim=(-0.8,1.2), xlim=(-8,8))

ax.grid()
plt.savefig('Ricker.png',dpi=600)
plt.show()


#===================== activation functions ===============================================
# Sigmoid activation function
sigmo = sigmoid(t)

fig, ax = plt.subplots()
ax.plot(t,sigmo, 'r',lw=1)
ax.set(title='Sigmoid function', ylim=(-0.1,1.1), xlim=(-8,8))

ax.grid()
plt.savefig('Sigmoid.png',dpi=600)
plt.show()

# tanh activation function
tan = tanh(t)

fig, ax = plt.subplots()
ax.plot(t,tan, 'r',lw=1)
ax.set(title='Tanh function', ylim=(-1.1,1.1), xlim=(-8,8))

ax.grid()
plt.savefig('tanh.png',dpi=600)
plt.show()

# Relu activation function
relu = Relu(t)

fig, ax = plt.subplots()
ax.plot(t,relu, 'r',lw=1)
ax.set(title='ReLU function', ylim=(-0.1,1.1), xlim=(-2,2))

ax.grid()
plt.savefig('ReLU.png',dpi=600)
plt.show()

# leakyRelu activation function
lrelu = leakyRelu(t,0.1)

fig, ax = plt.subplots()
ax.plot(t,lrelu, 'r',lw=1)
ax.set(title='leaky ReLU function', ylim=(-0.3,1.1), xlim=(-2,2))

ax.grid()
plt.savefig('lReLU.png',dpi=600)
plt.show()


# ================================ Function fitting examples ============================



# ============================ Sigmoid function =========================================


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
   
   return out, w1, w2, b1, b2
    
tt = normalize(np.c_[t,t**2,t**3])




num_nodes = 3
num_epoch = 10000


with tf.Session() as sess:
    
    x = tf.placeholder(dtype=tf.float32,shape=(tt.shape[1],len(t)))
    y = tf.placeholder(dtype=tf.float32,shape=(1,len(t)))
    
    pred, w1, w2, b1, b2 = nn_model(x, tt.shape[1] , num_nodes,std=0.01, activation = 'leakyRelu')
    
    loss  = tf.reduce_mean(tf.square(y - pred))
    
    training = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
    tf.global_variables_initializer().run()

    for epoch in range(num_epoch):
        
        train = sess.run([training], feed_dict = {x: tt.T, y: Ricker.T})
    
    out_train, w1, w2, b1, b2 = np.squeeze(sess.run([pred, w1, w2, b1, b2], feed_dict = {x: tt.T}))

plt.plot(out_train.T)














