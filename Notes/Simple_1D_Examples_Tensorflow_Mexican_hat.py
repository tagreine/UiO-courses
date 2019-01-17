# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:55:21 2019

@author: tagreine
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Functions import Mexican_Hat, sigmoid, tanh, Relu, leakyRelu, nn_model, Function_fitting
from sklearn.preprocessing import normalize

sigma = 1.5

X   = np.expand_dims(np.arange(-8,8,0.1),axis=1)

# Function to approximate
Ricker = Mexican_Hat(X,sigma)

fig, ax = plt.subplots()
ax.plot(X,Ricker, 'k',lw=4)
ax.set(xlabel='time (s)', ylabel='Amplitude', title='Function approximation', ylim=(-0.8,1.2), xlim=(-8,8))

ax.grid()
#plt.savefig('Ricker.png',dpi=600)
plt.show()


#===================== activation functions ===============================================
# Sigmoid activation function
sigmo = sigmoid(X)

fig, ax = plt.subplots()
ax.plot(X,sigmo, 'r',lw=4)
ax.set(title='Sigmoid function', ylim=(-0.1,1.1), xlim=(-8,8))

ax.grid()
plt.savefig('Sigmoid.png',dpi=600)
plt.show()

# tanh activation function
tan = tanh(X)

fig, ax = plt.subplots()
ax.plot(X,tan, 'r',lw=4)
ax.set(title='Tanh function', ylim=(-1.1,1.1), xlim=(-8,8))

ax.grid()
plt.savefig('tanh.png',dpi=600)
plt.show()

# Relu activation function
relu = Relu(X)

fig, ax = plt.subplots()
ax.plot(X,relu, 'r',lw=4)
ax.set(title='ReLU function', ylim=(-0.1,1.1), xlim=(-2,2))

ax.grid()
plt.savefig('ReLU.png',dpi=600)
plt.show()

# leakyRelu activation function
lrelu = leakyRelu(X,0.1)

fig, ax = plt.subplots()
ax.plot(X,lrelu, 'r',lw=4)
ax.set(title='leaky ReLU function', ylim=(-0.3,1.1), xlim=(-2,2))

ax.grid()
plt.savefig('lReLU.png',dpi=600)
plt.show()


#########################################################################################################################################

                                                ###############Node Testing##############

#########################################################################################################################################


#####################Function approximation: Nodes = 3, Feature = 1#######################################

n = 3
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(X, Ricker, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(X, Ricker, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(X, Ricker, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Ricker,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 3 Features = 1 Standard dev = 0.1', xlim=(-8,8))

ax.grid()
plt.legend(('Ricker','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()

#####################Function approximation: Nodes = 10, Feature = 1#######################################

n = 10
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(X, Ricker, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(X, Ricker, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(X, Ricker, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Ricker,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 10 Features = 1 Standard dev = 0.1', xlim=(-8,8))

ax.grid()
plt.legend(('Ricker','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()

