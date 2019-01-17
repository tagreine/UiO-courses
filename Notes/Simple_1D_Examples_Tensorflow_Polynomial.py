# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:38:03 2019

@author: tagreine
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Functions import nn_model
from sklearn.preprocessing import normalize


X   = np.expand_dims(np.arange(-2,2,0.1),axis=1)

# Function to approximate
Y = X**3 + X**2 - X -2 

fig, ax = plt.subplots()
ax.plot(X,Y, 'k',lw=4)
ax.set(xlabel='time (s)', ylabel='Amplitude', title='Function approximation', xlim=(-2,2))

ax.grid()
#plt.savefig('Ricker.png',dpi=600)
plt.show()



# ================================ Function fitting examples ============================



# ============================ Sigmoid function =========================================







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


#########################################################################################################################################

                                                ###############Node Testing##############

#########################################################################################################################################


#####################Function approximation: Nodes = 3, Feature = 1#######################################

n = 3
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Y,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 3 Features = 1 Standard dev = 0.1', xlim=(-2,2))

ax.grid()
plt.legend(('X**3 + X**2 - X -2','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()


#####################Function approximation: Nodes = 10, Feature = 1#######################################

n = 10
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Y,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 10 Features = 1 Standard dev = 0.1', xlim=(-2,2))

ax.grid()
plt.legend(('X**3 + X**2 - X -2','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()


#####################Function approximation: Nodes = 50, Feature = 1#######################################

n = 50
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Y,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 50 Features = 1 Standard dev = 0.1', xlim=(-2,2))

ax.grid()
plt.legend(('X**3 + X**2 - X -2','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()

#####################Function approximation: Nodes = 100, Feature = 1#######################################

n = 100
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Y,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 100 Features = 1 Standard dev = 0.1', xlim=(-2,2))

ax.grid()
plt.legend(('X**3 + X**2 - X -2','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()

#####################Function approximation: Nodes = 200, Feature = 1#######################################

n = 200
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Y,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 200 Features = 1 Standard dev = 0.1', xlim=(-2,2))

ax.grid()
plt.legend(('X**3 + X**2 - X -2','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()


#####################Function approximation: Nodes = 400, Feature = 1#######################################

n = 400
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(X, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Y,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 400 Features = 1 Standard dev = 0.1', xlim=(-2,2))

ax.grid()
plt.legend(('X**3 + X**2 - X -2','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()

#########################################################################################################################################

                                                ###############Feature Testing##############

#########################################################################################################################################

XX = np.c_[X,X**2]

#####################Function approximation: Nodes = 3, Feature = 2#######################################

n = 3
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Y,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 3 Features = 2 Standard dev = 0.1', xlim=(-2,2))

ax.grid()
plt.legend(('X**3 + X**2 - X -2','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()

#####################Function approximation: Nodes = 10, Feature = 2#######################################

n = 10
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Y,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 10 Features = 2 Standard dev = 0.1', xlim=(-2,2))

ax.grid()
plt.legend((r'$X^3 + X^2 - X -2$','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()

#####################Function approximation: Nodes = 50, Feature = 2#######################################

n = 50
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Y,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 50 Features = 2 Standard dev = 0.1', xlim=(-2,2))

ax.grid()
plt.legend((r'$X^3 + X^2 - X -2$','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()

#########################################################################################################################################

XX = np.c_[X,X**2,X**3]

#####################Function approximation: Nodes = 3, Feature = 3#######################################

n = 3
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Y,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 3 Features = 3 Standard dev = 0.1', xlim=(-2,2))

ax.grid()
plt.legend(('X**3 + X**2 - X -2','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()

#####################Function approximation: Nodes = 10, Feature = 3#######################################

n = 10
s = 0.1

out_train0, w10, w20, b10, b20, a20 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'Relu')
out_train1, w11, w21, b11, b21, a21 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'sigmoid')
out_train2, w12, w22, b12, b22, a22 = Function_fitting(XX, Y, num_nodes=n, num_epoch = 10000,std=s, activation = 'tanh')


fig, ax = plt.subplots()
ax.plot(X,Y,'k', X, out_train0.T,'--r', X, out_train1.T,'--*b',X, out_train2.T,'--g',lw=4)

ax.set(xlabel='time (s)', ylabel='Amplitude', title='Nodes = 10 Features = 3 Standard dev = 0.1', xlim=(-2,2))

ax.grid()
plt.legend(('X**3 + X**2 - X -2','Relu','Sigmoid','Tanh'))
#plt.savefig('Ricker.png',dpi=600)
plt.show()




