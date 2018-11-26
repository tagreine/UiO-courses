# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:27:48 2018

@author: tlgreiner
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
from scipy.misc import imresize
import tensorflow as tf
import numpy as np



(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32') / 255.

X_train = x_train[0:10,:,:]
X_test  = x_test[0:5,:,:]


# Define target values
label_     = X_train
label_test = X_test
label_     = np.expand_dims(label_,axis=3)
label_test = np.expand_dims(label_test,axis=3)
ratio   = 4
shape   = label_.shape
# Define input values
input_ = label_[:,:,::ratio,:]

#============Plot target and inputs======= 
plt.imshow(label_[5,:,:,0])
plt.imshow(input_[5,:,:,0])

# Define the high resolution (hr) and low resolution samples (lr) samples
samp_hr = np.int(shape[1])
samp_lr = np.int(shape[1]/ratio)

#=================================================================================
# Define placeholders for inputs and labels
x = tf.placeholder(tf.float32, [shape[0], samp_hr, samp_lr, 1], name='inputs')
y = tf.placeholder(tf.float32, [shape[0], samp_hr, samp_hr, 1], name='labels')

Y  = label_
X  = input_
Xt = label_test

with tf.Session() as sess:
        

        weight1 = tf.Variable(tf.random_normal([5, 5, 1, 64], stddev=0.2), name='w1')       #(9x9x1 filter size, 64 filters)
        weight2 = tf.Variable(tf.random_normal([3, 3, 64, ratio], stddev=0.2), name='w2')   #(3x3x1 filter size, ratio filters)
        weight3 = tf.Variable(tf.random_normal([3, 3, 1, ratio], stddev=0.2), name='w3')   #(9x9x1 filter size, ratio filters)
        weight4 = tf.Variable(tf.random_normal([3, 3, 1, 128], stddev=0.2), name='w4')   #(9x9x1 filter size, ratio filters)
        weight5 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev=0.2), name='w5')   #(9x9x1 filter size, ratio filters)
        weight6 = tf.Variable(tf.random_normal([3, 3, 64, 1], stddev=0.2), name='w6')   #(9x9x1 filter size, ratio filters)

        
        bias1   =  tf.Variable(tf.zeros([64], name='b1'))
        bias2   =  tf.Variable(tf.zeros([ratio], name='b2'))
        bias3   =  tf.Variable(tf.zeros([1], name='b3'))
        bias4   =  tf.Variable(tf.zeros([128], name='b4'))
        bias5   =  tf.Variable(tf.zeros([64], name='b5'))        
        bias6   =  tf.Variable(tf.zeros([1], name='b6'))


        conv1 = tf.nn.relu(tf.nn.conv2d(x, weight1, strides=[1,1,1,1], padding='SAME') + bias1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weight2, strides=[1,1,1,1], padding='SAME') + bias2)
        conv3 = tf.nn.relu(tf.nn.conv2d_transpose(conv2, weight3, output_shape=(shape[0],samp_hr,samp_hr,1), strides=[1,1,4,1], padding='SAME') + bias3)
        conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weight4, strides=[1,1,1,1], padding='SAME') + bias4)
        conv5 = tf.nn.relu(tf.nn.conv2d(conv4, weight5, strides=[1,1,1,1], padding='SAME') + bias5)
        pred  = tf.nn.conv2d(conv5, weight6, strides=[1,1,1,1], padding='SAME') + bias6
        
    # Loss function with L2 Regularization with beta=0.01
        regularizers = tf.nn.l2_loss(weight1) + tf.nn.l2_loss(weight2)+ tf.nn.l2_loss(weight3) + tf.nn.l2_loss(weight4) + tf.nn.l2_loss(weight5) + tf.nn.l2_loss(weight6)
        Sq       = tf.square(y - pred)
        loss     = tf.reduce_mean(Sq)
        loss     = loss + 0.01 * regularizers 
        train_op = tf.train.AdamOptimizer()
        training = train_op.minimize(loss)
        tf.global_variables_initializer().run()
        
        
        #test = sess.run(Sq, feed_dict={x: X, y: Y})
        #y11 = sess.run(conv1, feed_dict={x: X})
        #y22 = sess.run(conv2, feed_dict={x: X})
        #y33 = sess.run(conv3, feed_dict={x: X})
        #y44 = sess.run(conv4, feed_dict={x: X})
        #y55 = sess.run(conv5, feed_dict={x: X})
        #yt  = sess.run(pred, feed_dict={x: X})
        
        #y5, y6 = sess.run([training, loss], feed_dict={x: X, y: Y})
        
        
        hm_epochs  = 1000
        epoch_loss = 0
        
        
        for epoch in range(hm_epochs):
            
            test, c = sess.run([training, loss], feed_dict = {x: X, y: Y})
            epoch_loss = c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
               
                
        test = np.squeeze(sess.run(pred, feed_dict = {x: X}))
        #test_test = np.squeeze(sess.run(pred, feed_dict = {x: Xt}))