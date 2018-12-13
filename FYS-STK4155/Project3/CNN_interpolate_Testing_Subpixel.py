# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 07:24:01 2018

@author: tagreine
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
from scipy.misc import imresize
import tensorflow as tf
import numpy as np



(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32') / 255.

X_train = x_train[0:500,:,:]
X_test  = x_test[0:500,:,:]


# Define target values
label_  = X_train
label_  = np.expand_dims(label_,axis=3)
label_t = X_test
label_t  = np.expand_dims(label_t,axis=3)
ratio   = 4
shape   = label_.shape
# Define input values
input_     = label_[:,:,::ratio,:]
input_test = X_test[:,:,::ratio]
input_test = np.expand_dims(input_test,axis=3)
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
yt= tf.placeholder(tf.float32, [shape[0], samp_hr, samp_hr, 1], name='labels_test')
xt= tf.placeholder(tf.float32, [shape[0], samp_hr, samp_lr, 1], name='inputs_test')

Y  = label_
Yt = label_t
X  = input_
Xt = input_test

with tf.Session() as sess:
        

        weight1 = tf.Variable(tf.truncated_normal([7, 7, 1, 64],     stddev=0.1), name='w1')
        weight2 = tf.Variable(tf.truncated_normal([3, 3, 64, 32],    stddev=0.1), name='w2')
        weight3 = tf.Variable(tf.truncated_normal([3, 3, 32, 32],    stddev=0.1), name='w3')        
        weight4 = tf.Variable(tf.truncated_normal([3, 3, 32, ratio], stddev=0.1), name='w4')  
        #weight5 = tf.Variable(tf.truncated_normal([5, 5, 1, ratio],  stddev=0.1), name='w5')   

       
        bias1   =  tf.Variable(tf.zeros([64],    name='b1'))
        bias2   =  tf.Variable(tf.zeros([32],    name='b2'))
        bias3   =  tf.Variable(tf.zeros([32],    name='b3'))
        bias4   =  tf.Variable(tf.zeros([ratio], name='b4'))
        #bias5   =  tf.Variable(tf.zeros([ratio], name='b5'))


        # Training
        conv1 = tf.nn.relu(tf.nn.conv2d(x, weight1, strides=[1,1,1,1],     padding='SAME') + bias1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weight2, strides=[1,1,1,1], padding='SAME') + bias2)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weight3, strides=[1,1,1,1], padding='SAME') + bias3)
        conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weight4, strides=[1,1,1,1], padding='SAME') + bias4)
        #conv5 = tf.nn.relu(tf.nn.conv2d(conv4, weight5, strides=[1,1,1,1], padding='SAME') + bias5)
        
        
        pred  = tf.contrib.periodic_resample.periodic_resample(conv4, [shape[0],samp_hr, samp_hr, None]) 
        
        Sq       = tf.square(y - pred)
        loss     = tf.reduce_mean(Sq)
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
        
        
        hm_epochs  = 400
        epoch_loss = 0
        
        
        for epoch in range(hm_epochs):
            
            test, c = sess.run([training, loss], feed_dict = {x: X, y: Y})
            epoch_loss = c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
               
                
        test = np.squeeze(sess.run(pred, feed_dict = {x: X}))
        

# Display interpolated training data
n = 10  # how many digits we will display
plt.figure(figsize=(10, 4))
for i in range(n):
    
    # display original
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(label_[i].reshape(28, 28),aspect="auto")
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