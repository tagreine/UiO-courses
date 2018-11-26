# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:33:08 2018

@author: tagreine
"""


from keras.datasets import mnist
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

#===================================== Extract training data ================================================
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize the images

x_train = x_train.astype('float32')/255
x_test  = x_test.astype('float32')/255

# plot the training and test images
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#============================================================================================================

#===================================== Make noisy images from the clean =====================================

# Include random noise
#noise_scal = 0.3
#x_train_noisy = x_train + noise_scal * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
#x_test_noisy  = x_test  + noise_scal * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)  

#x_train_noisy = np.clip(x_train_noisy, 0., 1.)
#x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# plot the noisy images
#n = 10
#plt.figure(figsize=(20, 2))
#for i in range(n):
#    ax = plt.subplot(1, n, i+1)
#    plt.imshow(x_train[i].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()

#n = 10
#plt.figure(figsize=(20, 2))
#for i in range(n):
#    ax = plt.subplot(1, n, i+1)
#    plt.imshow(x_test[i].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#   ax.get_yaxis().set_visible(False)
#plt.show()

#============================================================================================================

#===================================== Building convolutional interpolator ==================================

# Extract a smaller subsample of the training and test set
x_train = x_train[0:1000,:,:]
x_test  = x_test[0:500,:,:,]

# Decimate

k,l,m = x_train.shape
x_train_dec = 0*np.arange(k*l*(m/2)).reshape(k,l,int(m/2))
for i in range(k):
      for j in range(int(m/2)):     
          x_train_dec[i,:,j] = x_train[i,:,2*j] 
          
k,l,m = x_test.shape
x_test_dec  = 0*np.arange(k*l*(m/2)).reshape(k,l,int(m/2))
for i in range(k):
      for j in range(int(m/2)):     
          x_test_dec[i,:,j] = x_test[i,:,2*j] 

    
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_train_dec[i].reshape(28, 14))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# Reshape for input to convolutions
x_train     = np.reshape(x_train, (len(x_train), 28, 28, 1))  
x_train_dec = np.reshape(x_train_dec, (len(x_train), 28, 14, 1))
x_test      = np.reshape(x_test, (len(x_test), 28, 28, 1))  
x_test_dec  = np.reshape(x_test_dec, (len(x_test), 28, 14, 1))


input_image = keras.Input(shape=(28,14,1))

# Create the encoder
x       = keras.layers.Conv2D(16, (5,5), activation = 'relu', padding = 'same')(input_image)    #(28x14x1)
x       = keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)                 #(28x14x1)
x       = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)                 #(28x14x1)
x       = keras.layers.MaxPool2D((2, 1), padding='same')(x)                                     #(14x14x1)
encoded = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)                #(14x14x1)

# Create the decoder
x       = keras.layers.Deconv2D(128, (3, 3), activation='relu', padding='same')(encoded)        #(14x14x1)
x       = keras.layers.Deconv2D(32, (3, 3), activation='relu', padding='same')(x)               #(14x14x1)
x       = keras.layers.Deconv2D(16, (3, 3), activation='relu', padding='same')(x)               #(14x14x1)
x       = keras.layers.UpSampling2D((2, 2))(x)                                                  #(28x28x1)
decoded = keras.layers.Deconv2D(1, (5, 5), activation='sigmoid', padding='same')(x)             #(28x28x1)

autoencoder = keras.Model(input_image, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#tensorboard --logdir=/tmp/autoencoder


autoencoder.fit(x_train_dec,x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_dec,x_test))


interpolated_imgs = autoencoder.predict(x_test_dec[:,:,:,:])

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(interpolated_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

