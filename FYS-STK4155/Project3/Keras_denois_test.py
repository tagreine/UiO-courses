# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:11:58 2018

@author: tagreine
"""

from keras.datasets import mnist
import keras
import numpy as np
import matplotlib.pyplot as plt


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
noise_scal = 0.5
x_train_noisy = x_train + noise_scal * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy  = x_test  + noise_scal * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)  

#x_train_noisy = np.clip(x_train_noisy, 0., 1.)
#x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# plot the noisy images
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_train_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#============================================================================================================

#===================================== Building convolutional autoencoder ===================================
x_train       = np.reshape(x_train, (len(x_train), 28, 28, 1))  
x_train_noisy = np.reshape(x_train_noisy, (len(x_train), 28, 28, 1))
x_test        = np.reshape(x_test, (len(x_test), 28, 28, 1))  
x_test_noisy  = np.reshape(x_test_noisy, (len(x_test), 28, 28, 1))


input_image = keras.Input(shape=(28,28,1))

# Create the encoder
x       = keras.layers.Conv2D(16, (3,3), activation = 'relu', padding = 'same')(input_image)    #(28x28x1)
x       = keras.layers.MaxPool2D((2, 2), padding='same')(x)                                     #(14x14x1)
x       = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)                 #(14x14x1)                                    
x       = keras.layers.MaxPool2D((2, 2), padding='same')(x)                                     #(7x7x1)
encoded = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)                 #(7x7x1)

# Create the decoder
x       = keras.layers.Deconv2D(64, (3, 3), activation='relu', padding='same')(encoded)         #(7x7x1)
x       = keras.layers.UpSampling2D((2, 2))(x)                                                  #(14x14x1) 
x       = keras.layers.Deconv2D(32, (3, 3), activation='relu', padding='same')(x)               #(14x14x1)
x       = keras.layers.UpSampling2D((2, 2))(x)                                                  #(28x28x1)
decoded = keras.layers.Deconv2D(1, (3, 3), activation='relu', padding='same')(x)                #(28x28x1)

autoencoder = keras.Model(input_image, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


x_train1        = x_train[0:1000,:,:,:]
x_train1_noisy  = x_train_noisy[0:1000,:,:,:]

x_test1        = x_test[0:500,:,:,:]
x_test1_noisy  = x_test_noisy[0:500,:,:,:]

autoencoder.fit(x_train1, x_train1,
                epochs=20,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test1, x_test1))


decoded_imgs = autoencoder.predict(x_train1_noisy[0:100,:,:,:])

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test1_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#============================================================================================================
#====================================Test trained function on coherent noise=================================

I = np.identity(28).reshape(1,28,28,1)

test = x_test1_noisy + I

decoded_imgs_I = autoencoder.predict(test[0:100,:,:,:])


n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(decoded_imgs_I[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()























