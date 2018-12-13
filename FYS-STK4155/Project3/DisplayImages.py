# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:30:17 2018

@author: tagreine
"""
import matplotlib.pyplot as plt
import numpy as np

def display_gathers(target_img, input_img, output_img, a=200, b=100, ratio = 4,n=1,m=0,c=0.05,save='True',name='Interpolated_shot_test_set5.png'):
    plt.figure(figsize=(20, 10))
    for i in range(n):
    
        # display original
        ax = plt.subplot(n, 4, i + 1)
        plt.imshow(target_img[i+m].reshape(a,b),aspect="auto")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Target')
        plt.colorbar()
        plt.clim(-c,c)
    
        # display input
        ax = plt.subplot(n, 4, i + 1 + n)
        plt.imshow(input_img[i+m].reshape(a, np.int(b/ratio)),aspect="auto")
        #input_int = resize(input_test[i,:,:,0], (samp_hr,samp_hr), order=5)
        #plt.imshow(input_int,aspect="auto")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Input')
        plt.colorbar()
        plt.clim(-c,c)

        # display reconstruction
        ax = plt.subplot(n, 4, i + 1 + 2*n)
        plt.imshow(output_img[i+m].reshape(a,b),aspect="auto")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Predicted')
        plt.colorbar()
        plt.clim(-c,c)
    
        # display reconstruction
        ax = plt.subplot(n, 4, i + 1 + 3*n)
        plt.imshow((target_img[i+m].reshape(a,b) - output_img[i+m].reshape(a,b)),aspect="auto")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Truth-Predicted')
        plt.colorbar()
        plt.clim(-c,c)
        
        if save=='True':
            plt.savefig(name, dpi=600)
        if save=='False':
            plt.show()
            
            
            
def display_training_im(target_img, input_img, a=200, b=100, ratio = 4,n=1,c=0.05,save='True',name='Targets_Inputs.png'):

    plt.figure(figsize=(20, 10))
    for i in range(n):
        
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(target_img[i+2].reshape(a, b),aspect="auto")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Target')    
        plt.colorbar()
        plt.clim(-c,c)

        # display input
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(input_img[i+2].reshape(a, np.int(b/ratio)),aspect="auto")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Input')    
        plt.colorbar()
        plt.clim(-c,c)
     
    if save == 'True':
        plt.savefig(name, dpi=600)
    if save == 'False':
        plt.show()
        
        
def plotCNNFilter(units,fsize=5,save='True',name='Trained_weights_subpix.png'):
    filters = units.shape[2]
    plt.figure(1, figsize=(10,10))
    n_columns = 8
    n_rows    = 8
    for i in range(filters):
        ax = plt.subplot(n_rows, n_columns, i+1)
        plt.imshow(units[:,:,i,:].reshape(fsize,fsize), interpolation="bicubic", cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        plt.subplots_adjust(hspace=0.1,wspace=0.01)
    if save =='True':
        plt.savefig(name, dpi=600)
    if save =='False':
        plt.show()
