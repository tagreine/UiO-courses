# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:38:21 2021

@author: tlgreiner
"""

import numpy as np
import tensorflow as tf

def _upsampling(a,weight,num_ch=128,stride=2):
         
    d = a.shape[1].value*stride
    h = a.shape[2].value*stride
    w = a.shape[3].value*stride
              
    new_shape = [a.shape[0].value,d,h,w,num_ch]
    output_shape = tf.stack(new_shape)
    
    conv = tf.nn.conv3d_transpose(a,weight,output_shape=output_shape,strides=(1,stride,stride,stride,1),padding='SAME')

    return conv

def Haar_framelets():
    phi1 = 1/np.sqrt(8)*np.array([1,1,1,1,1,1,1,1]).reshape(1,8)     # approximation
    phi2 = 1/np.sqrt(8)*np.array([1,1,1,1,-1,-1,-1,-1]).reshape(1,8) # vertical detail in xline direction
    phi3 = 1/np.sqrt(8)*np.array([1,-1,1,-1,1,-1,1,-1]).reshape(1,8) # vertical detail in inline direction
    phi4 = 1/np.sqrt(8)*np.array([1,1,-1,-1,1,1,-1,-1]).reshape(1,8) # horizontal detail
    phi5 = 1/np.sqrt(8)*np.array([-1,1,1,-1,-1,1,1,-1]).reshape(1,8) # diag-horiz detail in inline direction
    phi6 = 1/np.sqrt(8)*np.array([-1,-1,1,1,1,1,-1,-1]).reshape(1,8) # diag-horiz detail in xline direction
    phi7 = 1/np.sqrt(8)*np.array([-1,1,-1,1,1,-1,1,-1]).reshape(1,8) # diag-vertical detail in xline direction
    phi8 = 1/np.sqrt(8)*np.array([-1,1,1,-1,1,-1,-1,1]).reshape(1,8) # diag-diag detail in inine direction


    phiT = np.concatenate((phi1,phi2,phi3,phi4,phi5,phi6,phi7,phi8),axis=0)

    analysis  = np.zeros([2,2,2,1,8]) # for convolution in tensorflow
    synthesis = np.zeros([2,2,2,1,8]) # for transposed convolution in tensorflow
    for i in range(8):
        analysis[:,:,:,0,i]  = phiT[i,:].reshape(2,2,2)
        synthesis[:,:,:,0,i] = phiT.T[:,i].reshape(2,2,2) #

    return analysis,synthesis


class GENPOOL3D:
    def __init__(self):
        
        analysis,synthesis = Haar_framelets()        
        self.analysis  = tf.constant(analysis,dtype=tf.float32)
        self.synthesis = tf.constant(synthesis,dtype=tf.float32)
        
    def wave_dec(self,x):
        
        shape = [i.value for i in x.get_shape()]
        num_features = shape[4]
              
    
        for i in range(num_features):
            
            xx = tf.nn.conv3d(x[:,:,:,:,i:(i+1)],self.analysis,strides=[1,2,2,2,1],padding='SAME')
            
            if i==0:    
                low_pass  = xx[:,:,:,:,0:1]
                high_pass = xx[:,:,:,:,1:]
            
            else:
                low_pass  = tf.concat((low_pass,xx[:,:,:,:,0:1]),axis=4)
                high_pass = tf.concat((high_pass,xx[:,:,:,:,1:]),axis=4)
    
        return low_pass, high_pass


    def wave_rec(self,x_low,x_high):
        
        shape = [i.value for i in x_low.get_shape()]
        num_low_pass = shape[4]
        
        # concat
        x_concat = []
        for i in range(num_low_pass):
            x = tf.concat((x_low[:,:,:,:,i:(i+1)],x_high[:,:,:,:,i*7:((i*7)+7)]),axis=4)
            x = _upsampling(x,self.synthesis,num_ch=1,stride=2)
            x_concat.append(x)
        x_concat = tf.concat(x_concat,axis=4)
            
        return x_concat

# test of the generalized pooling
        
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    data = np.load('synthetic.npy')
    data = data[:44,:44,:44,:4]
    
    data = np.expand_dims(data,axis=0)
    data[:,:,:,:,3] =data[:,:,:,:,3] + np.random.normal(0,0.0001,data[:,:,:,:,3].shape)

    
    tf.reset_default_graph()
    
    x = tf.placeholder(tf.float32,shape=(1,44,44,44,4))
    
    genpool = GENPOOL3D()
    
    x_low,x_high = genpool.wave_dec(x)
    
    y = genpool.wave_rec(x_low, x_high)
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer()
        
        test = sess.run(y,feed_dict={x: data})
        
    n  = 2    
    plt.figure()
    plt.imshow(data[0,10,:,:,n])
    plt.colorbar()
    plt.figure()
    plt.imshow(test[0,10,:,:,n])
    plt.colorbar()
    plt.figure()
    plt.imshow(data[0,10,:,:,n] - test[0,10,:,:,n])
    plt.colorbar()
    

