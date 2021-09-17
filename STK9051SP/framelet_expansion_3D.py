# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:14:55 2021

@author: tlgreiner
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time

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

def wave_dec(x,analysis):
    
    shape = [i.value for i in x.get_shape()]
    num_features = shape[4]
          

    for i in range(num_features):
        
        xx = tf.nn.conv3d(x[:,:,:,:,i:(i+1)],analysis,strides=[1,2,2,2,1],padding='SAME')
        
        if i==0:    
            low_pass  = xx[:,:,:,:,0:1]
            high_pass = xx[:,:,:,:,1:]
        
        else:
            low_pass  = tf.concat((low_pass,xx[:,:,:,:,0:1]),axis=4)
            high_pass = tf.concat((high_pass,xx[:,:,:,:,1:]),axis=4)

    return low_pass, high_pass


def wave_rec(x_low,x_high,synthesis):
    
    shape = [i.value for i in x_low.get_shape()]
    num_low_pass = shape[4]
    
    # concat
    x_concat = []
    for i in range(num_low_pass):
        x = tf.concat((x_low[:,:,:,:,i:(i+1)],x_high[:,:,:,:,i*7:((i*7)+7)]),axis=4)
        x = _upsampling(x,synthesis,num_ch=1,stride=2)
        x_concat.append(x)
    x_concat = tf.concat(x_concat,axis=4)
        
    return x_concat




class CNN3DFRAME:
    def __init__(self):
        
        self.analysis,self.synthesis = Haar_framelets()        
    
    def set_model(self):
        
        self.kernels = [[5,5,5,1,8],  # regular conv -> wave_dec 
                        [3,3,3,8,16], # regular conv
                        [3,3,3,16,8], # regular conv -> wave_rec
                        [3,3,3,16,8], # concatenate conv 
                        [3,3,3,8,1],
                        ]
        self.bias    = [[8],
                        [16],
                        [8],
                        [8],
                        [1]
                        ]
        
        
    def model_genpool(self,x):
        
        init = tf.contrib.layers.variance_scaling_initializer(factor=2,mode='FAN_IN',uniform=True,seed=42)
        reg  = None
                
        self.set_model()
        kernels = self.kernels
        bias    = self.bias
        analysis  = tf.constant(self.analysis,dtype=tf.float32)
        synthesis = tf.constant(self.synthesis,dtype=tf.float32)
        
        w = {}
        b = {}
        
        for i in range(len(kernels)):
            w['w{}'.format(i+1)] = tf.get_variable(name='w{}'.format(i+1),
                                                    shape=kernels[i],
                                                    initializer=init,
                                                    regularizer=reg)
            b['b{}'.format(i+1)] = tf.Variable(tf.zeros(bias[i],
                                               name='b{}'.format(i+1))
                                               ) 
            #params = [w,b]
        
        
        # level 1 conv
        x_lvl1 = tf.nn.conv3d(x,         w['w1'],strides=[1,1,1,1,1],padding='SAME') # (batch,d,h,w,8)
        x_lvl1 = tf.keras.activations.relu(x_lvl1,threshold=0.0)                         
        
        # wavelet decomposition into level 2
        x_low_lvl1, x_high_lvl1 = wave_dec(x_lvl1,analysis)                                # x_low: (batch,d/2,h/2,w/2,8), x_high: (batch,d/2,h/2,w/2,56)  
        x_lvl2 = tf.nn.conv3d(x_low_lvl1,w['w2'],strides=[1,1,1,1,1],padding='SAME') # (batch,d/2,h/2,w/2,16)
        x_lvl2 = tf.keras.activations.relu(x_lvl2,threshold=0.0)
        x_lvl2 = tf.nn.conv3d(x_lvl2,    w['w3'],strides=[1,1,1,1,1],padding='SAME') # (batch,d/2,h/2,w/2,8)
        x_lvl2 = tf.keras.activations.relu(x_lvl2,threshold=0.0)
        # wavelet reconstruction into level 1
        x_rec  = wave_rec(x_lvl2,x_high_lvl1,synthesis)                                # (batch,d,h,w,8)       
        # concatenate the skipped features and the reconstructed
        x      = tf.concat((x_rec,x_lvl1),axis=4)                                    # (batch,d,h,w,16)
        x      = tf.nn.conv3d(x,         w['w4'],strides=[1,1,1,1,1],padding='SAME') 
        x      = tf.keras.activations.relu(x,threshold=0.0)
        x      = tf.nn.conv3d(x,         w['w5'],strides=[1,1,1,1,1],padding='SAME')
        
        return x

    def model_pool(self,x):
        
        init = tf.contrib.layers.variance_scaling_initializer(factor=2,mode='FAN_IN',uniform=True,seed=42)
        reg  = None
                
        self.set_model()
        kernels = self.kernels
        bias    = self.bias
        
        w = {}
        b = {}
        
        for i in range(len(kernels)):
            w['w{}'.format(i+1)] = tf.get_variable(name='w{}'.format(i+1),
                                                    shape=kernels[i],
                                                    initializer=init,
                                                    regularizer=reg)
            b['b{}'.format(i+1)] = tf.Variable(tf.zeros(bias[i],
                                               name='b{}'.format(i+1))
                                               ) 
            #params = [w,b]
        
        
        # level 1 conv
        x_lvl1 = tf.nn.conv3d(x,         w['w1'],strides=[1,1,1,1,1],padding='SAME') # (batch,d,h,w,8)
        x_lvl1 = tf.keras.activations.relu(x_lvl1,threshold=0.0)                         
        
        # low rank approximation into level 2
        x_low_lvl1 = tf.nn.avg_pool3d(x_lvl1,ksize=(2,2,2),strides=(2,2,2),padding='SAME') # x_low: (batch,d/2,h/2,w/2,8) 
        x_lvl2 = tf.nn.conv3d(x_low_lvl1,w['w2'],strides=[1,1,1,1,1],padding='SAME') # (batch,d/2,h/2,w/2,16)
        x_lvl2 = tf.keras.activations.relu(x_lvl2,threshold=0.0)
        x_lvl2 = tf.nn.conv3d(x_lvl2,    w['w3'],strides=[1,1,1,1,1],padding='SAME') # (batch,d/2,h/2,w/2,8)
        x_lvl2 = tf.keras.activations.relu(x_lvl2,threshold=0.0)
        # wavelet reconstruction into level 1
        x_rec  = tf.keras.layers.UpSampling3D(size=(2,2,2))(x_lvl2)                          # (batch,d,h,w,8)       
        # concatenate the skipped features and the reconstructed
        x      = tf.concat((x_rec,x_lvl1),axis=4)                                    # (batch,d,h,w,16)
        x      = tf.nn.conv3d(x,         w['w4'],strides=[1,1,1,1,1],padding='SAME') 
        x      = tf.keras.activations.relu(x,threshold=0.0)
        x      = tf.nn.conv3d(x,         w['w5'],strides=[1,1,1,1,1],padding='SAME')
        
        return x
    

def train_model(x,epochs,learning_rate,weight_decay=1e-5,gpu=True,method='genpool'):    
    
    tf.reset_default_graph()
    
    x     = np.expand_dims(x,axis=0) 
    shape = x.shape
    cnn   = CNN3DFRAME()
    
    tf_inp = tf.placeholder(tf.float32,shape=[shape[0],shape[1],shape[2],shape[3],shape[4]])
    tf_dat = tf.placeholder(tf.float32,shape=[shape[0],shape[1],shape[2],shape[3],shape[4]])
    
    if method=='genpool':
        pred = cnn.model_genpool(tf_inp)
    elif method=='pool':
        pred = cnn.model_pool(tf_inp)
    else:
        raise ValueError('Choose either genpool or pool for method')
    
    loss = tf.reduce_mean(tf.square( tf_dat - pred ))
    optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate=learning_rate,
                                              weight_decay=weight_decay
                                              ).minimize(loss)
    
    if gpu==True:
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True) 
        config.gpu_options.allow_growth = True  
    else:
        config=None
        print('No gpu available')
    
    epoch_loss = []
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        start = time.time()
        for epoch in tqdm(range(epochs)):
            
            _, loss_ = sess.run((optimizer,loss),
                                feed_dict={tf_inp: x,
                                           tf_dat: x
                                           })
            epoch_loss.append(loss_)
            print('Epoch nr:',epoch+1 ,'L2 loss:',loss_)
        
        elapsed_time = (time.time() - start)    
        prediction = sess.run((pred),feed_dict={tf_inp:x})
    
    print('Method {} took {} sec to complete'.format(method,elapsed_time))    
    
    return np.squeeze(prediction),epoch_loss



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    
    data = np.load('synthetic_gain.npy')
    data = data[:180,:120,:180,:1]
    data = data/np.max(np.abs(data)) 
    data = data + np.random.normal(0,0.01,data.shape)
    
    
    epochs        = 2000
    learning_rate = 0.0001
    weight_decay  = 1e-3
    gpu           = True
    method        = 'genpool' # 'pool' 'genpool'
    denoise,loss  = train_model(data,epochs=epochs,
                                learning_rate=learning_rate,
                                weight_decay=weight_decay,
                                gpu=gpu,
                                method=method
                                ) 
    
    '''
    plt.figure()
    plt.plot(np.arange(epochs)+1,lossgenpool)
    plt.plot(np.arange(epochs)+1,losspool)
    plt.title('Training loss')
    plt.ylabel('L2 loss')
    plt.xlabel('Epoch')
    plt.legend(('Generalized pooling','Max pooling'))
    plt.grid()
    '''
    
    plt.figure()
    plt.imshow(data[10,:,:,0],cmap='gray')
    plt.clim([-0.1,0.1])
    plt.colorbar()
    plt.figure()
    plt.imshow(denoise[10,:,:],cmap='gray')
    plt.clim([-0.1,0.1])
    plt.colorbar()
    plt.figure()
    plt.imshow(data[10,:,:,0]-denoise[10,:,:],cmap='gray')
    plt.clim([-0.1,0.1])
    plt.colorbar()


    plt.figure()
    plt.imshow(data[10,:,::-1,0],cmap='gray')
    plt.clim([-0.1,0.1])    






