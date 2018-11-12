# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:25:08 2018

@author: tagreine
"""

'''
Defining a python class for neural networks 
'''
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils import resample
from tools import MSE_metric
        
# Defining and initializing the weights
def initialize_params(inputLayerSize,hiddenLayerSize1,hiddenLayerSize2,outputLayerSize):
    
    W1 = np.random.randn(hiddenLayerSize1,inputLayerSize)*0.01
    W2 = np.random.randn(hiddenLayerSize2,hiddenLayerSize1)*0.01
    W3 = np.random.randn(outputLayerSize,hiddenLayerSize2)*0.01

    # Defining and initializing the biases
    bias1 = np.zeros([hiddenLayerSize1,1])
    bias2 = np.zeros([hiddenLayerSize2,1])
    bias3 = np.zeros([outputLayerSize,1])
    
    return W1,W2,W3,bias1,bias2,bias3  

        
def forwardProp(X,W1,W2,W3,bias1,bias2,bias3,activation = 'sigmoid'):
    # Forward propagation algorithm for neural network
    # The input X should have the features aligned as rows. In this case
    # we have 
    # Z = XW 
    # where W.T is the transpose of the weigths going
    # from layer l to layer l+1 
    
    # Input layer       
    a1 = X.copy()    
    # Hidden layers
    z2 = W1.dot(a1) + bias1    
    if activation == 'sigmoid':    
        a2 = sigmoid(z2)
    if activation == 'tanh':    
        a2 = tanh(z2)
    if activation == 'Relu':    
        a2 = Relu(z2)
        
    z3 = W2.dot(a2) + bias2
    if activation == 'sigmoid':    
        a3 = sigmoid(z3)
    if activation == 'tanh':    
        a3 = tanh(z3)
    if activation == 'Relu':    
        a3 = Relu(z3)
          
    # Output layer
    z4 = W3.dot(a3) + bias3
    if activation == 'sigmoid':    
        y_pred  = sigmoid(z4)
    if activation == 'tanh':    
        y_pred  = tanh(z4)
    if activation == 'Relu':    
        y_pred  = Relu(z4)
    
           
    return y_pred,z4,z3,z2,a3,a2,a1
             
    #=========================cost functions===================================
    
def costGrad(X,y,W1,W2,W3,bias1,bias2,bias3,activation = 'sigmoid'):
    # Backward propagation algorithm for neural network
        
    # Do forward propagation
    y_pred,z4,z3,z2,a3,a2,a1 = forwardProp(X,W1,W2,W3,bias1,bias2,bias3,activation)
    
 
    # Do back propagation errors in all layer (exept input layer)        
    # Define the error terms and derivatives wrt the weights 
    # Output error
    
    if activation == 'sigmoid':
        # Output layer
        delta4 = -(y - y_pred)
        
        # Hidden layer 2 error
        delta3 = np.multiply( np.dot( W3.T , delta4 ) , sigmoidGradient(z3) )  
        
        # Hidden layer 1 error
        delta2 = np.multiply( np.dot( W2.T , delta3 ) , sigmoidGradient(z2) ) 
        
    if activation == 'tanh':
        # Output layer
        delta4 = -(y - y_pred) 
        
        # Hidden layer 2 error
        delta3 = np.multiply( np.dot( W3.T , delta4 ) , tanhGradient(z3) )  
        
        # Hidden layer 1 error
        delta2 = np.multiply( np.dot( W2.T , delta3 ) , tanhGradient(z2) )
        
    if activation == 'Relu':
        # Output layer
        delta4 = -(y - y_pred)
        
        # Hidden layer 2 error
        delta3 = np.multiply( np.dot( W3.T , delta4 ) , ReluGradient(z3) )  
        
        # Hidden layer 1 error
        delta2 = np.multiply( np.dot( W2.T , delta3 ) , ReluGradient(z2) ) 

    # Partial derivatives of cost function wrt weights and biases
    dCdW3  = np.dot( delta4 , a3.T)
    dCdW2  = np.dot( delta3 , a2.T)
    dCdW1  = np.dot( delta2 , a1.T)
        
    dCdb3 = np.sum(delta4,axis=1,keepdims=True)
    dCdb2 = np.sum(delta3,axis=1,keepdims=True)
    dCdb1 = np.sum(delta2,axis=1,keepdims=True)
    
        
    # Returning the cost function derivatives wrt to weights 
    return dCdW3, dCdW2, dCdW1, dCdb3, dCdb2, dCdb1     
        
def gradientDescentOptimizer(X,params, grads,eta=0.001,lmbda=0.001):
    
    m = X.shape[0] 
    
    # Parameters to update        
    W1 = params[0]
    W2 = params[1]    
    W3 = params[2]     
    bias1 = params[3]
    bias2 = params[4]    
    bias3 = params[5]    
    
    dCdW1 = grads[0]
    dCdW2 = grads[1]    
    dCdW3 = grads[2]    
    
    dCdb1 = grads[3]
    dCdb2 = grads[4]    
    dCdb3 = grads[5] 
    
    # Derivatives of cost function including L2 regularization
    # wrt weights
    GradW1 = (1/m)*(dCdW1 + lmbda*W1)
    GradW2 = (1/m)*(dCdW2 + lmbda*W2)
    GradW3 = (1/m)*(dCdW3 + lmbda*W3)
    # wrt biases    
    Gradb1 = (1/m)*(dCdb1)
    Gradb2 = (1/m)*(dCdb2)
    Gradb3 = (1/m)*(dCdb3)
        
        
    # Update weights and biases
    W1 = W1 - eta*GradW1
    W2 = W2 - eta*GradW2
    W3 = W3 - eta*GradW3

    bias1 = bias1 - eta*Gradb1
    bias2 = bias2 - eta*Gradb2
    bias3 = bias3 - eta*Gradb3        
                
    return W1,W2,W3,bias1,bias2,bias3
        
    
    #=========================activation functions=============================    
    
# Sigmoid    
def sigmoid(z):
    # The sigmoid activation function
    return 1/(1 + np.exp(-z))
    
def sigmoidGradient(z):
    # The derivative of the sigmoid function
    sigm = 1/(1 + np.exp(-z))
    return np.multiply(sigm,(1 - sigm))

# Rectified linear unit
def Relu(z):
    relu = np.maximum(0,z)
    return relu
    
def ReluGradient(z):
    # The derivative of the Relu function
    relu = Relu(z)
    return np.int64(relu>0)
           
# Tangens hyperbolicus
def tanh(z):
    # The tangens hyperbolicus activation function
    tan = np.tanh(z)
    return tan 
    
def tanhGradient(z):
    # The derivative of tangens hyperbolicus activation function
    tan = tanh(z)
    return (1 - np.square(tan))


def cost_nn(y, y_pred):
    m    = len(y) 
    cost = (1/m)*np.sum( y*y_pred - np.log(1 + np.exp(y_pred)) )
    return cost



def neural_net_batch( X, Y, method='Classification', hiddenLayerSize1=20,hiddenLayerSize2=20, epochs = 100, batch_size = 10,alpha = 0.0001, lmbda = 0.01, activation='sigmoid', intercept = 'False'):    
    
    m = X.shape[1]
    
    inputLayerSize  = X.shape[0]
    outputLayerSize = 1
    
    W1,W2,W3,bias1,bias2,bias3 = initialize_params(inputLayerSize,hiddenLayerSize1,hiddenLayerSize2,outputLayerSize)

    
    for k in range(epochs):
        for j in range(np.int(m/batch_size)):
            
            y_pred,z4,z3,z2,a3,a2,a1 = forwardProp(X[:,(j*batch_size):((j+1)*batch_size)],W1,W2,W3,bias1,bias2,bias3,activation)
            
            dCdW3, dCdW2, dCdW1, dCdb3, dCdb2, dCdb1 = costGrad(X[:,(j*batch_size):((j+1)*batch_size)],Y[:,(j*batch_size):((j+1)*batch_size)],W1,W2,W3,bias1,bias2,bias3,activation)

            params = [W1, W2, W3, 
                      bias1, bias2, bias3,
                      ]

            grads = [dCdW1, dCdW2, dCdW3,
                     dCdb1, dCdb2, dCdb3, 
                     ]

            W1,W2,W3,bias1,bias2,bias3 = gradientDescentOptimizer(X[:,(j*batch_size):((j+1)*batch_size)], params, grads,eta=alpha,lmbda=lmbda)
                  
            if method == 'Classification': 
                cost_SGD = cost_nn(Y[:,(j*batch_size):((j+1)*batch_size)], y_pred)
            if method == 'Regression':
                cost_SGD = MSE_metric(Y[:,(j*batch_size):((j+1)*batch_size)].T, y_pred.T)
            
            if j % 1000 == 0:
                print( cost_SGD)
    
    return cost_SGD, W1, W2, W3, bias1, bias2, bias3





#=========================model assessment=============================    


def bootstrap_resampling_neural_net_classifier(y_train, y_test, x_train, x_test, model_complx, batch,hidden_layers = (15,15), activation='relu', n_boostraps = 500):
    
    # Bootstrap algorithm for model assessment (not fully correct bootstrap, since test samples could be drawn from the samples which are within training set)
    
    mc = len(model_complx)
    
    error     = np.zeros([mc])
    error_t   = np.zeros([mc])
    
    #boot_er   = np.zeros([n_boostraps])
    #boot_er_t = np.zeros([n_boostraps])
    
    for j in range(mc):   
        
        
        MLP_class = MLPClassifier(activation=activation,solver='sgd', batch_size=batch, alpha=model_complx[j], hidden_layer_sizes=hidden_layers, shuffle=True)
        
        y_pred   = np.empty((y_train.shape[0], n_boostraps))
        y_pred_t = np.empty((y_test.shape[0], n_boostraps))
        
        for i in range(n_boostraps):
            x_, y_   = resample(x_train, y_train)
            x_t, y_t = resample(x_test, y_test)
            
            # Predict training and test data for each bootstrap
           
            y_pred[:,i]   = np.ravel(MLP_class.fit(x_,y_).predict(x_train))
            y_pred_t[:,i] = np.ravel(MLP_class.fit(x_,y_).predict(x_test))
            
            #boot_er[i]   = MLP_class.score(x_,y_pred)
            #boot_er_t[i] = MLP_class.score(x_t,y_pred_t)
              
            # Compute the error, variance and bias squared at eah point in the model    
        error[j]   = np.mean( np.mean((y_train - y_pred)**2, axis=1, keepdims=True) )
        error_t[j] = np.mean( np.mean((y_test - y_pred_t)**2, axis=1, keepdims=True) )
        
        #error[j]   = np.mean(boot_er)  
        #error_t[j] = np.mean(boot_er_t)
        
        
        print( 'Next regularization parameter' )        
    
    return error, error_t

def bootstrap_resampling_neural_net_regressor(y_train, y_test, x_train, x_test, model_complx, batch,hidden_layers = (15,15), activation='relu', n_boostraps = 500):
    
    # Bootstrap algorithm for model assessment (not fully correct bootstrap, since test samples could be drawn from the samples which are within training set)
    
    mc = len(model_complx)
    
    error     = np.zeros([mc])
    error_t   = np.zeros([mc])
    
    bias     = np.zeros([mc])
    bias_t     = np.zeros([mc])
    
    variance_t = np.zeros([mc])    
    variance = np.zeros([mc])
    #boot_er   = np.zeros([n_boostraps])
    #boot_er_t = np.zeros([n_boostraps])
    
    for j in range(mc):   
        
        
        MLP_class = MLPRegressor(activation=activation,solver='sgd', batch_size=batch, alpha=model_complx[j], hidden_layer_sizes=hidden_layers, shuffle=True)
        
        y_pred   = np.empty((y_train.shape[0], n_boostraps))
        y_pred_t = np.empty((y_test.shape[0], n_boostraps))
        
        for i in range(n_boostraps):
            x_, y_   = resample(x_train, y_train)
            x_t, y_t = resample(x_test, y_test)
            
            # Predict training and test data for each bootstrap
           
            y_pred[:,i]   = np.ravel(MLP_class.fit(x_,y_).predict(x_train))
            y_pred_t[:,i] = np.ravel(MLP_class.fit(x_,y_).predict(x_test))
          
           
        # Compute the error, variance and bias squared at eah point in the model    
        error[j]   = np.mean( np.mean((y_train - y_pred)**2, axis=1, keepdims=True) )
        error_t[j] = np.mean( np.mean((y_test - y_pred_t)**2, axis=1, keepdims=True) )

        bias[j]    = np.mean( (y_train - np.mean(y_pred, axis=1, keepdims=True))**2 )
        bias_t[j]  = np.mean( (y_test - np.mean(y_pred_t, axis=1, keepdims=True))**2 )
        
        variance_t[j] = np.mean( np.var(y_pred_t, axis=1, keepdims=True) )
        variance[j]   = np.mean( np.var(y_pred, axis=1, keepdims=True) )        
        
        print( 'Next regularization parameter' )        
    
    return error, error_t, bias, bias_t, variance, variance_t
