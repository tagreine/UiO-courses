# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:25:08 2018

@author: tagreine
"""

'''
Defining a python class for neural networks 
'''
import numpy as np

        
# Defining and initializing the weights
def initialize_params(inputLayerSize,hiddenLayerSize1,hiddenLayerSize2,outputLayerSize):

    #Params = {}
    
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
    a1 = X    
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

    # Output layer
    delta4 = np.multiply( -(y - y_pred) , sigmoidGradient(z4))  
        
    # Hidden layer 2 error
    delta3 = np.multiply( np.dot( W3.T , delta4 ) , sigmoidGradient(z3) )  
        
    # Hidden layer 1 error
    delta2 = np.multiply( np.dot( W2.T , delta3 ) , sigmoidGradient(z2) ) 
        
    # Partial derivatives of cost function wrt weights and biases
    dCdW3  = np.dot( delta4 , a3.T)
    dCdW2  = np.dot( delta3 , a2.T)
    dCdW1  = np.dot( delta2 , a1.T)
        
    dCdb3 = delta4
    dCdb2 = delta3        
    dCdb1 = delta2         
    
        
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
    Gradb1 = (1/m)*(dCdb1 + lmbda*bias1)
    Gradb2 = (1/m)*(dCdb2 + lmbda*bias2)
    Gradb3 = (1/m)*(dCdb3 + lmbda*bias3)
        
        
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


