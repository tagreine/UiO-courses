# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:25:08 2018

@author: tagreine
"""

'''
Defining a python class for neural networks 
'''
import numpy as np


class Neural_Network(object):
    
    def __init__(self):
        # Setting up Hyperparameters: constant values which defines the 
        # architecture of the neural network
        self.inputLayerSize   = 2
        self.outputLayerSize  = 1
        self.hiddenLayerSize1 = 3
        self.hiddenLayerSize2 = 3

        
        # Defining and initializing the weights        
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize1)
        self.W2 = np.random.randn(self.hiddenLayerSize1,self.hiddenLayerSize2)
        self.W3 = np.random.randn(self.hiddenLayerSize2,self.outputLayerSize)
        
        # Defining and initializing the biases
        self.bias1 = np.random.randn(self.hiddenLayerSize1,1)
        self.bias2 = np.random.randn(self.hiddenLayerSize2,1)
        self.bias3 = np.random.randn(self.outputLayerSize,1)
        
        
    def forwardProp(self,X):
        # Forward propagation algorithm for neural network
        # The input X should have the features aligned as collumns. In this case
        # we have 
        # Z = XW 
        # where W.T is the transpose of the weigths going
        # from layer l to layer l+1 
        # a are the activations in layer l
            
        self.a1 = X
        
        # Hidden layers
        self.z2 = self.a1.dot(self.W1) + self.bias1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = self.a2.dot(self.W2) + self.bias2
        self.a3 = self.sigmoid(self.z3)
        
        # Output layer
        self.z4 = self.a3.dot(self.W3) + self.bias3
        y_pred  = self.sigmoid(self.z4)
           
        return y_pred
             
    #=========================cost functions===================================
    
    def costGrad(self,X,y):
        # Backward propagation algorithm for neural network
        
        # Do forward propagation
        self.y_pred = self.forwardProp(X)
        
        # Do back propagation errors in all layer (exept input layer)
        
        # Define the error terms and derivatives wrt the weights 
        # Output error
        delta4 = np.multiply(-(y - self.y_pred), self.sigmoidGradient( self.z4 ) )
        dCdW3   = self.a3.T.dot( delta4 )
        
        # Hidden layer 2 error
        delta3 = np.multiply( delta4 , self.sigmoidGradient( self.z3 ) )
        dCdW2   = self.a2.T.dot( delta3 )
        
        # Hidden layer 1 error
        delta2 = np.multiply( delta3 , self.sigmoidGradient( self.z2 ) )
        dCdW1  = self.a1.T.dot( delta2 )
        
        # Define the error terms and derivatives wrt the biases
        
        # Output error
        dCdb4 = delta4
        
        # Hidden layer 2 error
        dCdb3 = delta3        
        
        # Hidden layer 1 error
        dCdb2 = delta2         
    
        
        # Returning the cost function derivatives wrt to weights 
        return dCdW3, dCdW2, dCdW1, dCdb4, dCdb3, dCdb2     
        
    def gradientDescentOptimizer(self,):
        a = 1
        return a
        
        
        
    
    #=========================activation functions=============================    
    
    # Sigmoid    
    def sigmoid(self,z):
        # The sigmoid activation function
        return 1/(1 + np.exp(-z))
    
    def sigmoidGradient(self,z):
        # The derivative of the sigmoid function
        sigm = 1/(1 + np.exp(-z))
        return np.multiply(sigm,(1 - sigm))

    # Rectified linear unit
    def Relu(self,z):
        relu = np.maximum(0,z)
        return relu
    
    def ReluGradient(self,z):
        # The derivative of the Relu function
        relu = self.Relu(z)
        return np.int64(relu>0)
           
    # Tangens hyperbolicus
    def tanh(self,z):
        # The tangens hyperbolicus activation function
        tan = np.tanh(z)
        return tan 
    
    def tanhGradient(self,z):
        # The derivative of tangens hyperbolicus activation function
        tan = self.tanh(z)
        return (1 - np.square(tan))



    
