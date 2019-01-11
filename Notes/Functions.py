# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:21:01 2019

@author: tlgreiner
"""

import numpy as np

def Mexican_Hat(t,sigma):
    # t     = time vector
    # sigma = standard deviation 
    
    a  = np.divide(2,np.sqrt(np.multiply(3,sigma)))
    b  = (1 - (t/sigma)**2)
    c  = -np.divide(t**2,2*sigma**2)
    
    MH = a*b*np.exp(c)
    
    return MH

def sigmoid(z):
    # Sigmoid activation function
    return np.divide(1,1 + np.exp(-z))

def Relu(z):
    # Rectified linear unit
    relu = np.maximum(0,z)
    return relu

def tanh(z):
    # The tangens hyperbolicus activation function
    tan = np.tanh(z)
    return tan 


def leakyRelu(z,eta):
    # Rectified linear unit
    lrelu = np.maximum(eta*z,z)
    return lrelu