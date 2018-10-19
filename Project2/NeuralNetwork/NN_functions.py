# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:47:00 2018

@author: tagreine
"""

#=============================================================================
# Stand-alone functions

# Algorithm for defining and initializing the weights. 
#The weigth matrix should be of size NxL where 
# L = size of layer l and N = size of layer l+1  

import numpy as np

def sigmoid(z):
    # The sigmoid activation function
    return 1/(1 + np.exp(-z))
    
    
def sigmoidGradient(z):
    # The derivative of the sigmoid function
    sigmoid = 1/(1 + np.exp(-z))
    return np.multiply(sigmoid,(1 - sigmoid))


def logistic_reg_cost(x, y, theta, lmbda = 0.01, intercept = 'False'):
    # Cost function for logistic regression
    # The algorithm is written using L2 norm regularization. Set 
    # lmbda = 0.0 for no regularization
    
    m = len(y)
    
    if intercept == 'True':
        X = np.c_[np.ones([x.shape[0],1]),x]
    elif intercept == 'False':
        X = x
    
    # Compute the linear function 
    z = X.dot(theta)
    # Compute the non-linear function
    g = sigmoid(z)
    
    
    # Regularization term for cost function
    RegCost = (lmbda/m)*np.transpose( theta ).dot( theta )
    # Logistic regression cost function
    Cost    = ( 1/m )*( - np.transpose( y ).dot( np.log( g ) ) - np.transpose( ( 1 - y ) ).dot( np.log( 1 - g ) ) ) + RegCost
    
    return Cost

def regression_cost(x, y, theta, lmbda = 0.01, intercept = 'False'):
    # Cost function for logistic regression
    # The algorithm is written using L2 norm regularization. Set 
    # lmbda = 0.0 for no regularization
    
    m = len(y)
    
    if intercept == 'True':
        X = np.c_[np.ones([x.shape[0],1]),x]
    elif intercept == 'False':
        X = x
    
    h = X.dot(theta)

    # Regularization term for cost function
    RegCost = (lmbda/m)*np.transpose( theta ).dot( theta )
    # Logistic regression cost function
    Cost    = ( 1/(2*m) )*np.transpose( h - y ).dot((h - y)) + RegCost
    
    return Cost


def gradientDescent(x, y, theta, method = 'Logistic', alpha = 0.001, lmbda = 0.001, num_iters = 100, intercept = 'False'):   
    # Gradient descent optimization algorithm 
    # The algorithm is written using L2 norm regularization. 
    # Set lmbda = 0.0 for no regularization
    
    m = len(y)

    Cost = np.zeros(num_iters)    
    
    if method == 'Logistic':
        
        if intercept == 'True':
            X = np.c_[np.ones([x.shape[0],1]),x]
        elif intercept == 'False':
            X = x
       
        theta_new = theta
        
        for i in range(num_iters):
            
            # Compute the linear function 
            z = X.dot(theta_new)
            # Compute the non-linear function
            g = sigmoid(z)
            # Regularization term for gradient descent
            RegGrad = ( lmbda/( 2*m ) )*theta
            # Gradient of the cost function
            Grad    = ( 1/m )*np.transpose(X).dot( g - y ) + RegGrad                
            # Do the gradient descent optimization
            theta_new = theta_new - alpha*Grad 
            
            # Save the cost for every iteration
            Cost[i] = logistic_reg_cost(X, y, theta_new, lmbda, intercept = 'False')
            
    if method == 'Regression':
        
        if intercept == 'True':
            X = np.c_[np.ones([x.shape[0],1]),x]
        elif intercept == 'False':
            X = x
       
        theta_new = theta
        
        for i in range(num_iters):
            
            # Regularization term for gradient descent
            RegGrad = ( lmbda/( 2*m ) )*theta
            # Gradient of the cost function
            Grad    = ( 1/m )*np.transpose(X).dot( X.dot(theta_new) - y ) + RegGrad                
            # Do the gradient descent optimization
            theta_new = theta_new - alpha*Grad 
            
            # Save the cost for every iteration
            Cost[i] = regression_cost(X, y, theta_new, lmbda, intercept = 'False')
      
    return Cost, theta