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


#=====================Activation functions==============================================================================

def sigmoid(z):
    # The sigmoid activation function
    return 1/(1 + np.exp(-z))
    
    
def sigmoidGradient(z):
    # The derivative of the sigmoid function
    sigmoid = 1/(1 + np.exp(-z))
    return np.multiply(sigmoid,(1 - sigmoid))



#=================Prediction=================================================================================

def predict(X,threshold):
    prob = sigmoid(X)
    
    probabilities = np.zeros([prob.shape[0],prob.shape[1]])
    probabilities[prob>=threshold] = 1  
    
    return probabilities

def predict_nn(Y,threshold):    
    Y[Y>=threshold] = 1
    Y[Y<threshold] = 0
    probabilities   = Y
    return probabilities

#=====================Cost functions====================================================================================


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
    RegCost = (lmbda/2*m)*np.transpose( theta ).dot( theta )
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

def log_likelihood(x, y, theta):
    m    = x.shape[0] 
    pred = np.dot(x, theta)
    cost = (1/m)*np.sum( y*pred - np.log(1 + np.exp(pred)) )
    return cost



#=====================Optimization algorithms====================================================================================

def gradientDescent(x, y, theta, alpha = 0.001, lmbda = 0.001, num_iter=100, intercept = 'False',print_cost= 'True'):   
    # Gradient descent optimization algorithm 
    # The algorithm is written using L2 norm regularization. 
    # Set lmbda = 0.0 for no regularization
    
  
    if intercept == 'True':
        X = np.c_[np.ones([x.shape[0],1]),x]
    elif intercept == 'False':
        X = x
    
    m = X.shape[0]
    
    for step in range(num_iter):
        z = np.dot(X, theta)
        predictions = sigmoid(z)

        # Update weights with gradient
        output_error = y - predictions
        gradient = np.dot(X.T, output_error)
        theta += ( 1/m )*alpha * (gradient + lmbda*theta)
        
        Cost = log_likelihood(X, y, theta)
        
        if print_cost == 'True':
            if step % 100 == 0:
                print( Cost )
            
    return Cost, theta


def logistic_regression_batch( X, Y, theta_train, epochs = 100, batch_size = 10,alpha = 0.0001, lmbda = 0.01, intercept = 'False'):    
    
    m = X.shape[0]
      
    for k in range(epochs):
        for j in range(np.int(m/batch_size)):
            cost_SGD, theta = gradientDescent(X[(j*batch_size):((j + 1)*batch_size),:].reshape(batch_size,X.shape[1]), Y[(j*batch_size):((j + 1)*batch_size),:].reshape(batch_size,1), theta_train, alpha, lmbda, num_iter=1, intercept = 'False')
            theta_train = theta.copy()
            
            if j % 1000 == 0:
                print( cost_SGD)
    return cost_SGD, theta 
