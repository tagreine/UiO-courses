# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:14:31 2018

@author: tlgreiner
"""
import numpy as np

# R2 score
def R2_metric(y ,y_):
    
    # extract the shape (dimensions) of the model 
    shape = y.shape
    n     = shape[0] 
    
    # compute the mean and store it as a vector
    y_m  = np.mean(y_)
    y_mu = y_m*np.ones([n,1])
    
    A = np.dot(np.transpose((y - y_)),(y-y_))
    B = np.dot(np.transpose((y - y_mu)),(y-y_mu))
    
    # compute the R2 score
    R2 = 1 - A/B
    
    return R2


def MSE_metric(y, y_):
    
    # extract the shape (dimensions) of the model 
    shape = y.shape
    n     = shape[0] 
    
    # compute the MSE score
    Err   = np.dot(np.transpose((y - y_)),(y - y_))/n
    Err   = np.squeeze(Err)
    
    return Err

def ord_least_square(x,y):
    
    # exctract the shape of the model
    shape   = x.shape
    n       = shape[0]
        
    # include bias within the data
    x0      = np.ones([n,1])
        
    X = np.concatenate((x0,x),axis=1)

    Xt   = np.transpose(X)
    
    Hat  = np.dot(np.linalg.inv(np.dot(Xt,X)),Xt)
    beta = np.dot(Hat,y)
    y_   = np.dot(X,beta)
    
    return y_


def ridge_least_square(x, y, lamb = 0.01):
    
    # exctract the shape of the model
    shape   = x.shape
    n       = shape[0]
    p       = shape[1] 
        
    # include bias within the data
    x0      = np.ones([n,1])
        
    X = np.concatenate((x0,x),axis=1)
    
    I = np.eye(p+1)
    
    I[0,0] = 0

    Xt   = np.transpose(X)
    
    Hat  = np.dot(np.linalg.inv(np.dot(Xt,X) + lamb*I),Xt)
    beta = np.dot(Hat,y)
    y_   = np.dot(X,beta)
    
    return y_

def least_square_w_basis_exp(x,y,basis=0):
    
    # exctract the shape of the model
    shape   = x.shape
    n       = shape[0]
        
    # include bias within the data
    x0      = np.ones([n,1])
        
    # basis expansion on x, with basis = 0 for straight line fit
        
    X       = np.zeros([n,basis+2])
    X [:,0] = x0[:,0]
    
    
    for i in range(basis+1):
        X[:,(i+1)] = x[:,0]**(i+1)
        
    Xt   = np.transpose(X)
    
    Hat  = np.dot(np.linalg.inv(np.dot(Xt,X)),Xt)
    beta = np.dot(Hat,y)
    y_   = np.dot(X,beta)
    
    return y_

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def basis_exp_2d(x, y, basis = 2, alt = 1):
    
    shapex = x.shape
    #shapey = y.shape
    
    xx = np.zeros([shapex[0], basis])
    yy = np.zeros([shapex[0], basis])
    
    for i in range(basis):
        xx[:,i] = x[:,0]**[i+1]
        yy[:,i] = y[:,0]**[i+1]
        
    if alt == 1:
        
        XX = np.zeros([shapex[0], basis**2])
        
        k = 0
        for i in range(basis):
            for j in range(basis):
                
                XX[:,k] = xx[:,i]*yy[:,j]
                k = k + 1
        
        X = np.c_[xx,yy,XX]
        
        return X
    if alt == 2:
        X = np.c_[xx,yy]
        
        return X
    else:
        print('Fourth parameter should be 1 or 2')
