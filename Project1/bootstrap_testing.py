# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:37:34 2018

@author: tagreine
"""
import numpy as np
from random import random, seed
from tools import ord_least_square, FrankeFunction, ridge_least_square, basis_exp_2d, display_fig, bootstrap_resampling, LegendObject, plot_mean_and_CI, Metrics_param, MSE_metric, R2_metric
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter as cc

def bootstrap_resampling(data_vector, Design_matrix):
    
    shaped  = data_vector.shape
    shapeD  = Design_matrix.shape
    n       = shaped[0]*shaped[1]
    indexes = np.arange(n).reshape(n,1) 
    
    # Extracting index positions for bootstrap resampling
    BootInd = np.random.choice(n,n,replace=True).reshape(n,1)
    BootInd.sort(axis=0)
    
    # Extracting index positions for test (not in bootstrap sample)
    Tmp     = [i for i in indexes if i not in BootInd]
    L       = len(Tmp)
    TestInd = np.zeros([L,1]) 
    for i in range(L):
        TestInd[i] = np.asscalar(Tmp[i])
    
    # Boostrap resampling data set
    data_boot          = np.zeros(shaped)
    Design_matrix_boot = np.zeros(shapeD)
    
    for i in range(n):    
        data_boot[i] = data_vector[int(BootInd[i])]
        for j in range(shapeD[1]):
            Design_matrix_boot[i,j] = Design_matrix[int(BootInd[i]),j]
    
    # Test data set
    shapet             = TestInd.shape
    data_test          = np.zeros(shapet)
    Design_matrix_test = np.zeros([shapet[0],shapeD[1]])
    
    for i in range(shapet[0]):    
        data_test[i] = data_vector[int(TestInd[i])]
        for j in range(shapeD[1]):
            Design_matrix_test[i,j] = Design_matrix[int(TestInd[i]),j]
    
    # Compute statistics on the bootstrap sample
    BootMean = np.mean(data_boot)
    BootVar  = np.var(data_boot)
    BootStd  = np.std(data_boot)
    
    return data_boot, Design_matrix_boot, data_test, Design_matrix_test, BootMean, BootVar, BootStd


shaped    = Data_vector.shape
shapeX    = Design_matrix.shape
num_boots = 10


data_train_n          = np.zeros([shaped[0], shaped[1], num_boots])
Design_matrix_train_n = np.zeros([shapeX[0], shapeX[1], num_boots])


pred_train        = np.zeros([shaped[0], shaped[1], num_boots])
Training_boot_mse = np.zeros([num_boots,1])
Test_boot_mse     = np.zeros([num_boots,1])
Bias_test_n       = np.zeros([num_boots,1])


for i in range(num_boots):
    # Randomly resample the data set with replacement using bootstrap. Test data is constructed by using the samples not within the bootstrap sample
    data_train_n[:,:,i], Design_matrix_train_n[:,:,i], data_test, Design_matrix_test = bootstrap_resampling(Data_vector,Design_matrix)
    
    shapeXt                 = Design_matrix_test.shape
    pred_train[:,:,i], beta = ord_least_square(Design_matrix_train_n[:,:,i],data_train_n[:,:,i])
    pred_test               = np.dot(np.c_[np.ones([shapeXt[0],1]),Design_matrix_test],beta)
        
    Training_boot_mse[i] = MSE_metric(data_train_n[:,:,i],pred_train[:,:,i])
    Test_boot_mse[i]     = MSE_metric(data_test,pred_test)
      
# Compute the variance
VarBoot = np.var(pred_train,axis=2)
VarBoot = np.mean(VarBoot)

# Compute the bias
mean_pred = np.mean(np.mean(pred_train,axis=2))
mean_data = np.mean(Data_vector)

Bias2 = (Data_vector - mean_pred)**2
    
# Compute the average training and test error

Training_MSE = np.mean(Training_boot_mse)

Test_MSE = np.mean(Test_boot_mse)