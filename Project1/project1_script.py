# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from random import random, seed
from tools import ord_least_square, FrankeFunction, ridge_least_square, basis_exp_2d, display_fig, LegendObject, plot_mean_and_CI, Metrics_param, MSE_metric, R2_metric, k_fold_CV
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter as cc


# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)
# Plot the surface.

display_fig(z,x,y,save=False)
#display_fig(z,x,y,save=True,name='',title='Franke function')

'''
a)###################################################################################################################################
'''

# 1. OLS with no basis expansions or xy correlations #####

x    = np.arange(0, 1, 0.05)
y    = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)

shapex = x.shape
shapey = y.shape
shapez = z.shape

xf1 = x.flatten().reshape(shapex[0]*shapex[1],1)
yf1 = y.flatten().reshape(shapey[0]*shapey[1],1)
Z   = z.flatten().reshape(shapex[0]*shapex[1],1)

X = basis_exp_2d(xf1,yf1,order=1)

z_pred, beta = ord_least_square(X,Z)
z_pred       = z_pred.reshape(z.shape)

display_fig(z_pred,x,y,save=False)
#display_fig(z_pred,x,y,save=True,name='',title='Predicted function')

# 2. OLS and basis expansions with and without x,y correlations #####


X = basis_exp_2d(xf1,yf1,order=8)

z_pred, beta = ord_least_square(X,Z)
z_pred       = z_pred.reshape(z.shape)

display_fig(z_pred,x,y,save=False)

#display_fig(z_pred,x,y,save=True,name='',title='Predicted function')

# 3. OLS and basis expansions with and without x,y correlations and added noise #####

np.random.seed(1)
Z_n = Z + 0.1*np.max(Z)*np.random.randn(shapez[0]*shapez[1],1)

display_fig(Z_n.reshape(shapez),x,y,save=False)

z_pred_n, beta = ord_least_square(X,Z_n)
z_pred_n       = z_pred_n.reshape(z.shape)

display_fig(z_pred_n,x,y,save=False)
#display_fig(z_pred_n,x,y,save=True,name='',title='Predicted function with noise')

##########################################################################################################################################
# 4. Computing the metrics of the predictions and model assessment
X = basis_exp_2d(xf1,yf1,order=8)
Z_n = Z + 0.1*np.max(Z)*np.random.randn(shapez[0]*shapez[1],1)
Data_vector   = Z_n
Pred_vector   = z_pred_n.reshape(Z_n.shape)
shapeX        = X.shape
Design_matrix = np.c_[np.ones([shapeX[0],1]), X]


# Computing the variance and standard error of the estimated least squares parameters    
VarBeta, StdBeta = Metrics_param(beta, Data_vector, Pred_vector, Design_matrix,compute_var=True)

# Computing the confidence intervals
betalow  = beta - StdBeta
betahigh = beta + StdBeta

# Estimated Beta, Beta + Se and Beta - Se 
fig = plt.figure(1, figsize=(7, 2.5))
plot_mean_and_CI(beta.reshape(shapeX[1] + 1,), betahigh.reshape(shapeX[1] + 1,), betalow.reshape(shapeX[1] + 1,), color_mean='k', color_shading='k')

bg = np.array([1, 1, 1]) 
colors = ['black', 'blue', 'green']
colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]
 
plt.legend([0], [r'$\hat \beta$$\pm$SE[$\hat \beta$]'],
           handler_map={
               0: LegendObject(colors[0], colors_faded[0])
            })
plt.xlabel(r'$\hat \beta_i$')
plt.title('Confidence interval')
plt.tight_layout()
plt.grid()
plt.show()
    
# Computing MSE and R2 wrt model complexity

m_comp = shapeX[1] + 1
MSE    = np.zeros([m_comp,1])
R2     = np.zeros([m_comp,1])

for i in range(m_comp):
    pred, beta = ord_least_square(X[:,0:i],Z_n)

    MSE[i]   = MSE_metric(Z_n,pred)
    R2[i]    = R2_metric(Z_n,pred)
    

plt.subplot(1, 2, 1)
plt.plot(np.linspace(0,shapeX[1],shapeX[1]+1), MSE, 'g', lw=2)
plt.xlabel(r'Model Complexity ($\beta_i$)')
plt.ylabel('MSE')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.linspace(0,shapeX[1],shapeX[1]+1), R2, 'r', lw=2)
plt.xlabel(r'Model Complexity ($\beta_i$)')
plt.ylabel('R2 score')
plt.grid(True)

plt.tight_layout()
plt.show()
    
#plt.plot(X_test, y_noise, "c", label="$noise(x)$")

##########################################################################################################################################
# 5. Computing the metrics of the predictions and model assessment using bootstrap (with leave one out estimate)
# Defining the data and design matrix to bootstrap

X = basis_exp_2d(xf1,yf1,order=5)
Z_n = Z + 0.1*np.max(Z)*np.random.randn(shapez[0]*shapez[1],1) 
Data_vector   = Z_n
shapeX = X.shape
Design_matrix = X[:,0:2] 

# Computing model assessment
Err_train = np.zeros([shapeX[1]-1,1]) 
Err_test  = np.zeros([shapeX[1]-1,1])

for i in range(shapeX[1]-1):
    Err_train[i], Err_test[i] = k_fold_CV(Data_vector,Design_matrix[:,0:(i+1)],k=2)

plt.plot(np.linspace(0,42,43), Err_train, 'g', np.linspace(0,42,43), Err_test, 'r', lw=2)
plt.xlabel(r'Model Complexity')
plt.ylabel('Prediction Error')
plt.legend(('Training set','Test set'))
plt.grid(True)
  
      
##############################################################################
# 5. Testing interpolation using the estimated parameters  

# Make data.
x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)
x, y = np.meshgrid(x,y)

shapex = x.shape
shapey = y.shape

xf1 = x.flatten().reshape(shapex[0]*shapex[1],1)
yf1 = y.flatten().reshape(shapey[0]*shapey[1],1)

X  = basis_exp_2d(xf1,yf1,basis=5,alt=1)
X0 = np.ones([shapex[0]*shapey[0],1])

Xd = np.c_[X0,X]

z_pred_int = np.dot(Xd,beta)
z_pred_int = z_pred_int.reshape(x.shape)


display_fig(z_pred_int,x,y,save=False)

'''
b)###################################################################################################################################
'''

#### Ridge regression and basis expansions with x,y correlations #####

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)

shapex = x.shape
shapey = y.shape

xf1 = x.flatten().reshape(shapex[0]*shapex[1],1)
yf1 = y.flatten().reshape(shapey[0]*shapey[1],1)
Z   = z.flatten().reshape(shapex[0]*shapex[1],1)

X = basis_exp_2d(xf1,yf1,basis=5,alt=1)

z_pred, beta = ridge_least_square(X, Z, lamb=0.5)
z_pred       = z_pred.reshape(z.shape)



#plt.savefig('Predicted_function_basis5_exp_xy_cor_ridge.png', dpi=600)


'''
c)
'''











