# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:20:10 2018

@author: tlgreiner
"""


import numpy as np
from random import random, seed
from tools import ord_least_square, FrankeFunction, ridge_least_square, basis_exp_2d, display_fig
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso


# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)
# Plot the surface.

display_fig(z,x,y,save=False)

'''
a)
'''

# 1. OLS with no basis expansions or xy correlations #####

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)

shapex = x.shape
shapey = y.shape
shapez = z.shape

xf1 = x.flatten().reshape(shapex[0]*shapex[1],1)
yf1 = y.flatten().reshape(shapey[0]*shapey[1],1)
Z   = z.flatten().reshape(shapex[0]*shapex[1],1)

X = basis_exp_2d(xf1,yf1,basis=1,alt=2)

z_pred, beta = ord_least_square(X,Z)
z_pred       = z_pred.reshape(z.shape)

display_fig(z_pred,x,y,save=False)

# 2. OLS and basis expansions with and without x,y correlations #####


X = basis_exp_2d(xf1,yf1,basis=5,alt=1)

z_pred, beta = ord_least_square(X,Z)
z_pred       = z_pred.reshape(z.shape)

display_fig(z_pred,x,y,save=False)

# 3. OLS and basis expansions with and without x,y correlations and added noise #####
np.random.seed(1)
Z_n = Z + 0.02*np.max(Z)*np.random.randn(shapez[0]*shapez[1],1)

display_fig(Z_n.reshape(shapez),x,y,save=False)

z_pred_n, beta = ord_least_square(X,Z_n)
z_pred_n       = z_pred.reshape(z.shape)

display_fig(z_pred_n,x,y,save=False)



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
    
    


# 4. Interpolation using the estimated parameters  ######

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
b)
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

# Scikit-learn least squares estimate
#reg     = Ridge(alpha=0.5,fit_intercept=True)

#reg.fit(X,Z)
#z_sci = reg.predict(X)
#z_sci = z_sci.reshape(z.shape)


fig = plt.figure()
ax  = fig.gca(projection='3d')
surf_pred = ax.plot_surface(x, y, z_pred, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.set_title('Predicted function')
# Add a color bar which maps values to colors.
fig.colorbar(surf_pred, shrink=0.5, aspect=5)
plt.show()
#plt.savefig('Predicted_function_basis5_exp_xy_cor_ridge.png', dpi=600)


'''
c)
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


# Scikit-learn least squares estimate
reg     = Lasso(alpha=0.00001,fit_intercept=True,max_iter=1000000)
reg.fit(X,Z)
z_sci   = reg.predict(X)
z_sci   = z_sci.reshape(z.shape)


fig = plt.figure()
ax  = fig.gca(projection='3d')
surf_pred = ax.plot_surface(x, y, z_sci, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.set_title('Predicted function')
# Add a color bar which maps values to colors.
fig.colorbar(surf_pred, shrink=0.5, aspect=5)
plt.show()
#plt.savefig('Predicted_function_basis5_exp_xy_cor_ridge.png', dpi=600)
