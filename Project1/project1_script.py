# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from tools import ord_least_square, FrankeFunction, ridge_least_square, basis_exp_2d
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso


# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)
# Plot the surface.
fig = plt.figure()
ax  = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.set_title('Franke function')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()
plt.savefig('franke_function.png', dpi=600)

'''
a)
'''

#### OLS with no basis expansions or xy correlations #####

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)

shapex = x.shape
shapey = y.shape

xf1 = x.flatten().reshape(shapex[0]*shapex[1],1)
yf1 = y.flatten().reshape(shapey[0]*shapey[1],1)
Z   = z.flatten().reshape(shapex[0]*shapex[1],1)

X = basis_exp_2d(xf1,yf1,basis=1,alt=2)

z_pred, beta = ord_least_square(X,Z)
z_pred       = z_pred.reshape(z.shape)

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
#plt.show()
plt.savefig('Predicted_function_linear.png', dpi=600)


#### OLS and basis expansions with and without x,y correlations #####


X = basis_exp_2d(xf1,yf1,basis=5,alt=1)

z_pred, beta = ord_least_square(X,Z)
z_pred       = z_pred.reshape(z.shape)


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
#plt.show()
plt.savefig('Predicted_function_basis5_exp_xy_cor.png', dpi=600)


##### Interpolation using the estimated parameters  ######

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
z_pred_int       = z_pred_int.reshape(x.shape)

fig = plt.figure()
ax  = fig.gca(projection='3d')
surf_pred = ax.plot_surface(x, y, z_pred_int, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.set_title('Interpolated function')
# Add a color bar which maps values to colors.
fig.colorbar(surf_pred, shrink=0.5, aspect=5)
#plt.show()
plt.savefig('Predicted_function_basis5_exp_xy_cor_int.png', dpi=600)


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









