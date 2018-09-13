# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:37:47 2018

@author: tlgreiner
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from tools import ord_least_square, FrankeFunction, ridge_least_square

fig = plt.figure()
ax  = fig.gca(projection='3d')


# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# Testing


xf = x.flatten()
yf = y.flatten()
zf = z.flatten()


X = np.c_[xf, xf**2, xf**3, xf**4, xf**5, yf, yf**2, yf**3, yf**4, yf**5,2*np.sin(xf),2*np.sin(yf)]


#z_pred = ord_least_square(X,zf).reshape(z.shape)
z_pred = ridge_least_square(X, zf, lamb = 0.0001).reshape(z.shape)

fig = plt.figure()
ax  = fig.gca(projection='3d')
surf_pred = ax.plot_surface(x, y, z_pred, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf_pred, shrink=0.5, aspect=5)
plt.show()


















