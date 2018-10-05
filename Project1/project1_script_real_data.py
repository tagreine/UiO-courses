# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:38:37 2018

@author: tlgreiner
"""


from tools import least_square, basis_exp_2d, display_fig2, Metrics_param, MSE_metric, R2_metric, k_fold_CV
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from imageio import imread
from sklearn.preprocessing import PolynomialFeatures

# Load the terrain
terrain1 = imread('n59_e010_1arc_v3.tif')
# Show the terrain


sub1 = 150
sub2 = 350

Data_org = terrain1[sub1:sub2,sub1:sub2]
Data = Data_org/np.max(Data_org)

shapeT = Data.shape

x = np.linspace(0, shapeT[1]-1,shapeT[1])
y = np.linspace(0, shapeT[0]-1, shapeT[0])
x, y = np.meshgrid(x,y)

shapex = x.shape
shapey = y.shape

plt.figure()
plt.subplot(1,2,1)
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1,2,2)
plt.imshow(Data, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
#plt.savefig('Terrain_and_sub_tarrain.png', dpi=600)
plt.show()


# 1. Predict real data using Ridge

xf1 = x.flatten().reshape(shapex[0]*shapex[1],1)
yf1 = y.flatten().reshape(shapey[0]*shapey[1],1)
Dataf = Data.flatten().reshape(shapeT[0]*shapeT[1],1)

X = np.c_[xf1,yf1]
poly = PolynomialFeatures(8,include_bias=False)
X = poly.fit_transform(X)


#X = basis_exp_2d(xf1,yf1,order=8)

# Estimate the rank before choosing the model
rank = np.linalg.matrix_rank(np.dot(X.T,X))

z_pred_ols, beta = least_square(X,Dataf,method='OLS',lamb = 0.00001)
z_pred_ols       = z_pred_ols.reshape(Data.shape)

z_pred_ridge, beta = least_square(X,Dataf,method='Ridge',lamb = 0.00001)
z_pred_ridge       = z_pred_ridge.reshape(Data.shape)

z_pred_lasso, beta = least_square(X,Dataf,method='Lasso',lamb = 0.00001)
z_pred_lasso       = z_pred_lasso.reshape(Data.shape)

display_fig2(Data,x,y, save=True,name='Terrain_sub_norway',title='Terrain')

display_fig2(z_pred_ols,x,y, save=True,name='Predicted_sub_norway_ols',title='Predicted Terrain')
display_fig2(z_pred_ridge,x,y, save=True,name='Predicted_sub_norway_ridge',title='Predicted Terrain')
display_fig2(z_pred_lasso,x,y, save=True,name='Predicted_sub_norway_lasso',title='Predicted Terrain')


plt.figure()
plt.subplot(1, 2, 1)
plt.title('Terrain')
plt.imshow(Data, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 2, 2)
plt.title('Predicted OLS')
plt.imshow(z_pred_ols, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
#plt.savefig('Terrain_and_pred_ols_tarrain.png', dpi=600)
plt.show()


plt.figure()
plt.subplot(1, 2, 1)
plt.title('Terrain')
plt.imshow(Data, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 2, 2)
plt.title('Predicted Ridge')
plt.imshow(z_pred_ols, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('Terrain_and_pred_ridge_tarrain.png', dpi=600)
plt.show()


plt.figure()
plt.subplot(1, 2, 1)
plt.title('Terrain')
plt.imshow(Data, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 2, 2)
plt.title('Predicted Lasso')
plt.imshow(z_pred_ols, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('Terrain_and_pred_lasso_tarrain.png', dpi=600)
plt.show()





# Model assessment

Data_vector   = Dataf
Design_matrix = X
shapeX        = X.shape 

# Defining model assessment data
Err_train  = np.zeros([shapeX[1]-1,1]) 
Err_test   = np.zeros([shapeX[1]-1,1])
Var_train  = np.zeros([shapeX[1]-1,1])
Var_test  = np.zeros([shapeX[1]-1,1])  
Bias2_train = np.zeros([shapeX[1]-1,1])
Bias2_test = np.zeros([shapeX[1]-1,1])

# The k-fold cross validation algortihm. Computed the Prediction error, variance and bias^2 for 
# both the training and test set

Method = 'Ridge' 
for i in range(shapeX[1]-1):
    Err_train[i], Err_test[i], Var_train[i], Bias2_train[i], Var_test[i], Bias2_test[i] = k_fold_CV(Data_vector,Design_matrix[:,0:(i+1)], Method=Method, k=10, lambd = 0.00001)


plt.plot(np.linspace(0,42,43), Err_train, 'g', np.linspace(0,42,43), Err_test, 'r', lw=2)
plt.xlabel(r'Model Complexity ($\beta_i$)')
plt.ylabel('Prediction Error')
plt.legend(('Training set','Test set'))
plt.grid(True)


plt.tight_layout()
plt.savefig('Prediction_erro_bias_variance_ridge.png', dpi=600)
plt.show()




