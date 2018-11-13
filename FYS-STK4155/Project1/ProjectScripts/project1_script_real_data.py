# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:12:05 2018

@author: tagreine
"""

import numpy as np
from tools import least_square, ord_least_square, FrankeFunction, ridge_least_square, basis_exp_2d, display_fig2, LegendObject, plot_mean_and_CI, Metrics_param, MSE_metric, R2_metric, k_fold_CV
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter as cc
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from imageio import imread

# Load the terrain
terrain1 = imread('n59_e010_1arc_v3.tif')
# Show the terrain

# Define sub version of the terrain
sub1 = 150
sub2 = 200

# Extract the part of the terrain
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


xf1 = x.flatten().reshape(shapex[0]*shapex[1],1)
yf1 = y.flatten().reshape(shapey[0]*shapey[1],1)
Dataf = Data.flatten().reshape(shapeT[0]*shapeT[1],1)

X = np.c_[xf1,yf1]
X = basis_exp_2d(xf1,yf1,order=8)


################################################################################################
# 2. computing booststrap model assessment using OLS

deg = 12
Data_vector   = Dataf
x_matrix = xf1

x_train, x_test, y_train, y_test = train_test_split(x_matrix, Data_vector, test_size=0.2)

error    = np.zeros([deg])
bias     = np.zeros([deg])
variance = np.zeros([deg])

error_t    = np.zeros([deg])
bias_t     = np.zeros([deg])
variance_t = np.zeros([deg])


n_boostraps = 200

for j in range(deg):   

    model = make_pipeline(PolynomialFeatures(degree=j), LinearRegression(fit_intercept=True))

    y_pred   = np.empty((y_train.shape[0], n_boostraps))
    y_pred_t = np.empty((y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_, y_ = resample(x_train, y_train)

        # Predict training and test data for each bootstrap
        y_pred[:, i]   = np.ravel(model.fit(x_, y_).predict(x_train))
        y_pred_t[:, i] = np.ravel(model.fit(x_, y_).predict(x_test))
        
    # Compute the error, variance and bias squared at eah point in the model    
    error[j]    = np.mean( np.mean((y_train - y_pred)**2, axis=1, keepdims=True) )
    bias[j]     = np.mean( (y_train - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance[j] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    
    error_t[j]    = np.mean( np.mean((y_test - y_pred_t)**2, axis=1, keepdims=True) )
    bias_t[j]     = np.mean( (y_test - np.mean(y_pred_t, axis=1, keepdims=True))**2 )
    variance_t[j] = np.mean( np.var(y_pred_t, axis=1, keepdims=True) )


plt.subplot(1, 2, 1)
plt.plot(np.linspace(1,deg,deg), error, 'g', np.linspace(1,deg,deg), error_t, 'r', lw=2)
plt.xlabel(r'Model Complexity (Polynomial order)')
plt.ylabel('Prediction Error')
plt.legend(('Training set','Test set'))
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(np.linspace(1,deg,deg), variance, 'k', np.linspace(1,deg,deg), variance_t, 'c', np.linspace(1,deg,deg), bias, 'y', np.linspace(1,deg,deg), bias_t, 'm', lw=2)
plt.xlabel(r'Model Complexity (Polynomial order)')
#plt.ylabel('Variance')
plt.legend(('Training variance','Test variance','Training bias','Test bias'))
plt.grid(True)

plt.tight_layout()
plt.savefig('Prediction_erro_bias_variance_ols_terrain.png', dpi=600)
plt.show()


##########################################################################################################################################
# 3. Computing metrics of the predictions and model assessment using bootstrap and Ridge/Lasso estimate


Alpha = np.linspace(0,0.00009,20)
k     = len(Alpha)
Data_vector = Dataf
x_matrix = xf1

x_train, x_test, y_train, y_test = train_test_split(x_matrix, Data_vector, test_size=0.2)

error    = np.zeros([k])
bias     = np.zeros([k])
variance = np.zeros([k])

error_t    = np.zeros([k])
bias_t     = np.zeros([k])
variance_t = np.zeros([k])


n_boostraps = 1000

for j in range(len(Alpha)):   

    #model = make_pipeline(PolynomialFeatures(degree=12), Ridge(alpha=Alpha[j],fit_intercept=True))
    model = make_pipeline(PolynomialFeatures(degree=12), Lasso(alpha=Alpha[j],fit_intercept=True))

    y_pred   = np.empty((y_train.shape[0], n_boostraps))
    y_pred_t = np.empty((y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_, y_ = resample(x_train, y_train)

        # Predict training and test data for each bootstrap
        y_pred[:, i]   = np.ravel(model.fit(x_, y_).predict(x_train))
        y_pred_t[:, i] = np.ravel(model.fit(x_, y_).predict(x_test))
        
    # Compute the error, variance and bias squared at eah point in the model    
    error[j]    = np.mean( np.mean((y_train - y_pred)**2, axis=1, keepdims=True) )
    bias[j]     = np.mean( (y_train - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance[j] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    
    error_t[j]    = np.mean( np.mean((y_test - y_pred_t)**2, axis=1, keepdims=True) )
    bias_t[j]     = np.mean( (y_test - np.mean(y_pred_t, axis=1, keepdims=True))**2 )
    variance_t[j] = np.mean( np.var(y_pred_t, axis=1, keepdims=True) )


plt.subplot(1, 2, 1)
plt.plot(Alpha[0:10]*10000, error[0:10], 'g', Alpha[0:10]*10000, error_t[0:10], 'r', lw=2)
plt.xlabel(r'$\lambda$*10^4')
plt.ylabel('Prediction Error')
plt.legend(('Training set','Test set'))
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(Alpha[0:10]*10000, variance[0:10], 'k', Alpha[0:10]*10000, variance_t[0:10], 'c', Alpha[0:10]*10000, bias[0:10], 'y', Alpha[0:10]*10000, bias_t[0:10], 'm', lw=2)
plt.xlabel(r'$\lambda$*10^4')
#plt.ylabel('Variance')
plt.legend(('Training variance','Test variance','Training bias','Test bias'))
plt.grid(True)

plt.tight_layout()
#plt.savefig('Prediction_erro_bias_variance_Lasso_terrain.png', dpi=600)
plt.show()


################################################################################################
# 4. Compute the predictions using new polynomial fit

poly = PolynomialFeatures(10,include_bias=False)
X = poly.fit_transform(np.c_[xf1,yf1])

z_pred_ols, beta = least_square(X,Dataf,method='OLS',lamb = 0.00001)
z_pred_ols       = z_pred_ols.reshape(Data.shape)

z_pred_ridge, beta = least_square(X,Dataf,method='Ridge',lamb = 0.00001)
z_pred_ridge       = z_pred_ridge.reshape(Data.shape)

z_pred_lasso, beta = least_square(X,Dataf,method='Lasso',lamb = 0.00001)
z_pred_lasso       = z_pred_lasso.reshape(Data.shape)

display_fig2(Data,x,y, save=True,name='Terrain_sub_norway',title='Terrain')

display_fig2(z_pred_ols,x,y, save=True,name='Predicted_sub_norway_ols_2',title='Predicted Terrain')
display_fig2(z_pred_ridge,x,y, save=True,name='Predicted_sub_norway_ridge_2',title='Predicted Terrain')
display_fig2(z_pred_lasso,x,y, save=True,name='Predicted_sub_norway_lasso_2',title='Predicted Terrain')


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
plt.savefig('Terrain_and_pred_ols_tarrain_2.png', dpi=600)
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
plt.savefig('Terrain_and_pred_ridge_tarrain_2.png', dpi=600)
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
plt.savefig('Terrain_and_pred_lasso_tarrain_2.png', dpi=600)
plt.show()

#X = basis_exp_2d(xf1,yf1,order=8)

################################################################################################

# 5. Compute MSE and R2 metrics ################################################################


OLS_mse   =  MSE_metric(Dataf,z_pred_ols.reshape(Dataf.shape[0],1))
OLS_R2    =  R2_metric(Dataf,z_pred_ols.reshape(Dataf.shape[0],1))

Ridge_mse = MSE_metric(Dataf,z_pred_ridge.reshape(Dataf.shape[0],1))
Ridge_R2  = R2_metric(Dataf,z_pred_ridge.reshape(Dataf.shape[0],1))

Lasso_mse = MSE_metric(Dataf,z_pred_lasso.reshape(Dataf.shape[0],1))
Lasso_R2  = R2_metric(Dataf,z_pred_lasso.reshape(Dataf.shape[0],1))


##############################################################################################
# 6. compute the confidence intervals of the beta parameters ##################

Method = 'Lasso'
Data_vector   = Dataf
Pred_vector, beta = least_square(X,Dataf, method = Method, lamb = 0.00001)
shapeX        = X.shape
Design_matrix = np.c_[np.ones([shapeX[0],1]), X]

# Computing the variance and standard error of the estimated least squares parameters    
VarBeta, StdBeta = Metrics_param(beta, Data_vector, Pred_vector.reshape(shapeX[0],1), Design_matrix,compute_var=True)

# Computing the confidence intervals
betalow  = beta - 2*StdBeta
betahigh = beta + 2*StdBeta

# Estimated Beta, Beta + Se and Beta - Se 
fig = plt.figure(1, figsize=(7, 2.5))
plot_mean_and_CI(beta.reshape(shapeX[1] + 1,), betahigh.reshape(shapeX[1] + 1,), betalow.reshape(shapeX[1] + 1,), color_mean='k', color_shading='k')

bg = np.array([1, 1, 1]) 
colors = ['black', 'blue', 'green']
colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]
 
plt.legend([0], [r'$\hat \beta_i$$\pm$SE[$\hat \beta_i$]'],
           handler_map={
               0: LegendObject(colors[0], colors_faded[0])
            })
plt.xlabel(r'$i$')
#plt.title('')
plt.tight_layout()
plt.grid()
plt.savefig('Parameter_confidence_interval_Lasso_terrain.png', dpi=600)
plt.show()


################################################################################################




