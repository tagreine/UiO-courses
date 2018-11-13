# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from tools import least_square, ord_least_square, FrankeFunction, ridge_least_square, basis_exp_2d, display_fig, LegendObject, plot_mean_and_CI, Metrics_param, MSE_metric, R2_metric, k_fold_CV
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter as cc
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample


# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)
# Plot the surface.

display_fig(z,x,y,save=False, name='Franke_function',title='Franke function')
#display_fig(z,x,y,save=True,name='Franke_function',title='Franke function')



################################################################################################
# 1. Predictions with basis expansions and added noise ################################

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
Z_n = Z + 0.1*np.max(Z)*np.random.randn(shapez[0]*shapez[1],1)

X = basis_exp_2d(xf1,yf1,order=8)

z_pred, beta = ord_least_square(X,Z_n)
z_pred       = z_pred.reshape(z.shape)


# OLS
display_fig(Z_n.reshape(shapez),x,y,save=False, name='Franke_function_noise',title='Franke function (added noise)')
display_fig(z_pred,x,y,save=False)
#display_fig(Z_n.reshape(shapez),x,y,save=True,name='Franke_function_noise',title='Franke function with noise')
#display_fig(z_pred,x,y,save=True,name='Predicted_8order_nnoise',title='Predicted function')

# Ridge
z_pred_ridge, beta_ridge = least_square(X,Z_n,method='Ridge',lamb = .0000001)
z_pred_ridge             = z_pred_ridge.reshape(z.shape)

display_fig(z_pred_ridge,x,y,save=False, name='',title='')
#display_fig(z_pred_ridge,x,y,save=True,name='Predicted_8order_ridge',title='Predicted function')

# Lasso
z_pred_lasso, beta_lasso = least_square(X,Z_n,method='Lasso',lamb = 0.00001)
z_pred_lasso             = z_pred_lasso.reshape(z.shape)

display_fig(z_pred_lasso,x,y,save=False, name='',title='')
#display_fig(z_pred_lasso,x,y,save=True,name='Predicted_8order_lasso',title='Predicted function')



################################################################################################

#2. Compute MSE and R2 metrics ################################################################

OLS_mse   =  MSE_metric(Z_n,z_pred.reshape(shapez[0]*shapez[1],1))
OLS_R2    =  R2_metric(Z_n,z_pred.reshape(shapez[0]*shapez[1],1))

Ridge_mse = MSE_metric(Z_n,z_pred_ridge.reshape(shapez[0]*shapez[1],1))
Ridge_R2  = R2_metric(Z_n,z_pred_ridge.reshape(shapez[0]*shapez[1],1))

Lasso_mse = MSE_metric(Z_n,z_pred_lasso.reshape(shapez[0]*shapez[1],1))
Lasso_R2  = R2_metric(Z_n,z_pred_lasso.reshape(shapez[0]*shapez[1],1))
    
#plt.plot(X_test, y_noise, "c", label="$noise(x)$")

################################################################################################

# 3. compare the beta values ##################################################

Pred_X, beta_ols   = least_square(X,Z_n, method = 'OLS', lamb = 0.00001)
Pred_X, beta_ridge = least_square(X,Z_n, method = 'Ridge', lamb = 0.00001)
Pred_X, beta_lasso = least_square(X,Z_n, method = 'Lasso', lamb = 0.00001)


plt.plot(np.linspace(0,44,45), beta_ols, 'g', np.linspace(0,44,45), beta_ridge, 'r',np.linspace(0,44,45), beta_lasso, 'k', lw=2)
plt.xlabel(r'$i$')
plt.ylabel(r'$\beta_i$')
plt.legend(('OLS','RIdge','Lasso'))
plt.grid(True)
#plt.savefig('Beta_comparison.png', dpi=600)


################################################################################################
# 4. compute the confidence intervals of the beta parameters ##################

Method = 'Lasso'
Data_vector   = Z_n
Pred_vector, beta = least_square(X,Z_n, method = Method, lamb = 0.00001)
shapeX        = X.shape
Design_matrix = np.c_[np.ones([shapeX[0],1]), X]

# Computing the variance and standard error of the estimated least squares parameters    
VarBeta, StdBeta = Metrics_param(beta, Data_vector, Pred_vector.reshape(shapeX[0],1), Design_matrix,compute_var=True)

# Computing the confidence intervals
betalow  = beta - StdBeta
betahigh = beta + StdBeta

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
#plt.savefig('Parameter_confidence_interval_Lasso00001.png', dpi=600)
plt.show()


################################################################################################

# 5. Computing MSE and R2 wrt model complexity ###############################
shapeX = X.shape
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
plt.savefig('MSE_R2_score_wrt_model_comp.png', dpi=600)
plt.show()

###############################################################################################

# Computing ridge MSE and R2 wrt model penalty parameter and different noise setup

lambd = np.arange(0, 0.05, 0.00001) 
MSE1    = np.zeros([lambd.shape[0],1])
R21     = np.zeros([lambd.shape[0],1])
MSE2    = np.zeros([lambd.shape[0],1])
R22     = np.zeros([lambd.shape[0],1])
MSE3    = np.zeros([lambd.shape[0],1])
R23     = np.zeros([lambd.shape[0],1])

Z_n1 = Z_n
Z_n2 = Z + 0.05*np.max(Z)*np.random.randn(shapez[0]*shapez[1],1)
Z_n3 = Z + 0.15*np.max(Z)*np.random.randn(shapez[0]*shapez[1],1)

Method = 'Ridge'

for i in range(lambd.shape[0]):
    pred1, beta = least_square(X,Z_n1, method = Method, lamb = lambd[i])

    MSE1[i]   = MSE_metric(Z_n1,pred1)
    R21[i]    = R2_metric(Z_n1,pred1)

    pred2, beta = least_square(X,Z_n2, method = Method, lamb = lambd[i])

    MSE2[i]   = MSE_metric(Z_n2,pred2)
    R22[i]    = R2_metric(Z_n2,pred2)
    
    pred3, beta = least_square(X,Z_n3, method = Method, lamb = lambd[i])

    MSE3[i]   = MSE_metric(Z_n3,pred3)
    R23[i]    = R2_metric(Z_n3,pred3)    
    

plt.subplot(1, 2, 1)
plt.plot(lambd, MSE1, 'g', lambd, MSE2, 'k', lambd, MSE3, 'y', lw=2)
plt.xlabel(r'$\lambda$')
plt.ylabel('MSE')
plt.legend(('Medium noise','Low noise','High noise'))
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(lambd, R21, 'r', lambd, R22, 'b', lambd, R23, 'c', lw=2)
plt.xlabel(r'$\lambda$')
plt.ylabel('R2 score')
plt.legend(('Medium noise','Low noise','High noise'))
plt.grid(True)

plt.tight_layout()
plt.savefig('MSE_R2_ridge_score_wrt_model_lamb_noise.png', dpi=600)
#plt.savefig('MSE_R2_lasso_score_wrt_model_lamb_noise.png', dpi=600)
plt.show()


##########################################################################################################################################
# 7. Computing metrics of the predictions and model assessment using bootstrap and OLS


deg = 12
Data_vector   = Z
x_matrix = xf1

x_train, x_test, y_train, y_test = train_test_split(x_matrix, Data_vector, test_size=0.2)

error    = np.zeros([deg])
bias     = np.zeros([deg])
variance = np.zeros([deg])

error_t    = np.zeros([deg])
bias_t     = np.zeros([deg])
variance_t = np.zeros([deg])


n_boostraps = 500

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
plt.savefig('Prediction_erro_bias_variance_ols.png', dpi=600)
plt.show()


##########################################################################################################################################
# 8. Computing metrics of the predictions and model assessment using bootstrap and Ridge/Lasso estimate


Alpha = np.linspace(0,0.00009,20)
k     =len(Alpha)
Data_vector   = Z
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
plt.savefig('Prediction_erro_bias_variance_lasso.png', dpi=600)
plt.show()



'''
Testing interpolation using the estimated parameters###############################################
'''
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)

shapex = x.shape
shapey = y.shape

xf1 = x.flatten().reshape(shapex[0]*shapex[1],1)
yf1 = y.flatten().reshape(shapey[0]*shapey[1],1)
Z   = z.flatten().reshape(shapex[0]*shapex[1],1)
Z_n = Z + 0.1*np.max(Z)*np.random.randn(shapez[0]*shapez[1],1)

X = basis_exp_2d(xf1,yf1,order = 8)

z_pred_n, beta = least_square(X,Z,method='Ridge',lamb = 0.001)
z_pred_n       = z_pred_n.reshape(z.shape)

############## interpolation ######################
xint = np.arange(0, 1, 0.001)
yint = np.arange(0, 1, 0.001)
xint, yint = np.meshgrid(xint,yint)

shapexint = xint.shape
shapeyint = yint.shape

xf1int = xint.flatten().reshape(shapexint[0]*shapexint[1],1)
yf1int = yint.flatten().reshape(shapeyint[0]*shapeyint[1],1)

X0 = np.ones([shapeyint[0]*shapeyint[1],1])
X1 = basis_exp_2d(xf1int,yf1int,order = 8)
Xint = np.c_[X0,X1]

z_pred_int = np.dot(Xint,beta)
z_pred_int = z_pred_int.reshape(xint.shape)


display_fig(z_pred_int,xint,yint,save=False)









