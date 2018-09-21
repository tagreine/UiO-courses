# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:29:45 2018

@author: tagreine
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

'''
Metrics #######################################################################
'''

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

def Metrics_param(beta, y, y_, Design_matrix,compute_var=True):
    
    # Computing the variance and standard error of the parameters 
    
    # if the variance is not assumed to be 1
    if compute_var == True:
        shapey_  = y_.shape
        shapeb  = beta.shape
        A       = 1/(shapey_[0] - shapeb[0] - 1)
        B       = np.dot((y - y_).T,(y - y_)) 
        Var     = np.multiply(A,B)
        M       = np.dot(Design_matrix.T,Design_matrix)
        VarBeta = np.multiply(np.linalg.inv(M),Var)
        VarBeta = np.diag(VarBeta).reshape(shapeb)
        StdBeta = np.sqrt(VarBeta).reshape(shapeb)
    
    # if the variance is assumed to be 1
    if compute_var == False:
        shapeb  = beta.shape
        M       = np.dot(Design_matrix.T,Design_matrix)
        VarBeta = np.linalg.inv(M)
        VarBeta = np.diag(VarBeta).reshape(shapeb)
        StdBeta = np.sqrt(VarBeta).reshape(shapeb)
        
    return VarBeta, StdBeta

'''
Regression functions ##########################################################
'''

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
    
    return y_, beta


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
    
    return y_, beta

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
    
    return y_, beta

'''
Other functions ###############################################################
'''

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def basis_exp_2d(x, y, order = 1):
    
    # concatenate the vactors of higher orders
    if order == 1:
        Matrix = np.c_[x, y]
        
    if order == 2:
        Matrix = np.c_[x, y, x*y, x**2, y**2]
    
    if order == 3:
        Matrix = np.c_[x, y, x*y, x**2, y**2, x**3, y**3, (x**2)*y, (y**2)*x]
        
    if order == 4:
        Matrix = np.c_[x, y, x*y, x**2, y**2, x**3, y**3, (x**2)*y, (y**2)*x, x**4, 
                       y**4, (x**3)*y, (y**3)*x, (x**2)*(y**2)]
    
    if order == 5:
        Matrix = np.c_[x, y, x*y, x**2, y**2, x**3, y**3, (x**2)*y, (y**2)*x, 
                       x**4, y**4, (x**3)*y, (y**3)*x, (x**2)*(y**2), x**5, y**5, 
                       (x**4)*y, (y**4)*x, (x**3)*(y**2), (y**3)*(x**2)]
    if order == 6:
        Matrix = np.c_[x, y, x*y, x**2, y**2, x**3, y**3, (x**2)*y, (y**2)*x, 
                       x**4, y**4, (x**3)*y, (y**3)*x, (x**2)*(y**2), x**5, y**5, 
                       (x**4)*y, (y**4)*x, (x**3)*(y**2), (y**3)*(x**2),
                       x**6, y**6, (x**5)*y, (x**4)*(y**2), (x**3)*(y**3), (x**2)*(y**4), (x)*(y**5)]
    if order == 7:
        Matrix = np.c_[x, y, x*y, x**2, y**2, x**3, y**3, (x**2)*y, (y**2)*x, 
                       x**4, y**4, (x**3)*y, (y**3)*x, (x**2)*(y**2), x**5, y**5, 
                       (x**4)*y, (y**4)*x, (x**3)*(y**2), (y**3)*(x**2),
                       x**6, y**6, (x**5)*y, (x**4)*(y**2), (x**3)*(y**3), (x**2)*(y**4), (x)*(y**5),
                       x**7, y**7, (x**6)*y, (x**5)*(y**2), (x**4)*(y**3), (x**3)*(y**4), (x**2)*(y**5),(x)*(y**6)]
    if order == 8:
        Matrix = np.c_[x, y, x*y, x**2, y**2, x**3, y**3, (x**2)*y, (y**2)*x, 
                       x**4, y**4, (x**3)*y, (y**3)*x, (x**2)*(y**2), x**5, y**5, 
                       (x**4)*y, (y**4)*x, (x**3)*(y**2), (y**3)*(x**2),
                       x**6, y**6, (x**5)*y, (x**4)*(y**2), (x**3)*(y**3), (x**2)*(y**4), (x)*(y**5),
                       x**7, y**7, (x**6)*y, (x**5)*(y**2), (x**4)*(y**3), (x**3)*(y**4), (x**2)*(y**5), (x)*(y**6),
                       x**8, y**8, (x**7)*y, (x**6)*(y**2), (x**5)*(y**3), (x**4)*(y**4), (x**3)*(y**5), (x**2)*(y**6), (x)*(y**7)]
    
    return Matrix
        
def display_fig(z,x,y, save=False,name=None,title=None):
    
    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    surf_pred = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    ax.set_title(title)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf_pred, shrink=0.5, aspect=5)
    
    if save == True:
       plt.savefig(name +'.png', dpi=600)
       
    if save == False:
       plt.show()
       
'''
Model assessment ##############################################################
'''

def k_fold_CV(Data_vector, Design_matrix, k=2):
    
    shapeD  = Data_vector.shape
    shapeDX = Design_matrix.shape 
    n      = shapeD[0]
    m      = shapeD[1]
    N      = shapeDX[0]
    M      = shapeDX[1] 
    # random shuffle of the data before splitting
    # define indexes to shuffle
    np.random.seed(0)
    CVind = np.random.choice(n,n,replace=False).reshape(n,1)
    
    Data_new = np.zeros([n,m])
    Design_matrix_new = np.zeros([N,M])
    # create new shuffled data and design matrix
    for i in range(n):
        for j in range(M):
            Data_new[i,:]   = Data_vector[CVind[i],:]
            Design_matrix_new[i,j] = Design_matrix[CVind[i],j]
    
    # Data-splitting in k parts and k rounds for training and test error computation
    # First define the K range
    K = int(n/k)    
    Ks = np.zeros([k+1,1])
    for j in range(k+1):       
        Ks[j,:] = j*K
    
    Data_train   = np.zeros([int(n-K),m,k])
    Data_test    = np.zeros([int(K),m,k])
    Matrix_train = np.zeros([int(n-K),M,k])
    Matrix_test  = np.zeros([int(K),M,k])
    for j in range(k):
        a = 0
        b = 0
        for i in range(n):
            
            if (i < Ks[j]) or (i > Ks[j+1] - 1):
                Data_train[a,:,j]   = Data_new[i,:]
                Matrix_train[a,:,j] = Design_matrix_new[i,:]
                a = a + 1
            
            if (Ks[j] <= i <= Ks[j+1] - 1):
                Data_test[b,:,j]   = Data_new[i,:]
                Matrix_test[b,:,j] = Design_matrix_new[i,:]
                b = b + 1
    
    # define training data, test data and metrics
    beta_train = np.zeros([M + 1,m,k])
    Pred_train = np.zeros(Data_train.shape)
    Pred_test  = np.zeros(Data_test.shape)
    Error_train_k = np.zeros([n-K,1,k])
    Error_test_k  = np.zeros([K,1,k])
    
    for i in range(k):
        Pred_train[:,:,i], beta_train[:,:,i] = ord_least_square(Matrix_train[:,:,i],Data_train[:,:,i])
        Pred_test[:,:,i] = np.dot(np.c_[np.ones([K,1]) , Matrix_test[:,:,i]],beta_train[:,:,i])
        
        # compute the mean squared error for each round
        #Error_train_k[i] = MSE_metric(Data_train[:,:,i],Pred_train[:,:,i]) 
        #Error_test_k[i]  = MSE_metric(Data_test[:,:,i],Pred_test[:,:,i])
        Error_train_k[:,:,i] = (Data_train[:,:,i] - Pred_train[:,:,i])**2 
        Error_test_k[:,:,i]  = (Data_test[:,:,i] - Pred_test[:,:,i])**2
      
    # compute the average of the Errors
    #Err_train = (1/k)*np.sum(Error_train_k) 
    #Err_test  = (1/k)*np.sum(Error_test_k)
  
    Err_train = np.mean(np.sum(Error_train_k,axis=2)) 
    Err_test  = np.mean(np.sum(Error_test_k,axis=2)) 
    
    # compute the bias and variance 
    
    pred_k = np.zeros([n-K,1])
    data_k = np.zeros([n-K,1])
    
    for i in range(k):
        pred_k[:,:]   = pred_k[:,:] + Pred_train[:,:,i]
        data_k[:,:]   = data_k[:,:] + Data_train[:,:,i]

    pred  = (1/k)*pred_k
    data  = (1/k)*data_k
    
    Bias2 = np.mean((data - pred)**2)
    Var   = np.mean(pred**2) - np.mean(pred)**2
    
    return Err_train, Err_test, Var, Bias2







# function downloaded from https://studywolf.wordpress.com/2017/11/21/matplotlib-legends-for-mean-and-confidence-interval-plots/
class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed
 
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)
 
        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)
 
        return patch

# function downloaded from https://studywolf.wordpress.com/2017/11/21/matplotlib-legends-for-mean-and-confidence-interval-plots/
def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)












