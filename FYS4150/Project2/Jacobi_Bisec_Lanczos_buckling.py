# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:40:04 2020

@author: tlgreiner
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.linalg import toeplitz
from Compfys_functions import Functions
import time


############# eigenvalues of jacobi, bisection,numpy ################

N  = [10,20,50,70]
J  = []
B  = []
Nu = []
for i in N:
    rho_max = 1
    rho_min = 0
    
    h       = (rho_max-rho_min)/i

    rho = np.linspace(rho_min,rho_max,i)

    # Tridiagonal matrix setup
    a1 = np.array([2/(h**2),-1/h**2])
    a2 = np.zeros([i-2])
    a  = np.concatenate([a1,a2])   
    A  = toeplitz(a, a)

    # Normalize to get orthonormal matrix for Jacobi
    V = A.copy()
    for j in range(i):
        V[:,j] = V[:,j]/np.linalg.norm(V[:,j])


    ##############
    
    lmbda_an = np.zeros(i)
    for j in range(i):
        lmbda_an[j] = A[j,j] + 2*A[0,1]*np.cos(j*np.pi/i)    
    
    time_start_B = time.clock()
    eigB = Functions.bisect_eig(A,n_pol=i)
    eigB = Functions.sort_eig_val(eigB)
    time_elapsed_B = (time.clock() - time_start_B)
    B.append(time_elapsed_B)
 
    time_start_Nu = time.clock()
    eigNP,eigNPv = np.linalg.eig(A)	
    eigNP = Functions.sort_eig_val(eigNP)
    time_elapsed_Nu = (time.clock() - time_start_Nu)
    Nu.append(time_elapsed_Nu)

    time_start_J = time.clock()
    eigJ,eigJv = Functions.jacobi(A,V,n=i,conv=10**-8)
    eigJ,eigJv = Functions.sort_eig(eigJ,eigJv)  
    eigJ = Functions.sort_eig_val(eigJ)
    time_elapsed_J = (time.clock() - time_start_J)
    J.append(time_elapsed_J)
    
    #plt.figure()
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('N={}'.format(i))
    
    plt.scatter(np.arange(i),lmbda_an[::-1], marker='x')
    plt.scatter(np.arange(i),eigJ,     marker='*')
    plt.scatter(np.arange(i),eigB,     marker='+')
    plt.scatter(np.arange(i),eigNP,    marker='o')
    
    plt.legend(['Analytical','Jacobi','Bisection','Numpy'])
    plt.ylabel('Eigenvalue')
    plt.xlabel('$\lambda$')
    plt.grid()
    plt.show()


############# eigenvalues of jacobi, bisection+lanczos,numpy ################


N  = [10,20,50,70]
J  = []
B  = []
Nu = []
for i in N:
    rho_max = 1
    rho_min = 0
    
    h       = (rho_max-rho_min)/i

    rho = np.linspace(rho_min,rho_max,i)

    # Tridiagonal matrix setup
    a1 = np.array([2/(h**2),-1/h**2])
    a2 = np.zeros([i-2])
    a  = np.concatenate([a1,a2])   
    A  = toeplitz(a, a)

    # Normalize to get orthonormal matrix for Jacobi
    V = A.copy()
    for j in range(i):
        V[:,j] = V[:,j]/np.linalg.norm(V[:,j])


    ############## 
    lmbda_an = np.zeros(i)
    for j in range(i):
        lmbda_an[j] = A[j,j] + 2*A[0,1]*np.cos(j*np.pi/i)       

    time_start_B = time.clock()
    AB   = Functions.lanczos(A,random_init='False')
    AB   = Functions.sort_mat(AB,rotate='False')
    eigB = Functions.bisect_eig(AB,n_pol=i)
    eigB = Functions.sort_eig_val(eigB)
    time_elapsed_B = (time.clock() - time_start_B)
    B.append(time_elapsed_B)
    
    time_start_Nu = time.clock()
    eigNP,eigNPv = np.linalg.eig(A)	
    eigNP = Functions.sort_eig_val(eigNP)
    time_elapsed_Nu = (time.clock() - time_start_Nu)
    Nu.append(time_elapsed_Nu)

    time_start_J = time.clock()
    eigJ,eigJv = Functions.jacobi(A,V,n=i,conv=10**-8)
    eigJ,eigJv = Functions.sort_eig(eigJ,eigJv)  
    eigJ = Functions.sort_eig_val(eigJ)
    time_elapsed_J = (time.clock() - time_start_J)
    J.append(time_elapsed_J)
    
    #plt.figure()
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('N={}'.format(i))
    
    plt.scatter(np.arange(i),lmbda_an[::-1], marker='x')
    plt.scatter(np.arange(i),eigJ,marker='*')
    plt.scatter(np.arange(i),eigB,marker='+')
    plt.scatter(np.arange(i),eigNP,marker='o')
    
    plt.legend(['Analytical','Jacobi','Bisection+Lanczos','Numpy'])
    plt.ylabel('Eigenvalue')
    plt.xlabel('$\lambda$')
    plt.grid()
    plt.show()