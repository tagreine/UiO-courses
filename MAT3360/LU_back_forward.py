# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:33:41 2019

@author: tagreine
"""

import numpy as np
import scipy.linalg as la

# Solution of Ax=b through forward substitution. Only for lower triangular matrix A
def Forward_sub(A,b):
    
    A = A.astype(np.float)
    b = b.astype(np.float)
    
    # Size of A
    N,M = A.shape
    n,m = b.shape
    
    if N != M:
        print('Error: matrix is not square')
        
    if n != M:
        print('Error: input does not match, b.shape[0]!=A.shape[1]')
        
    x = np.zeros_like(b)
    x = x.astype(np.float)
    if np.abs(A[0,0]) > np.exp(-12):
        # Solve for the first element in x
        x[0] = b[0] / A[0,0] 
    else:
        print('Error: matrix is singular')
        
    
    for k in range(1,n):
        if np.abs(A[k,k]) > np.exp(-12):
            
            temp = 0
            for j in range(0,k):
                temp = temp + A[k,j] * x[j]
            x[k] = (b[k] - temp)/A[k,k]
            
        else:
            print('input is singular')
            
    return x
    
'''        
A = np.array([[1,0,0],[0,1,0],[1,1,-10]])

b = np.array([[1],[1],[1]])

x = Forward_sub(A,b)
'''

# Solution of Ax=b through forward substitution. Only for upper triangular matrix A
def Backward_sub(A,b):

    A = A.astype(np.float)
    b = b.astype(np.float)
    
    # Size of A
    M,N = A.shape
    m,n = b.shape
    
    if M != N:
        print('Error: matrix is not square')
        
    if m != N:
        print('Error: input does not match, b.shape[0]!=A.shape[1]')
        
    x = np.zeros_like(b)
    x = x.astype(np.float)
    if np.abs(A[M-1,M-1]) > np.exp(-12):
        # Solve for the last element in x
        x[M-1] = b[M-1] / A[M-1,M-1] 
    else:
        print('Error: matrix is singular')
        
    
    for k in range(M-2,-1,-1):
        if np.abs(A[k,k]) > np.exp(-12):
            
            temp = 0
            for j in range(M-1,k-1,-1):
                temp = temp + A[k,j] * x[j]
            x[k] = (b[k] - temp) / A[k,k]
            
        else:
            print('input is singular')
            
    return x
    
'''        
A = np.array([[1,0,1],[0,1,1],[0,0,-2]])

b = np.array([[2],[1],[1]])

x = Backward_sub(A,b)
'''



A = np.array([[8,6,-2,1],[8,8,-3,0],[-2,2,-2,1],[4,3,-2,5]])
b = np.array([[-2],[0],[2],[-1]])

lu = la.lu(A)

P = lu[0]
L = lu[1]
U = lu[2]

# From this LU fact, we have P.TAx = P.Tb, and P.TAx = LUx = P.Tb
# from Ly = b, find y by forward sub

y = Forward_sub(L,np.dot(P.T,b))

# from Ux = y, find x by backward sub

x = Backward_sub(U,y)

np.dot(A,x)

   