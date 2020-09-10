# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:42:47 2019

@author: tlgreiner
"""

#from scipy.linalg import inv
import numpy as np
from scipy.linalg import lu
'''
#########simple test cases##################

a = np.array([[1.0,1.0,1.0]]).T
b = np.array([[2.0,3.0,4.0,3.0]]).T
c = np.array([[1.0,1.0,1.0]]).T
d = np.array([[4.0,10.0,18.0,15.0]]).T

x = Thomas_algorithm_solution(a,b,c,d)

A = np.array([[2,1,0,0],[1,3,1,0],[0,1,4,1],[0,0,1,3]])


np.dot( inv(A), d )
#########simple test cases##################
n = 5
d = 2*np.ones([n])
e = -1*np.ones([n])
g = np.arange(5)

u = gaus_elim(d,e,g)
u = special_case(g)
  
A = np.array([[2,-1,0,0,0],[-1,2,-1,0,0],[0,-1,2,-1,0],[0,0,-1,2,-1],[0,0,0,-1,2]])

A.dot(u)  
'''

def Thomas_algorithm_solution(a,b,c,d):
    # Thomas algorithm for tridiagonal matrix inversion  
    # a, b, c and d need to be column vectors
    # a = lower off-diagonal elements
    # b = diagonal elements
    # c = upper off-diagonal elements
    # d = forcing function
    
    # Correct for BC
    #a = a[1:-1]
    #b = b[1:-1]
    #c = c[1:-1]    
    #d = d[1:-1]
    #a = np.c_[0,a.T].T
    #c = np.c_[c.T,0].T
    
    n = len(d)
    
    # Forward substitution with initial conditions
    c_      = np.zeros([n,1])
    d_      = np.zeros([n,1])
    c_[0,0] = c[0,0]/b[0,0]    
    d_[0,0] = d[0,0]/b[0,0]  
    
    for i in range(1,n-1):
        
        c_[i,0] = c[i,0]/( b[i,0] - a[i,0]*c_[i-1,0])           # 3(n-1) FLOP'S
    
    for i in range(1,n):    
        d_[i] = ( d[i] - a[i]*d_[i-1] )/( b[i] - a[i]*c_[i-1] ) # 5(n-1) FLOP'S 
    
    
    # Backward substitution with initial conditions to get the solution
    
    x      = np.zeros([n,1])
    x[n-1] = d_[n-1]
    for i in range(n-2,-1,-1):
        x[i] = d_[i] - c_[i]*x[i+1] # 2(n-1) FLOP's
        
    # Correct for BC
    #x0 = np.zeros([1,1]) 
    #x1 = np.zeros([1,1])
    #x  = np.concatenate([x0,x,x1])
    
    return x
  

def gaus_elim(d,e,g):
    # Gaus elimination for tridiagonal matrix inversion  
    # e = lower off-diagonal elements and upper off-diagonal elements
    # d = diagonal elements
    # g = forcing function

    n = len(g)
    
    d_ = np.zeros([n])
    g_ = np.zeros([n])
    u  = np.zeros([n])   

    d_[0] = d[0]
    g_[0] = g[0]

    for i in range(1,n):
        d_[i] = d[i] - (e[i-1]**2)/d_[i-1]      # 3(n-1) FLOP'S
        g_[i] = g[i] - (e[i-1]*g_[i-1])/d_[i-1] # 3(n-1) FLOP'S

    u[-1] = g_[-1]/d_[-1]                       
    for i in range(n-2,-1,-1):
        u[i] = (g_[i] - e[i]*u[i+1])/d_[i]      # 3(n-1) FLOPS
    
    return u

def special_case(g):

    # Simplified elimination for tridiagonal matrix inversion  
    # g = forcing function
    
    n = len(g)
    
    d_ = np.zeros([n])
    g_ = np.zeros([n])
    u  = np.zeros([n])   

    d_[0] = 2
    g_[0] = g[0]

    for i in range(1,n):
        d_[i] = 2 - 1/d_[i-1]          # 2(n-1) FLOP'S
        g_[i] = g[i] + g_[i-1]/d_[i-1] # 2(n-1) FLOP'S

    u[-1] = g_[-1]/d_[-1]              
    for i in range(n-2,-1,-1):
        u[i] = (g_[i] + u[i+1])/d_[i]  # 2(n-1) FLOPS
    
    return u

def LU_decomp_inverse(A,d):
    
    # LU decomposition of matrix A
    P,L,U = lu(A)
    
    # Solve for Lw = d
    w = np.linalg.solve(L,d)
    
    # Solve for Ux = w
    v = np.linalg.solve(U,w)
    
    return v

