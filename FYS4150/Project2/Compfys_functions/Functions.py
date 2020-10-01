# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:42:47 2019

@author: tlgreiner
"""

#from scipy.linalg import inv
import numpy as np
from scipy.linalg import lu

########################### matrix inverse solvers #####################

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

def Jacobi_it(A,x,d,eps=10**-5,it_test=100,stop=1000):
    
    b,c = A.shape
    
    err = np.sum(np.abs(A.dot(x)-d))
    
    # Try to vectorize to save computation time
    k = 0
    while err>eps:
        
        for i in range(b):
            B = 0
            for j in range(c):
                if i!=j:
                    B += A[i,j]*x[j]   
            
            x[i] = (d[i] - B)/A[i,i]
            
        # Test error
        if np.mod(k,it_test)==0:
            err = np.sum(np.abs(A.dot(x)-d))
            print(err)
        # Break if to many iterations
        if k>stop:
            break
    return x

def Jacobi_it_vec(A,x,d,eps=10**-5,it_test=100,stop=1000):
    
    b,c = A.shape
    
    D = np.multiply(np.eye(b,c),np.diag(A).reshape(c,1))
    U = np.triu(A,k=1)
    L = np.tril(A,k=-1)
    
    err = np.sum(np.abs(A.dot(x)-d))
    # Try to vectorize to save computation time
    k = 0
    while err>eps:
        
        # use pinv for non-square matrix
        x = np.linalg.pinv(D).dot((d - (L + U).dot(x)))
            
        # Test error
        if np.mod(k,it_test)==0:
            err = np.sum(np.abs(A.dot(x)-d))
            print(err)
        # Break if to many iterations
        if k>stop:
            break
    return x

############################ jacobi rotation #########################################

def find_max(A,aij = 10**-5):
    shape = A.shape
    p,q=0,0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i!=j and np.abs(A[i,j])>=aij:
                aij = np.abs(A[i,j])
                p = i
                q = j
    
    return aij,p,q


def jacobi(A,V,n=10,conv=10**-8):
    aip=0;aiq=0;vpi=0;vqi=0
    tau=0;t=0;s=0;c=0
    p = n-1;q=n-1
    
    apq = A[p,q]
    
    while np.abs(apq)>conv:
        apq = 0
        apq,q,p = find_max(A,aij = apq)
        aqq = A[q,q]
        app = A[p,p]
        tau = (aqq - app)/(2*apq)
      
        if tau>=0.0:
            t = 1/(tau + np.sqrt(1.0 + tau*tau))
        else:
            t = -1/(-tau + np.sqrt(1.0 + tau*tau))
            
        c = 1.0/np.sqrt(1+t*t)
        s = c*t
        
        if apq==0:
            s = 0.0
            c = 1.0
        
        for i in range(n):
            if i!=p and i!=q:
                aip = A[i,p]
                aiq = A[i,q]
                
                A[i,p] = aip*c-aiq*s
                A[p,i] = aip*c-aiq*s
                A[i,q] = aiq*c+aip*s
                A[q,i] = aiq*c+aip*s
                
            
            vpi = V[i,p]
            vqi = V[i,q]
            
            V[i,p] = c*vpi-s*vqi
            V[i,q] = c*vqi+s*vpi
            
        A[p,p]=app*c*c-2*apq*c*s+aqq*s*s
        A[q,q]=app*s*s+2*apq*c*s+aqq*c*c
        A[p,q]=0
        A[q,p]=0
    
    return A,V   

########################### bisection method for eigenvalues #########################
    
def bisect_eig(A,n_pol=4):
    # Matrix A = descending diagonal
    
    # first eigenvalue
    P1   = A[0,0]
    lmbd = []
    lmbd.append(P1)

    Pc = np.zeros(n_pol)
    Pz = np.zeros(n_pol)

    Pc[0] = P1
    Pz[0] = P1

    k  = 1

    while k<n_pol:
        
    
        if k<n_pol-1:
            y = A[k,k] - np.abs(A[k,k+1]) - np.abs(A[k+1,k])
            z = A[k,k] + np.abs(A[k,k+1]) + np.abs(A[k+1,k])
        else:
            y = A[k,k] - np.abs(A[k,k-1]) - np.abs(A[k-1,k])
            z = A[k,k] + np.abs(A[k,k-1]) + np.abs(A[k-1,k])
    
        # first poly
        c = (z+y)/2
    
        Pc[0]   = (A[0,0] - c)
        Pz[0]   = (A[0,0] - z)
        for i in range(1,k+1):
        
            if k<n_pol-1:
                bii = A[i,i+1]
                bij = A[i+1,i]
            else: 
                bii = 1
                bij = 1
            if i==1:
                Pc[i]   = Pc[i-1]*(A[i,i] - c) - bii*bij
                Pz[i]   = Pz[i-1]*(A[i,i] - z) - bii*bij
            else:
                Pc[i]   = Pc[i-1]*(A[i,i] - c) - Pc[i-2]
                Pz[i]   = Pz[i-1]*(A[i,i] - z) - Pz[i-2]  
            
        diff = z-y
    
        while np.abs(diff)>=10**-8:
            if Pc[k]*Pz[k]<0:
                y = c
                z = z
                c = (z+y)/2
                Pc[0]   = (A[0,0] - c)
                Pz[0]   = (A[0,0] - z)
                for i in range(1,k+1):
        
                    if k<n_pol-1:
                        bii = A[i,i+1]
                        bij = A[i+1,i]
                    else: 
                        bii = 1
                        bij = 1
                    if i==1:
                        Pc[i]   = Pc[i-1]*(A[i,i] - c) - bii*bij
                        Pz[i]   = Pz[i-1]*(A[i,i] - z) - bii*bij
                    else:
                        Pc[i]   = Pc[i-1]*(A[i,i] - c) - Pc[i-2]
                        Pz[i]   = Pz[i-1]*(A[i,i] - z) - Pz[i-2]  
            
            else:
                y = y
                z = c
                c = (z+y)/2
                Pc[0]   = (A[0,0] - c)
                Pz[0]   = (A[0,0] - z)
                for i in range(1,k+1):
        
                    if k<n_pol-1:
                        bii = A[i,i+1]
                        bij = A[i+1,i]
                    else: 
                        bii = 1
                        bij = 1
                    if i==1:
                        Pc[i]   = Pc[i-1]*(A[i,i] - c) - bii*bij
                        Pz[i]   = Pz[i-1]*(A[i,i] - z) - bii*bij
                    else:
                        Pc[i]   = Pc[i-1]*(A[i,i] - c) - Pc[i-2]
                        Pz[i]   = Pz[i-1]*(A[i,i] - z) - Pz[i-2]                  
            diff = z-y
        k += 1
     
        lmbd.append(c)
        
    return np.array(lmbd)



#################################### Lanczos' tridiagonalization#############################

def lanczos(A,random_init='True'):
    
    # initialize
    n     = A.shape[0]
    alpha = np.zeros(n)
    beta  = np.zeros(n) 
    Q     = np.zeros([n,n])
    
    if random_init=='True':   
        q      = np.random.randn(n).reshape(n,1)
        q      = q/np.linalg.norm(q)
        Q[:,0] = q[:,0]    
    else:
        q      = np.ones([n,1])/np.sqrt(n)
        Q[:,0] = q[:,0]      
    
    w        = A.dot(q)
    alpha[0] = w.T.dot(q)
    w        = w - alpha[0]*q
    
    for i in range(1,n):
        beta[i] = np.linalg.norm(w)
        if beta[i]!=0:
            q = w/beta[i]
        
        w = A.dot(q)
        alpha[i] = w.T.dot(q)
        w      = w - alpha[i]*q - beta[i]*Q[:,i-1].reshape(n,1)
        Q[:,i] = q[:,0]
        
           
    T = Q.T.dot(A).dot(Q)
    
    return T



########################## sorting matrices and eigenvalues/vectors####################
    

def sort_eig(eig,eigv):
    eig = np.diagonal(eig)
    idx = eig.argsort()
    eig = eig[idx]
    eig = eig[::-1]
    
    eigv = eigv[:,idx]
    eigv = eigv[:,::-1]
    
    return eig,eigv

def sort_eig_val(eig):
    
    idx = eig.argsort()
    eig = eig[idx]
    eig = eig[::-1]
    
    return eig


def sort_mat(T,rotate='True'):
    diag = np.diagonal(T)        
    idx  = diag.argsort()        
    T    = T[idx,:][:,idx]
    if rotate=='True':
        T    = T[:,::-1][::-1,:]
    return T
