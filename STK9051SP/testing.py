# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:00:44 2021

@author: tlgreiner
"""



import numpy as np

a = 1/2
b = 1/np.sqrt(2)

phiT = np.array([[a,a,a,a],[a,a,-a,-a],[b,-b,0,0],[0,0,b,-b]])

F    = phiT.T.dot(phiT)

# dual ( (phi.dot(phiT) )^-1 phiT)

phiT_trunc = phiT[:2,:]

H   = phiT_trunc.T.dot(phiT_trunc)

F_h = np.linalg.pinv(H).dot(phiT_trunc.T).dot(phiT_trunc)  

phiT = np.array([[b,b],[b,-b]])

F    = phiT.T.dot(phiT)
