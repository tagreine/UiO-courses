# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:37:43 2018

@author: tagreine
"""

from scipy.io import loadmat


def load_dat(seismic_data = 'data'):

    training_data = loadmat(seismic_data)

    training_data = training_data['Shots_training_gn']
    
    return training_data
