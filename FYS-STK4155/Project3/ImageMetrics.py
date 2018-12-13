# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 07:55:16 2018

@author: tagreine
"""

import numpy as np

def PSNR(target_, pred):
    # Assume channels > 1
    target_data = np.array(target_)

    pred_data = np.array(pred)
    
    diff = target_data - pred_data
    diff = diff.flatten('C')
    rmse = np.sqrt( np.mean(diff ** 2.) )
    return 20*np.log10(1/rmse)
