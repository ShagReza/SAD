# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:01:11 2019

@author: Administrator
"""

import numpy as np
import os

if not os.path.exists('./Results_Stage4'):
    os.mkdir('./Results_Stage4')
    
    
##================================================
s = np.load('./Results_Stage3/Test_Speech_GTZAN_SadFeats_cx32_hop11.npy')
m = np.load('./Results_Stage3/Test_Nonspeech_GTZAN_SadFeats_cx32_hop15.npy')
s_shap = s.shape
m_shap = m.shape
x = np.concatenate((s,m), axis=0)

y = np.concatenate((np.zeros(s_shap[0]), np.ones(m_shap[0])+1))
y = np.float32(y)

np.save('./Results_Stage4/x_SM_1115_musan_SadFeats.npy',x)
np.save('./Results_Stage4/y_SM_1115_musan_SadFeats.npy',y)


##================================================
