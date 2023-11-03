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
s = np.load('./Results_Stage3/Train_Speech_Drug_cx32_hop160.npy')
n = np.load('./Results_Stage3/Train_Noise_Drug_cx32_hop500.npy')
s_shap = s.shape
n_shap = n.shape
sn = np.concatenate((s,n), axis=0)
del s, n
m = np.load('./Results_Stage3/Train_Music_Drug_cx32_hop370.npy')
m_shap = m.shape
x = np.concatenate((sn,m), axis=0)
del m
y = np.concatenate((np.zeros(s_shap[0]),np.ones(n_shap[0]),np.ones(m_shap[0])+1))
y = np.float32(y)

np.save('./Results_Stage4/x_SNM_160540375_chroma12.npy',x)
np.save('./Results_Stage4/y_SNM_160540375_chroma12.npy',y)


##================================================
s = np.load('./Results_Stage3/Val_Speech_chroma12_cx32_hop160.npy')
n = np.load('./Results_Stage3/Val_Noise_chroma12_cx32_hop250.npy')
s_shap = s.shape
n_shap = n.shape
sn = np.concatenate((s,n), axis=0)
del s, n
m = np.load('./Results_Stage3/Val_Music_chroma12_cx32_hop375.npy')
m_shap = m.shape
x = np.concatenate((sn,m), axis=0)
del m
y = np.concatenate((np.zeros(s_shap[0]),np.ones(n_shap[0]),np.ones(m_shap[0])+1))
y = np.float32(y)

np.save('./Results_Stage4/x_SNM_160250375_val_chroma12.npy',x)
np.save('./Results_Stage4/y_SNM_160250375_val_chroma12.npy',y)

##================================================
s = np.load('./Results_Stage3/Test_Speech_chroma12_cx32_hop100.npy')
n = np.load('./Results_Stage3/Test_Noise_chroma12_cx32_hop230.npy')
s_shap = s.shape
n_shap = n.shape
sn = np.concatenate((s,n), axis=0)
del s, n
m = np.load('./Results_Stage3/Test_Music_chroma12_cx32_hop220.npy')
m_shap = m.shape
x = np.concatenate((sn,m), axis=0)
del m
y = np.concatenate((np.zeros(s_shap[0]),np.ones(n_shap[0]),np.ones(m_shap[0])+1))
y = np.float32(y)

np.save('./Results_Stage4/x_SNM_100230220_test_chroma12.npy',x)
np.save('./Results_Stage4/y_SNM_100230220_test_chroma12.npy',y)
