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
s1 = np.load('./Results_Stage3/Train_Speech_SadFeats_cx32_hop160_noMVN.npy')
s2 = np.load('./Results_Stage3/Train_Speech2PlusNoise_SadFeats_cx32_hop53_noMVN.npy')
s = np.concatenate((s1,s2), axis=0)

n = np.load('./Results_Stage3/Train_Noise_SadFeats_cx32_hop500_noMVN.npy')
s_shap = s.shape
n_shap = n.shape
sn = np.concatenate((s,n), axis=0)
del s, n
m = np.load('./Results_Stage3/Train_Music_SadFeats_cx32_hop370_noMVN.npy')
m_shap = m.shape
x = np.concatenate((sn,m), axis=0)
del m
y = np.concatenate((np.zeros(s_shap[0]),np.ones(n_shap[0]),np.ones(m_shap[0])+1))
y = np.float32(y)

np.save('./Results_Stage4/x_SnNM_SadFeats_16053500370_noMVN.npy',x)
np.save('./Results_Stage4/y_SnNM_SadFeats_16053500370_noMVN.npy',y)


##================================================
s1 = np.load('./Results_Stage3/Val_Speech_SadFeats_cx32_hop150_noMVN.npy')
s2 = np.load('./Results_Stage3/Val_Speech2PlusNoise_SadFeats_cx32_hop50_noMVN.npy')
s = np.concatenate((s1,s2), axis=0)
del s1, s2

n = np.load('./Results_Stage3/Val_Noise_SadFeats_cx32_hop160_noMVN.npy')
s_shap = s.shape
n_shap = n.shape
sn = np.concatenate((s,n), axis=0)
del s, n
m = np.load('./Results_Stage3/Val_Music_SadFeats_cx32_hop320_noMVN.npy')
m_shap = m.shape
x = np.concatenate((sn,m), axis=0)
del m
y = np.concatenate((np.zeros(s_shap[0]),np.ones(n_shap[0]),np.ones(m_shap[0])+1))
y = np.float32(y)

np.save('./Results_Stage4/x_SnNM_SadFeats_15050160320_val_noMVN.npy',x)
np.save('./Results_Stage4/y_SnNM_SadFeats_15050160320_val_noMVN.npy',y)

##================================================
s = np.load('./Results_Stage3/Test_Speech_SadFeats_cx32_hop120_noMVN.npy')
n = np.load('./Results_Stage3/Test_Noise_SadFeats_cx32_hop260_noMVN.npy')
s_shap = s.shape
n_shap = n.shape
sn = np.concatenate((s,n), axis=0)
del s, n
m = np.load('./Results_Stage3/Test_Music_SadFeats_cx32_hop270_noMVN.npy')
m_shap = m.shape
x = np.concatenate((sn,m), axis=0)
del m
y = np.concatenate((np.zeros(s_shap[0]),np.ones(n_shap[0]),np.ones(m_shap[0])+1))
y = np.float32(y)

np.save('./Results_Stage4/x_SNM_SadFeats_100260270_test_noMVN.npy',x)
np.save('./Results_Stage4/y_SNM_SadFeats_100260270_test_noMVN.npy',y)
