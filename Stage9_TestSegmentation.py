# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 15:26:42 2020

@author: Administrator
"""

import os
from glob import glob
import Speech_Feaures_SAD_21 as SF
import utils as u
import numpy as np

from keras.utils.np_utils import to_categorical
import config_models as config
from keras.models import load_model

##===========================================

lab_path1 = 'J:\Project_1398_2\SAD98\Codes\DeepSMN4\DataGen\Test_Labs'
wave_path1 = 'J:\Project_1398_2\SAD98\Codes\DeepSMN4\DataGen\Test_Wavs' 

feat_path = 'J:\Project_1398_2\SAD98\Codes\DeepSMN4\Results_Stage2\Test\\Real_Noise_log'
Feat_Cnx_path = 'J:\Project_1398_2\SAD98\Codes\DeepSMN4\Results_Stage3\Test\\Real_Noise_log'

#mdl = 'model_epoch_28_trn_1.0000_val_0.9592.hdf5'
#model = load_model(os.path.join(config.folder_model, mdl))


##===========================================
if not os.path.exists(feat_path):
    os.makedirs(feat_path)

if not os.path.exists(Feat_Cnx_path):
    os.makedirs(Feat_Cnx_path) 

##===========================================
lab_list = glob(lab_path1 + '\*.txt')   
    
    
##===========================================    
#wav_list = glob(wave_path1 + '\*.wav')
#
#for i,w in enumerate(wav_list):
#    print('file {}/{}: {}'.format(i, len(wav_list), w.split('\\')[-1]))
#    SF.wav2sadfeats(w)
##    SF.wave2melspec(w)
#    
#u.movefiles(wave_path1, feat_path)    

##===========================================
feat_list = glob(feat_path + '\*.npy')

for i,ft in enumerate(feat_list):
    print('file {}/{}: {}'.format(i, len(feat_list), ft.split('\\')[-1]))
    Feat_Cnx = SF.provide_context_feats_test(ft)    
    cnx_path = os.path.join(Feat_Cnx_path, 'Cnx_'+ft.split('\\')[-1])    
    np.save(cnx_path, Feat_Cnx)
    
    
##===========================================
#feat_cnx_list = glob(Feat_Cnx_path + '\*.npy') 
#
#for i,ft in enumerate(feat_cnx_list):
#    print('file {}/{}: {}'.format(i, len(feat_cnx_list), ft.split('\\')[-1]))
#    X = np.load(ft)
##    X1 = X[1:5,...]
#    
#    y_predict = model.predict([X])
#    del X














   
    
