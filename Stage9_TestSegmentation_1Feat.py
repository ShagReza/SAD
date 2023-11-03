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
import math
from keras.utils.np_utils import to_categorical
import config_models as config
from keras.models import load_model
from joblib import dump,load
#from sklearn.metrics import *


##----------------------
def get_id(ind, my_string, ID_true, ID_pred, SNR):
    matching = [i for i,s in enumerate(SNR[ind]) if my_string == s]
    id_tr = []
    id_pr = []
    for j in range(len(matching)):
        id_tr.append(ID_true[matching[j]]) 
        id_pr.append(ID_pred[matching[j]]) 
    
    return id_tr, id_pr

def get_id_2(mus_or_nos, snr, ID_true, ID_pred):
    matching = []
    for i,sn in enumerate(SNR[1]):
        m_or_n = SNR[0][i]
        if (sn == snr) and (m_or_n==mus_or_nos):
            matching.append(i)
    id_tr = []
    id_pr = []
    for j in range(len(matching)):
        id_tr.append(ID_true[matching[j]]) 
        id_pr.append(ID_pred[matching[j]]) 
    
    return id_tr, id_pr    
##===========================================
HalfFilterLen=25

lab_path1 = '.\DataGen\\Test_Labs'
wave_path1 = '.\DataGen\\Test_Wavs' 

feat_path = '.\Results_Stage9\FeatFrames\log'
Feat_Cnx_path = '.\Results_Stage9\FeatContext\log'

mdl = 'model_epoch_28_trn_1.0000_val_0.9592.hdf5'
model = load_model(os.path.join(config.folder_model, mdl))


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
#    print('file {}/{}: {}'.format(i+1, len(wav_list), w.split('\\')[-1]))
#    SF.wav2sadfeats(w)
##    SF.wave2melspec(w)
#    
#u.movefiles(wave_path1, feat_path)    
##
#####===========================================
#feat_list = glob(feat_path + '\*.npy')
#
#for i,ft in enumerate(feat_list):
#    print('file {}/{}: {}'.format(i+1, len(feat_list), ft.split('\\')[-1]))
#    Feat_Cnx = SF.provide_context_feats_test(ft)    
#    cnx_path = os.path.join(Feat_Cnx_path, 'Cnx_'+ft.split('\\')[-1])    
#    np.save(cnx_path, Feat_Cnx)
#       
##=============================================================================
feat_cnx_list = glob(Feat_Cnx_path + '\*.npy') 

ID_true = []
ID_pred = []

SNR = [[],[]]

for i,ft in enumerate(feat_cnx_list):
    nam = ft.split('\\')[-1]    
    snr_value = nam.split('_')[-1][0:-6]
    
    print('file {}/{}: {}'.format(i+1, len(feat_cnx_list), nam))
    X = np.load(ft)
    Len_file=X .shape[0]
    y_predict = model.predict([X])
    #----------------------------------------------------
    # Flter the output
    
    Criteria=math.floor(HalfFilterLen*2/3)
    x=range(Len_file)    
    IndexMaxOut=np.argmax(y_predict,axis=1)
    FilteredOutput=np.zeros(Len_file)
    for i1 in x:
        if i1<HalfFilterLen:
            I1=0
            I2=2*HalfFilterLen+1
        elif i1>=(Len_file-HalfFilterLen):
            I1=Len_file-(2*HalfFilterLen+1)
            I2=Len_file
        else:
            I1=i1-HalfFilterLen
            I2=i1+HalfFilterLen+1
        IndexMaxOut_Segment=IndexMaxOut[I1:I2]     
        SpeechFrames=np.where(IndexMaxOut_Segment==0)[0]
        NumSpeechFrames=SpeechFrames.shape[0]
        NoiseFrames=np.where(IndexMaxOut_Segment==1)[0]
        NumNoiseFrames=NoiseFrames.shape[0]
        MusicFrames=np.where(IndexMaxOut_Segment==2)[0]
        NumMusicFrames=MusicFrames.shape[0]
        if NumSpeechFrames>=Criteria:
           FilteredOutput[i1]=0   
        elif NumNoiseFrames>=Criteria:
           FilteredOutput[i1]=1
        else:
           FilteredOutput[i1]=2
    
    print('DNN output was filtered') 
    #-------------------------------------------------      
    # load Gold labels
    lblpath= lab_list[i]
    GoldLabels=np.zeros(Len_file)
    file = open(lblpath, "r")          
    contents =file.read()
    contents2 = contents.split('\n')
    a=range(len(contents2)-1)
    for i2 in a:
        contents3=contents2[i2].split('\t\t')
        I1=math.floor(float(contents3[0])*100)
        I2=math.floor(float(contents3[1])*100)
        if contents3[2]=='speech':
            classlabel=0
        elif contents3[2]=='noise':
            classlabel=1
        else:   
            classlabel=2 
        GoldLabels[I1:I2]=classlabel
        
    print('GoldLabels was achieved') 
    #-------------------------------------------------
    id_true0 = np.int64(GoldLabels)
    id_pred0 = np.int64(FilteredOutput)
    
    ID_true.append(id_true0)
    ID_pred.append(id_pred0)
    
    SNR[0].append(nam.split('_')[-2])
    SNR[1].append(snr_value)


##=============================================================================
id_true = np.concatenate(ID_true, axis=0)
id_pred = np.concatenate(ID_pred, axis=0)

Acu3_all, Acu2_all, PRF3_all, PRF2_all = u.Cal_measures(id_true, id_pred)

##=========================================
ID_true_m, ID_pred_m = get_id(0, "music", ID_true, ID_pred, SNR)
Acu3_M, Acu2_M, PRF3_M, PRF2_M = u.Cal_measures(np.concatenate(ID_true_m, axis=0), np.concatenate(ID_pred_m, axis=0))

ID_true_n, ID_pred_n = get_id(0, "noisy", ID_true, ID_pred, SNR)
Acu3_N, Acu2_N, PRF3_N, PRF2_N = u.Cal_measures(np.concatenate(ID_true_n, axis=0), np.concatenate(ID_pred_n, axis=0))


##=========================================
Snrs_noise = ['20','15','10','5','0','-5']
ACU_n = np.zeros((2,6))
for i, sn in enumerate(Snrs_noise):
    ID_true_5, ID_pred_5 = get_id_2("noisy", sn, ID_true, ID_pred)
    print(len(ID_true_5))
    Acu3_sn, Acu2_sn, _, _ = u.Cal_measures(np.concatenate(ID_true_5, axis=0), np.concatenate(ID_pred_5, axis=0))
    ACU_n[0][i] = Acu3_sn
    ACU_n[1][i] = Acu2_sn
    del Acu3_sn, Acu2_sn


##=========================================
Snrs_music = ['22.5','20','17.5']
ACU_m = np.zeros((2,3))
for i, sn in enumerate(Snrs_music):
    ID_true_5, ID_pred_5 = get_id_2("music", sn, ID_true, ID_pred)
    print(len(ID_true_5))
    Acu3_sn, Acu2_sn, _, _ = u.Cal_measures(np.concatenate(ID_true_5, axis=0), np.concatenate(ID_pred_5, axis=0))
    ACU_m[0][i] = Acu3_sn
    ACU_m[1][i] = Acu2_sn
    del Acu3_sn, Acu2_sn


##===============
Res = {'ID_true':ID_true, 'ID_pred':ID_pred, 'SNR':SNR}    
   
#dump(Res,'.\Measures\Res_SadFeats.job')

#Res = load('.\Measures\Res_log.job')
#ID_true = Res['ID_true']
#ID_pred = Res['ID_pred']
#SNR = Res['SNR']



