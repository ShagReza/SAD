# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:12:49 2019

@author: Administrator
"""

import numpy as np
import config as c
from shutil import copyfile, move
from glob import glob
import os
from sklearn.metrics import *

##=================================================================
def Cal_measures(id_true, id_pred):

    CM = confusion_matrix(id_true, id_pred)
    acc = accuracy_score(id_true, id_pred)*100 ## hrbk: is equal to sum(diag_elements)/sum(all)
    prc,rec,f1,numb = precision_recall_fscore_support(id_true, id_pred)
    
    PRF = np.append(prc.reshape((3,1)),rec.reshape((3,1)),axis=1)
    PRF = np.append(PRF,f1.reshape((3,1)),axis=1) * 100
    
    id_true2 = id_true
    id_true2 = np.where(id_true2==2,1,id_true2) ## convert music_labels(==2) to noise_labels(1)
    id_pred2 = id_pred
    id_pred2 = np.where(id_pred2==2,1,id_pred2)
    
    acc_SNS = accuracy_score(id_true2, id_pred2)*100 ## hrbk: is equal to sum(diag_elements)/sum(all)
    prc2,rec2,f12,numb2 = precision_recall_fscore_support(id_true2, id_pred2)
    PRF_SNS = np.append(prc2.reshape((2,1)),rec2.reshape((2,1)),axis=1)
    PRF_SNS = np.append(PRF_SNS,f12.reshape((2,1)),axis=1) * 100
    
    return acc, acc_SNS, PRF, PRF_SNS

##==================================================================================

def movefiles(wav_path, feat_path):
   feat_list = glob(wav_path + '\*.npy')

   for ft in feat_list:
       full_wav_name = ft.replace('\\','/')
       raw_name = full_wav_name.split('/')[-1][:-4]+'.npy'
       dst = os.path.join(feat_path, raw_name)
       move(ft, dst)
       
##==================================================================================
       
def hz2mel(hz):
    import numpy as np
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)

##==================================================================================
def get_mel_points():
    
    fft_len = c.n_fft//2+1
    hz_points = np.linspace(0,c.sample_rate//2,fft_len)
    
    lowmel  = hz2mel(0)
    highmel = hz2mel(c.sample_rate//2)
    mel_points = np.linspace(lowmel,highmel,fft_len)
    
    mel_points_hz = mel2hz(mel_points)
    
    return mel_points_hz, hz_points 

##==================================================================================
    
mel_points_hz, hz_points = get_mel_points()




