# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 10:02:50 2020

@author: user
"""


#--------------------------------------------------------------
from multiprocessing import Pool
import os
from datetime import datetime
from glob import glob
import librosa
from pathlib import Path
import Speech_Feaures_SAD_21 as SF
import utils as u
import config as c
import numpy as np
import scipy.io
#--------------------------------------------------------------



wave_path = 'D:\Shapar\ShaghayeghUni\AfterPropozal\SmalFarsdat' 
FeatPath='D:\Shapar\ShaghayeghUni\AfterPropozal\Step1-EventLandmark\Programs\MyPrograms\EventExtraction\Keras\FeatExtract\\MelSpec64'
wav_list = glob(wave_path + '\*.wav')

#Feat=SF.wav2fullmelspec(wav_list[1])
#s = np.load(wave_path1+'\S110.WAV.npy')

for i,w in enumerate(wav_list):      
    Feat=SF.wave2melspec(w)
    Name=wav_list[i]
    Name=Name.replace(wave_path,FeatPath)
    Name=Name.replace('.WAV','.mat')
    scipy.io.savemat(Name, {'Feat': Feat})
    
    
    
    
    
    
    
    
##       SF.wave2spec(w)
##       SF.wave2melspec(w)
##       SF.wav2chroma(w)
#       SF.wav2drugman(w)
   
#with Pool(processes = 3) as pool:       
 #    pool.map(SF.wave2melspec , wav_list)