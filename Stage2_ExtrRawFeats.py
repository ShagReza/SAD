# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:58:27 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:30:20 2019

@author: Kashani
"""

from multiprocessing import Pool
import os
from datetime import datetime
from glob import glob
import librosa
import config as c
from pathlib import Path
import Speech_Feaures_SAD_21 as SF
import utils as u

####=================================================== 

wave_path1 = 'D:\Shapar\ShaghayeghUni\AfterPropozal\SmalFarsdat' 
wave_path1 = os.path.join(os.path.join(wave_path1, sets), catg)
wav_list = glob(wave_path1 + '\*.wav')


if not os.path.exists(feat_path):
    os.makedirs(feat_path)
if not os.path.exists(feat_path/sets):
    os.makedirs(feat_path/sets)
if not os.path.exists(feat_path/sets/catg_feat):
    os.makedirs(feat_path/sets/catg_feat)
feat_path = feat_path/sets/catg_feat

####====================================================
if __name__ == '__main__': 
   
   print('Feature extraction Starts ...')
   start=datetime.now()
   
#   for i,w in enumerate(wav_list):
#       print('file #{}/{}: {}'.format(i, len(wav_list), w.split('\\')[-1]))        
##       SF.wav2fullmelspec(w)
##       SF.wave2spec(w)
##       SF.wave2melspec(w)
##       SF.wav2chroma(w)
#       SF.wav2drugman(w)
   
   with Pool(processes = 3) as pool:       
     pool.map(SF.wav2sadfeats , wav_list)
#     
   print(datetime.now()-start)
   print('Feature extraction Finished.')
#   
#   ####====================================================
   u.movefiles(wave_path1, feat_path)
#   
   ####====================================================
#   
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       