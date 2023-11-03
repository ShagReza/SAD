# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:32:50 2019

@author: Administrator
"""

from glob import glob
import librosa
import config as c

wave_path1 = 'K:\SAD98\SadData2\Validation\\temp' 
wav_list = glob(wave_path1 + '\*.wav')

for w in wav_list: 
    wav, orig_sr = librosa.load(w, c.sample_rate) ## Audio will be automatically resampled to the given rate
    wav        = preprocess_wav(path, c.sample_rate, c.nrz_max, c.nrz_mean) 