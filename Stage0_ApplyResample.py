# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:20:34 2019

@author: Administrator
"""


import config as c
import numpy as np
import librosa
import os
import scipy.io.wavfile as wav

##======================================================
wave_path1 = r'K:\SAD98\SadData2\Train\MusanNoise_vad'
out_folder = r'K:\SAD98\SadData2\Train\MusanNoise_vad_8k'

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


##======================================================
from glob import glob
wav_list = glob(wave_path1 + '\*.wav')

##======================================================
def resample_wav(wav, orig_sr, target_sr=8000):
    if orig_sr != target_sr:
        wav = librosa.resample(wav, orig_sr, target_sr)
    return wav


for i1,wv in enumerate(wav_list):
    
    print('File Number is {}'.format(i1))
    
    nams = wv.split('\\')
    raw_name = wv.split('\\')[-1]
    
    sig, orig_sr = librosa.load(wv, c.sample_rate) ## Audio will be automatically resampled to the given rate
    
    sig_r = resample_wav(sig, orig_sr, 8000)
    
    sig_r = sig_r / np.max(np.abs(sig_r))
    
    out_file = os.path.join(out_folder, raw_name.replace('.wav', '_8k.wav'))
    
    wav.write(out_file, c.sample_rate, np.int16(sig_r*32767))









