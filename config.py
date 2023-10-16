# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:59:28 2019

@author: user
"""
from scipy import signal
import librosa  
import numpy as np   

task = 'class'    ## 'class' or 'enhan'
top_db_vad = 40

sample_rate = 22050         #Sampling Frequency
n_fft       = 705
win_length  = n_fft
hop_length  = 256
center_lib = False
window = signal.hamming
amp         = 'log'  ## 'lin' 'log' 'pow'
min_mag = 10**(-6)
n_mels = 64

n_mfcc=18 

nrz_max  = True   ## waveform normalization to max value
nrz_mean = False    ## waveform normalization to mean value

win_len_mvn = 2 ## 3sec

cnx_mvn = np.int(win_len_mvn / (hop_length/sample_rate))

####==========
apply_eq = 0
eq_win = 0.250
eq_shift = 0.010

####==========
context = 32
offset = 5

hop_frames = 160
mvn_type = 'win'  ## 'utt' 'win' 'no' 
  
contex_type = 'even'   ## 'even' or 'odd'

n_chroma = 12
nrm_chroma = None   ### np.inf   or None

##===== test
context_test = 32
hop_frames_test = 1

####====================================================
def mel_filter(sample_rate, n_fft, n_mels):
    mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels)
    return mel_basis

mel_basis = mel_filter(sample_rate, n_fft, n_mels)
