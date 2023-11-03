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
wave_path1 = 'K:\\SAD98\\Music_Noise\\NonVocalMusicAddedToSpeech'
out_folder = 'K:\\SAD98\\Music_Noise\\NonVocalMusicAddedToSpeech_vad'

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


##======================================================
from glob import glob
wav_list = glob(wave_path1 + '\*.wav')

##======================================================
def VAD_librosa(path):
    """written by hrbk 98/08/13
    """
    wav, _ = librosa.load(path, c.sample_rate) ## Audio will be automatically resampled to the given rate
    
    if c.nrz_mean:
        wav -= np.mean(wav)
        
    vad = librosa.effects.split(wav, top_db=c.top_db_vad, frame_length=int(c.win_length), 
                        hop_length=int(c.hop_length))
    
    isig = []
    for v in vad:
        isig.append(np.arange(*v))
    isig = np.concatenate(isig, axis=0)
    isil = np.setdiff1d(np.arange(len(wav)), isig)    
    
    return wav, isig, isil


for i1,wv in enumerate(wav_list):
    
    print('File Number is {}'.format(i1))
    
    sig, isig, isil = VAD_librosa(wv)
    
    nams = wv.split('\\')
    raw_name = wv.split('\\')[-1]
    
    out_file = os.path.join(out_folder, raw_name.replace('.wav', '_vad.wav'))
    output = sig[isig]
    
    wav.write(out_file, c.sample_rate, np.int16(output*32767))









