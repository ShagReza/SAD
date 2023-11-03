# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:49:07 2020

@author: Administrator
"""

import python_speech_features as psf
import numpy as np
import librosa
import os
import scipy.io.wavfile as wav

fs = 8000
path = 'J:\\Project_1398_2\\SAD98\\Codes\\DeepSMN4\\fa_4054_ch1_vad_noisy_5dB.wav'
sig, orig_sr = librosa.load(path, fs)

frms = psf.sigproc.framesig(sig, .2*fs, .01*fs)
r = frms.std(axis=-1, keepdims=True)
frms = frms/r
sig1 = psf.sigproc.deframesig(frms, siglen=len(sig), 
                              frame_len=.2*fs, frame_step= .01*fs)

sig1 = sig1 / max(abs(sig1))

out_file = 'J:\\Project_1398_2\\SAD98\\Codes\\DeepSMN4\\fa_4054_ch1_vad_noisy_5dB_eq.wav'
wav.write(out_file, fs, np.int16(sig1*32767))
