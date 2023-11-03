# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:08:25 2019

@author: Administrator
"""

import numpy as np
import config as c

Xn = np.load('J:\Project_1398_2\SAD98\Codes\DeepSMN4\Results_Stage2\Train\Music_log\\00011_vad.npy')

frames = range(c.context+c.offset, Xn.shape[1]-c.context-1, c.hop_frames)
Xn_utt = []
for j in range(len(frames)):
    if c.contex_type == 'odd':
      Xn_utt.append(Xn[0:c.n_fft//2, frames[j]-c.context : frames[j]+c.context+1])
    else:
      Xn_utt.append(Xn[0:c.n_fft//2, frames[j]-c.context : frames[j]+c.context])
del Xn