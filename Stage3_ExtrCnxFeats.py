# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:30:20 2019

@author: Kashani
"""

from multiprocessing import Pool
import os
from datetime import datetime

from pathlib import Path

import Speech_Feaures_SAD_21 as SF
import utils as u
import numpy as np
import config as c

####====================================================
sets = 'Validation'  ## Train Validation Test
#catg = 'Speech2\\-5&-10_SadFeats' ## Speech  Noise_vad  Music_vad Speech2PlusNoise
catg = 'Music_SadFeats' ## Speech  Noise  Music Speech2PlusNoise

feat_name = './Results_Stage3/Val_Music_SadFeats_cx32_hop320_winMVN2sec.npy'
#feat_name = './Results_Stage3/Test_Speech2_n5_n10_SadFeats_cx32_hop80_winMVN.npy'


feat_path = Path(r'.\Results_Stage2')
feat_path = feat_path/sets/catg


if not os.path.exists('./Results_Stage3'):
    os.makedirs('./Results_Stage3') 


####====================================================
   
print('Feature Context extraction Starts ...')
start=datetime.now()
   
#num_frames = SF.numframes_context(feat_path)

if c.mvn_type == 'win':
    print(' MVN on {}sec '.format(c.win_len_mvn))
X = SF.provide_context_feats(feat_path)  
   
print(datetime.now()-start)
print('Feature extraction Finished.')
print('hop is {}'.format(c.hop_frames))


#np.save(feat_name, X)


#   ####====================================================

       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       