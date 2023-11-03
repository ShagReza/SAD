# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:30:20 2019

@author: Administrator
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
sets = 'Train'  ## Train Validation Test
#catg = 'Speech2\\25&20_SadFeats' ## Speech  Noise_vad  Music_vad Speech2PlusNoise
catg = 'Speech_log' ## Speech  Noise  Music Speech2PlusNoise
#catg = 'Temp' ## Speech  Noise  Music Speech2PlusNoise

feat_name = './Results_Stage3/Train_Speech_log_cx32_hop160_winMVN2sec.npy'
#feat_name = './Results_Stage3/Test_Speech2_25_20_SadFeats_cx32_hop80_noMVN.npy'


feat_path = Path(r'.\Results_Stage2')
feat_path = feat_path/sets/catg


if not os.path.exists('./Results_Stage3'):
    os.makedirs('./Results_Stage3') 


####====================================================
if __name__ == '__main__':  
    
    #num_frames = SF.numframes_context(feat_path)
    
    print('Feature Context extraction Starts ...')
    if c.mvn_type == 'win':
        print(' MVN on {}sec '.format(c.win_len_mvn))
    
    start=datetime.now()
    
    feat_list = list(feat_path.glob('**/*.npy'))     
    with Pool(processes = np.min([len(feat_list), 3])) as pool:       
        X = pool.map(SF.provide_context_feats_par , feat_list)
    X = np.concatenate(X, axis=0)
    X = np.float32(X)
    
    print(datetime.now()-start)
    print('Feature extraction Finished.')
    print('hop is {}'.format(c.hop_frames))

#np.save(feat_name, X)

#   ####====================================================

       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       