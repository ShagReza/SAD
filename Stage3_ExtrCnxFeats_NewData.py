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
#feat_path = Path(r'.\Results_Stage2\Test\Speech_GTZAN_SadFeats')
#feat_name = './Results_Stage3/Test_Speech_GTZAN_SadFeats_cx32_hop11.npy'

#feat_path = Path(r'.\Results_Stage2\Test\Nonspeech_GTZAN_SadFeats')
#feat_name = './Results_Stage3/Test_Nonspeech_GTZAN_SadFeats_cx32_hop15.npy'

#feat_path = Path(r'.\Results_Stage2\Test\CallFriend_SadFeats')
#feat_name = './Results_Stage3/Test_CallFriend_SadFeats_cx32_hop72_winMVN.npy'

#feat_path = Path(r'.\Results_Stage2\Test\Noise_8k_SadFeats')
#feat_name = './Results_Stage3/Test_Noise_8k_SadFeats_cx32_hop6_winMVN.npy'


#feat_path = Path(r'.\Results_Stage2\Test\Musan_Test_noise8k_SadFeats')
#feat_name = './Results_Stage3/Test_Musan_Test_noise8k_SadFeats_cx32_hop5_winMVN.npy'

#feat_path = Path(r'.\Results_Stage2\Test\Music_Vocal_Nonvocal_SadFeats')
#feat_name = './Results_Stage3/Test_Music_Vocal_Nonvocal_SadFeats_cx32_hop36_noMVN.npy'

#feat_path = Path(r'.\Results_Stage2\Test\NonVocalMusicAddedToSpeech_SadFeats')
#feat_name = './Results_Stage3/Test_NonVocalMusicAddedToSpeech_SadFeats_cx32_hop5_winMVN.npy'

feat_path = Path(r'.\Results_Stage2\Test\Noise_SadFeats')
feat_name = './Results_Stage3/Test_Noise_SadFeats_cx32_hop40_winMVN.npy'


if not os.path.exists('./Results_Stage3'):
    os.makedirs('./Results_Stage3') 


####====================================================
   
print('Feature Context extraction Starts ...')
start=datetime.now()
   
#   num_frames = SF.numframes_context(feat_path)
X = SF.provide_context_feats(feat_path)  
   
#   np.save(feat_name, X)
   
print(datetime.now()-start)
print('Feature extraction Finished.')
print('hop is {}'.format(c.hop_frames))
#   ####====================================================

       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       