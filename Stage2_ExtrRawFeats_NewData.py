# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:58:27 2019

@author: Administrator
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

####====================================================
#### GTZAN
#wave_path1 = 'J:\Project_1398_2\SAD98\Data\Speech_Music\GTZAN_Kahrizi\\Speech' 
#feat_path = 'J:\Project_1398_2\SAD98\Codes\DeepSMN4\Results_Stage2\Test\\Speech_GTZAN_SadFeats'

#wave_path1 = 'J:\Project_1398_2\SAD98\Data\Speech_Music\GTZAN_Kahrizi\\nonspeech' 
#feat_path = 'J:\Project_1398_2\SAD98\Codes\DeepSMN4\Results_Stage2\Test\\Nonspeech_GTZAN_SadFeats'

#wave_path1 = 'K:\\SAD98\\CallFriend\\Wave_Mono_ch1_vad' 
#feat_path = 'J:\Project_1398_2\SAD98\Codes\DeepSMN4\Results_Stage2\Test\\CallFriend_SadFeats'

#wave_path1 = 'K:\SAD98\\Music_Noise\\Noise_8k' 
#feat_path = 'J:\Project_1398_2\SAD98\Codes\DeepSMN4\Results_Stage2\Test\\Noise_8k_SadFeats'

#wave_path1 = 'K:\SAD98\\Music_Noise\\Music_Vocal_Nonvocal' 
#feat_path = 'J:\Project_1398_2\SAD98\Codes\DeepSMN4\Results_Stage2\Test\\Music_Vocal_Nonvocal_SadFeats'

#wave_path1 = 'K:\SAD98\\Music_Noise\\NonVocalMusicAddedToSpeech_vad' 
#feat_path = 'J:\Project_1398_2\SAD98\Codes\DeepSMN4\Results_Stage2\Test\\NonVocalMusicAddedToSpeech_SadFeats'

wave_path1 = 'K:\\SAD98\\Musan_Split_vad\\Test\\noise_8k' 
feat_path = 'J:\Project_1398_2\SAD98\Codes\DeepSMN4\Results_Stage2\Test\\Musan_Test_noise8k_SadFeats'



wav_list = glob(wave_path1 + '\*.wav')

if not os.path.exists(feat_path):
    os.makedirs(feat_path)


####====================================================
if __name__ == '__main__': 
   
   print('Feature extraction Starts ...')
   start=datetime.now()
   
#   for i,w in enumerate(wav_list):
#       print('file #{}/{}: {}'.format(i+1, len(wav_list), w.split('\\')[-1]))
#       SF.wav2sadfeats(w)
#       SF.wave2melspec(w)
    
   
   with Pool(processes = 4) as pool:
#     pool.map(SF.wave2melspec , wav_list)
#     pool.map(SF.wav2chroma, wav_list)
     pool.map(SF.wav2sadfeats, wav_list)
#     
   print(datetime.now()-start)
   print('Feature extraction Finished.')
#   
#   ####====================================================
   u.movefiles(wave_path1, feat_path)
#   
   ####====================================================
#   
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       