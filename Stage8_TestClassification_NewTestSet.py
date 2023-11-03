# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:20:00 2019

@author: Administrator
"""

##==========================================
import numpy as np
import os
from sklearn.metrics import *

from keras.utils.np_utils import to_categorical
import config_models as config
from keras.models import load_model

##==========================================
### speech
#x_test = np.load('./Results_Stage3/Test_Speech2_25_20_SadFeats_cx32_hop80_noMVN.npy')
#y_test = np.zeros(x_test.shape[0])

#x_test = np.load('./Results_Stage3/Test_CallFriend_SadFeats_cx32_hop72_noMVN.npy')
#y_test = np.zeros(x_test.shape[0])

#x_test = np.load('./Results_Stage3/Test_Noise_8k_SadFeats_cx32_hop6_noMVN.npy')
#y_test = np.zeros(x_test.shape[0])+1

#x_test = np.load('./Results_Stage3/Test_Musan_Test_noise8k_SadFeats_cx32_hop5_noMVN.npy')
#y_test = np.zeros(x_test.shape[0])+1

x_test = np.load('./Results_Stage3/Test_Noise_SadFeats_cx32_hop40_noMVN.npy')
y_test = np.zeros(x_test.shape[0])+1

#x_test = np.load('./Results_Stage3/Test_Music_Vocal_Nonvocal_SadFeats_cx32_hop36_noMVN.npy')
#y_test = np.zeros(x_test.shape[0])+2

#x_test = np.load('./Results_Stage3/Test_NonVocalMusicAddedToSpeech_SadFeats_cx32_hop5_noMVN.npy')
#y_test = np.zeros(x_test.shape[0])+2


y_test = to_categorical(y_test, num_classes=3)


#mdl = 'model_epoch_33_trn_0.9593_val_0.9395.hdf5'
mdl = 'model_epoch_41_trn_0.9622_val_0.9410.hdf5'


model = load_model(os.path.join(config.folder_model, mdl))

#y_predict = model.predict(x_test)
y_predict = model.predict([x_test])

#acu_value = model.evaluate(x_test, y_test)

id_true = np.argmax(y_test, axis=1)
id_pred = np.argmax(y_predict, axis=1)

#id_true2 = id_true
#id_true2 = np.where(id_true2==2,1,id_true2) ## convert music_labels(==2) to noise_labels(1)
#id_pred2 = id_pred
#id_pred2 = np.where(id_pred2==2,1,id_pred2)

## Evaluation Measures
#CM = confusion_matrix(id_true, id_pred)
acc = accuracy_score(id_true, id_pred)*100 ## hrbk: is equal to sum(diag_elements)/sum(all)
#prc,rec,f1,numb = precision_recall_fscore_support(id_true, id_pred)
#
#
#Prec = prc*100
#Recal = rec*100
#F1m = f1*100
#Conf_Matrix = CM.reshape((3,3))
#
##ff = f1_score(id_true, id_pred, average=None) ## same as f1 in above code
#
#### accuracy speech/non-speech
#acc_SNS = accuracy_score(id_true2, id_pred2) ## hrbk: is equal to sum(diag_elements)/sum(all)
#prc2,rec2,f12,numb2 = precision_recall_fscore_support(id_true2, id_pred2)
