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
import utils as u

###==========================================
x_test = np.load('./Results_Stage4/x_SNM_SadFeats_100260270_test_noMVN.npy')
y_test = np.load('./Results_Stage4/y_SNM_SadFeats_100260270_test_noMVN.npy')

#x_test = np.load('./Results_Stage4/x_SM_1115_musan_SadFeats.npy')
#y_test = np.load('./Results_Stage4/y_SM_1115_musan_SadFeats.npy')

#x_test = np.load('./Results_Stage4/x_SNM_100230220_test_chroma12.npy')
#y_test = np.load('./Results_Stage4/y_SNM_100230220_test_chroma12.npy')


y_test = to_categorical(y_test)


#mdl = 'model_epoch_36_trn_0.9526_val_0.9001.hdf5'
mdl = 'model_epoch_41_trn_0.9622_val_0.9410.hdf5'


model = load_model(os.path.join(config.folder_model, mdl))

y_predict = model.predict(x_test)
#acu_value = model.evaluate(x_test, y_test)


id_true = np.argmax(y_test, axis=1)
id_pred = np.argmax(y_predict, axis=1)

## Evaluation Measures
Acu3_all, Acu2_all, PRF3_all, PRF2_all = u.Cal_measures(id_true, id_pred)






## Evaluation Measures
#CM = confusion_matrix(id_true, id_pred)
#acc = accuracy_score(id_true, id_pred)*100 ## hrbk: is equal to sum(diag_elements)/sum(all)
#prc,rec,f1,numb = precision_recall_fscore_support(id_true, id_pred)
#
#PRF = np.append(prc.reshape((3,1)),rec.reshape((3,1)),axis=1)
#PRF = np.append(PRF,f1.reshape((3,1)),axis=1) * 100
#
#Prec = prc*100
#Recal = rec*100
#F1m = f1*100
#Conf_Matrix = CM.reshape((y_test.shape[1],y_test.shape[1]))
#
##ff = f1_score(id_true, id_pred, average=None) ## same as f1 in above code
#
#### accuracy speech/non-speech
#id_true2 = id_true
#id_true2 = np.where(id_true2==2,1,id_true2) ## convert music_labels(==2) to noise_labels(1)
#id_pred2 = id_pred
#id_pred2 = np.where(id_pred2==2,1,id_pred2)
#
#acc_SNS = accuracy_score(id_true2, id_pred2)*100 ## hrbk: is equal to sum(diag_elements)/sum(all)
#prc2,rec2,f12,numb2 = precision_recall_fscore_support(id_true2, id_pred2)
#PRF2 = np.append(prc2.reshape((2,1)),rec2.reshape((2,1)),axis=1)
#PRF2 = np.append(PRF2,f12.reshape((2,1)),axis=1) * 100
