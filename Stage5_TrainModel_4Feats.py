# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:14:35 2019

@author: Administrator
"""

import numpy as np
from keras.utils.np_utils import to_categorical
from keras.optimizers import *
from keras import callbacks
from keras.callbacks import TensorBoard, CSVLogger

#from keras.utils.io_utils import HDF5Matrix
import config_models as config

import os

##
if not os.path.exists(config.folder_model):
    os.mkdir(config.folder_model)
    
if not os.path.exists(config.folder_log):   os.makedirs(config.folder_log)

##==========================
x_train1 = np.load('./Results_Stage4/x_SnNM_log_16053500370.npy')
x_train2 = np.load('./Results_Stage4/x_SnNM_normChroma_16053500370.npy')
x_train3 = np.load('./Results_Stage4/x_SnNM_SadFeats_16053500370.npy')
x_train4 = np.load('./Results_Stage4/x_SnNM_Drug_16053500370.npy')
x_train4 = x_train4[:,:,:,0]

y_train  = np.load('./Results_Stage4/y_SnNM_log_16053500370.npy')
y_train  = to_categorical(y_train)

x_val1 = np.load('./Results_Stage4/x_SnNM_log_15050160320_val.npy')
x_val2 = np.load('./Results_Stage4/x_SnNM_normChroma_15050160320_val.npy')
x_val3 = np.load('./Results_Stage4/x_SnNM_SadFeats_15050160320_val.npy')
x_val4 = np.load('./Results_Stage4/x_SnNM_Drug_15050160320_val.npy')
x_val4 = x_val4[:,:,:,0]

y_val  = np.load('./Results_Stage4/y_SnNM_log_15050160320_val.npy')
y_val  = to_categorical(y_val)

##==========================================
import config_models as config

#from Models.ParallelCnn3_gap import create_model
#model = create_model((64,64,1), (12,64,1), (12,64,1), config)

from Models.ParallelCnn4_gap import create_model
model = create_model((64,64,1), (12,64,1), (12,64,1), (7,64), config)

model.summary()

opt1 = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt1, loss='categorical_crossentropy', metrics= ['accuracy'])

##==========================================
reduce_LR = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=7, verbose=1, mode='max', epsilon=0.0001, cooldown=0, min_lr=0)

tensorboard1 = TensorBoard(log_dir = config.folder_log)
logger = CSVLogger(config.folder_log + '/training.log')
path_model = os.path.join(config.folder_model, 'model_epoch_{epoch:02d}_trn_{acc:.4f}_val_{val_acc:.4f}.hdf5')

model_checkpoint = callbacks.ModelCheckpoint(
    						filepath= path_model,
    						monitor="val_acc",
    						mode="max",
    						verbose=0,
    						save_best_only=False,
                            save_weights_only=False        
                            )    

history = model.fit([x_train1, x_train2, x_train3, x_train4], y_train, 
                    batch_size=128, 
                    epochs=50,
                    validation_data=([x_val1, x_val2, x_val3, x_val4], y_val),
                    shuffle=True,
                    verbose=1,
                    callbacks= [model_checkpoint, tensorboard1, logger, reduce_LR])









