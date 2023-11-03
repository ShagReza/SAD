# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:14:35 2019

@author: Administrator
"""
import tensorflow as tf

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import *
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import TensorBoard, CSVLogger

#from keras.utils.io_utils import HDF5Matrix
import config_models as config

import os

##
if not os.path.exists(config.folder_model):
    os.mkdir(config.folder_model)
    
if not os.path.exists(config.folder_log):   os.makedirs(config.folder_log)

##==========================
x_train = np.load('./Results_Stage4/x_SnNM_log_16053500370.npy')
y_train = np.load('./Results_Stage4/y_SnNM_log_16053500370.npy')

#x_train = np.load('./Results_Stage4/x_SnNM_16060540375_chroma12.npy')
#y_train = np.load('./Results_Stage4/y_SnNM_16060540375_chroma12.npy')

#x_train = np.load('./Results_Stage4/x_SNM_MuSS.npy')
#y_train = np.load('./Results_Stage4/y_SNM_MuSS.npy')

y_train = to_categorical(y_train)

x_val = np.load('./Results_Stage4/x_SnNM_log_15050160320_val.npy')
y_val = np.load('./Results_Stage4/y_SnNM_log_15050160320_val.npy')

#x_val = np.load('./Results_Stage4/x_SnNM_16060250375_val_chroma12.npy')
#y_val = np.load('./Results_Stage4/y_SnNM_16060250375_val_chroma12.npy')

y_val = to_categorical(y_val)

##==========================================
import config_models as config
# from Models.cnn2 import create_model
# model = create_model(config.inp_shape, config)

from Models.cnn2_aten import create_model
model = create_model(config.inp_shape, config)

model.summary()

opt1 = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt1, loss='categorical_crossentropy', metrics= ['accuracy'])

##==========================================
reduce_LR = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)

tensorboard1 = TensorBoard(log_dir = config.folder_log)
logger = CSVLogger(config.folder_log + '/training.log')
path_model = os.path.join(config.folder_model, 'model_epoch_{epoch:02d}_trn_{accuracy:.4f}_val_{val_accuracy:.4f}.hdf5')

model_checkpoint = callbacks.ModelCheckpoint(
    						filepath= path_model,
    						monitor="val_accuracy",
    						mode="max",
    						verbose=0,
    						save_best_only=False,
                            save_weights_only=False        
                            )    

history = model.fit(x_train, y_train, 
                    batch_size=128, 
                    epochs=50,
                    validation_data=(x_val, y_val),
                    shuffle=True,
                    verbose=1,
                    callbacks= [
                        model_checkpoint
                        # , tensorboard1
                        , logger
                        , reduce_LR
                        ]
                    )



