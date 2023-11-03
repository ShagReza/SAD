# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:59:58 2020

@author: Administrator
"""
import keras
import tensorflow as tf
import numpy as np
from keras.layers import *
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
import time
#from SAD_parameters import *
#from data_loader import Data

def ConvMPBlock(x, num_convs=2, fsize=32, kernel_size=3, pool_size=(2,2), strides=(2,2), BN=False, DO=False, MP=True):
    for i in range(num_convs):
       x = Conv2D(fsize, kernel_size, padding='same')(x)
       if BN:
           x = BatchNormalization()(x)
       if DO:
           x = Dropout(DO)(x)
       x = Activation('relu')(x)
    if MP:
        x = MaxPooling2D(pool_size=pool_size, strides=strides, padding='same')(x)
    return x

def FullyConnectedLayer(x, nodes=512, act='relu', BN=False, DO=False):
    x = Dense(nodes)(x)
    if BN:
        x = BatchNormalization()(x)
    if DO:
        x = Dropout(DO)(x)
    x = Activation(act)(x)
    return x

#''' Define Speech activity detection model.
#'''
def define_keras_model(input_shape= (64, 64, 1), optimizer='adam', loss='binary_crossentropy'):    
    fsize = 32
    td_dim = 256
    inp = Input(shape=input_shape)
    x = ConvMPBlock(inp, num_convs=2, fsize=fsize, BN=True)
    x = ConvMPBlock(x, num_convs=2, fsize=2*fsize, BN=True)
    x = ConvMPBlock(x, num_convs=3, fsize=4*fsize, BN=True)
    x = Reshape((x._keras_shape[1], x._keras_shape[2]*x._keras_shape[3]))(x)
    x = TimeDistributed(Dense(td_dim, activation='relu'))(x)
#    x = Dense(td_dim, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = FullyConnectedLayer(x, 128, BN=True)
    x = FullyConnectedLayer(x, 64, BN=True)
    x = FullyConnectedLayer(x, 2, 'softmax')
    model = Model(inp, x)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


# model = define_keras_model(input_shape= (64, 64, 1), optimizer='adam', loss='binary_crossentropy')

# model.summary()

