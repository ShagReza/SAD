# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:49:33 2019

@author: Kashani
"""

from keras.layers.core import Dense, Flatten
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential
from keras.regularizers import l2

NAME = "CNN"

def create_model(input_shape, config, is_training=True):

#    weight_decay = 0.001

    model = Sequential()
    model.add(Reshape((7, 64), input_shape=(7, 64, 1)))
    
    for i in range(len(config.cnns)):
        model.add(Convolution1D(config.cnns[i], config.kernel[i], padding=config.pad, kernel_initializer = 'he_normal', activation="relu", input_shape=input_shape, dilation_rate=config.dil))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(config.pool_size, config.strides))

#    model.add(Flatten())
    model.add(GlobalAveragePooling1D())
    
    for i in range(len(config.dnns)):
        model.add(Dense(config.dnns[i], kernel_initializer = 'he_normal', activation="relu"))

    model.add(Dense(config.num_classes, activation="softmax"))

    return model