# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:49:33 2019

@author: Administrator
"""

from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Dense, Permute, Reshape
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM


NAME = "CRNN"

def create_model(input_shape, config, is_training=True):

#    weight_decay = 0.001

    model = Sequential()
    
    for i in range(len(config.cnns)):
        model.add(Convolution2D(config.cnns[i], config.kernel[i], padding=config.pad, kernel_initializer = 'he_normal', activation="relu", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(config.pool_size, config.strides))

     # (bs, y, x, c) --> (bs, x, y, c)
    model.add(Permute((2, 1, 3)))

    # (bs, x, y, c) --> (bs, x, y * c)
    bs, x, y, c = model.layers[-1].output_shape
    model.add(Reshape((x, y*c)))

    for i in range(len(config.rnns)-1):
        model.add(Bidirectional(LSTM(config.rnns[i], return_sequences=True), merge_mode="concat"))
    model.add(Bidirectional(LSTM(config.rnns[len(config.rnns)-1], return_sequences=False), merge_mode="concat"))
    
#    for i in range(len(config.dnns)):
#        model.add(Dense(config.dnns[i], kernel_initializer = 'he_normal', activation="relu"))

    model.add(Dense(config.num_classes, activation="softmax"))

    return model


import config_models_temp as config
model = create_model((64,64,1), config, is_training=True)
model.summary()
