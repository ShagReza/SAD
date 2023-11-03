# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:08:10 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:16:58 2019

@author: user
"""
import keras.backend as K
#from keras.layers.core import Dense, Flatten
from keras.layers import *
from keras.activations import *
#from keras.layers.pooling import MaxPooling2D

from keras.models import Model
from keras import layers
from keras import Input
#NAME = "ParallelCNN2"

##=====================================    
def create_model(input_shape1, config):
    
    ##=====================CNN1
    input_tensor1 = Input(shape=input_shape1)
    x1=input_tensor1
    for i in range(len(config.P1_cnns)):
        x1 = Convolution2D(config.P1_cnns[i], config.P1_kernel[i], padding=config.P1_pad, kernel_initializer = 'he_normal', activation="relu")(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(config.P1_pool_size, config.P1_strides, padding=config.P1_pad)(x1)
    
    x1 = Permute((2,1,3))(x1)
    bs, a, b, c = x1._keras_shape
    x1 = Reshape((a, b*c))(x1)   
    
    x1 = TimeDistributed(Dense(x1._keras_shape[-1]*2, activation='relu'))(x1)
    x1 = GlobalAveragePooling1D()(x1)
    
    for i in range(len(config.dnns)):
        x1=Dense(config.dnns[i], kernel_initializer = 'he_normal', activation="relu")(x1)
   
    ##==================== Output    
    output_final = Dense(config.num_classes, activation="softmax", name='outfinal')(x1)
    
    
    model=Model([input_tensor1], output_final)
    
    return model
    



# input_shape1=(64,64,1)
# import config_models_temp as config
# model = create_model(input_shape1, config)
# model.summary()

  
    














