# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:16:58 2019

@author: user
"""

from keras.layers.core import Dense, Flatten
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

from keras.models import Model
from keras import layers
from keras import Input
NAME = "ParallelCNN2"

def create_model(input_shape1,input_shape2, config, is_training=True):
    
    #CNN1
    input_tensor1 = Input(shape=input_shape1)
    x1=input_tensor1
    for i in range(len(config.P1_cnns)):
        x1 = Convolution2D(config.P1_cnns[i], config.P1_kernel[i], padding=config.P1_pad, kernel_initializer = 'he_normal', activation="relu")(x1)
        x1=BatchNormalization()(x1)
        x1=MaxPooling2D(config.P1_pool_size, config.P1_strides)(x1)
    x1=GlobalAveragePooling2D()(x1)       
    
    #CNN2
    input_tensor2 = Input(shape=input_shape2)
    x2=input_tensor2
    for i in range(len(config.P2_cnns)):
        x2 = Convolution2D(config.P2_cnns[i], config.P2_kernel[i], padding=config.P2_pad, kernel_initializer = 'he_normal', activation="relu")(x2)
        x2=BatchNormalization()(x2)
        x2=MaxPooling2D(config.P2_pool_size, config.P2_strides)(x2)
    x2=GlobalAveragePooling2D()(x2)
          
    #Concatenate
    concatenated = layers.concatenate([x1, x2],axis=-1) # axis????
    y=concatenated
    
    #Classifier
    for i in range(len(config.P_dnns)):
        y=Dense(config.P_dnns[i], kernel_initializer = 'he_normal', activation="relu")(y)
        
    #Output
    
    output=Dense(config.num_classes, activation="softmax")(y)
    
    model=Model([input_tensor1, input_tensor2], output)
    
    return model
    



#input_shape1=(256,64,1)
#input_shape2=(12,64,1)
#import config_models as config
#model = create_model(input_shape1, input_shape2, config)
#model.summary()

  
    














