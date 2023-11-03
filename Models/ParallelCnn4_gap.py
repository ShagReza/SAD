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

def create_model(input_shape1,input_shape2, input_shape3, input_shape4, config, is_training=True):
    
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
    
    #CNN3
    input_tensor3 = Input(shape=input_shape3)
    x3=input_tensor3
    for i in range(len(config.P3_cnns)):
        x3 = Convolution2D(config.P3_cnns[i], config.P3_kernel[i], padding=config.P3_pad, kernel_initializer = 'he_normal', activation="relu")(x3)
        x3=BatchNormalization()(x3)
        x3=MaxPooling2D(config.P3_pool_size, config.P3_strides)(x3)
    x3=GlobalAveragePooling2D()(x3)
     
    #CNN4
    input_tensor4 = Input(shape=input_shape4)
    x4=input_tensor4
    for i in range(len(config.P4_cnns)):
        x4 = Convolution1D(config.P4_cnns[i], config.P4_kernel[i], padding=config.P4_pad, kernel_initializer = 'he_normal', activation="relu")(x4)
        x4=BatchNormalization()(x4)
        x4=MaxPooling1D(config.P4_pool_size, config.P4_strides)(x4)
    x4=GlobalAveragePooling1D()(x4)
    
    #Concatenate
    concatenated = layers.concatenate([x1, x2, x3, x4],axis=-1) # axis????
#    concatenated123 = layers.concatenate([concatenated12, x3],axis=-1)
#    concatenated = layers.concatenate([concatenated123, x4],axis=-1)
    y=concatenated
    
    #Classifier
    for i in range(len(config.P_dnns)):
        y=Dense(config.P_dnns[i], kernel_initializer = 'he_normal', activation="relu")(y)
        
    #Output
    
    output=Dense(config.num_classes, activation="softmax")(y)
    
    model=Model([input_tensor1, input_tensor2, input_tensor3, input_tensor4], output)
    
    return model
    



#input_shape1=(256,64,1)
#input_shape2=(12,64,1)
#import config_models as config
#model = create_model(input_shape1, input_shape2, config)
#model.summary()

  
    














