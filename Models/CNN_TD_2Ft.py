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
def create_model(input_shape1, input_shape2, config):
    
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
    x1_embed = GlobalAveragePooling1D(name='embed_1')(x1)
    
    
    ##=====================CNN2
    input_tensor2 = Input(shape=input_shape2)
    x2=input_tensor2
    for i in range(len(config.P2_cnns)):
        x2 = Convolution2D(config.P2_cnns[i], config.P2_kernel[i], padding=config.P2_pad, kernel_initializer = 'he_normal', activation="relu")(x2)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(config.P2_pool_size, config.P2_strides, padding=config.P2_pad)(x2)
    
    x2 = Permute((2,1,3))(x2)
    bs, a, b, c = x2._keras_shape
    x2 = Reshape((a, b*c))(x2)   
    
    x2 = TimeDistributed(Dense(x1_embed._keras_shape[-1], activation='relu'))(x2)
    x2_embed = GlobalAveragePooling1D(name='embed_2')(x2)
    
    y = layers.concatenate([x1_embed, x2_embed],axis=-1) # axis????
    
    
    ##===================== FCN2
    for i in range(len(config.P_dnns)):
        y=Dense(config.P_dnns[i], kernel_initializer = 'he_normal', activation="relu")(y)
   
    ##==================== Output    
    output_final = Dense(config.num_classes, activation="softmax", name='outfinal')(y)
    
    
    model=Model([input_tensor1, input_tensor2], output_final)
    
    return model
    



# input_shape1=(64,64,1)
# input_shape2=(12,64,1)

# import config_models_temp as config
# model = create_model(input_shape1,input_shape2, config)
# model.summary()

  
    














