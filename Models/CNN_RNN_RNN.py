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

from keras.layers.core import Dense, Flatten
from keras.layers import *
from keras.activations import *
#from keras.layers.pooling import MaxPooling2D

from keras.models import Model
from keras import layers
from keras import Input
#NAME = "ParallelCNN2"

def create_model(input_shape1, input_shape2, config):
    
    ##=====================CNN1
    input_tensor1 = Input(shape=input_shape1)
    x1=input_tensor1
    for i in range(len(config.P1_cnns)):
        x1 = Convolution2D(config.P1_cnns[i], config.P1_kernel[i], padding=config.P1_pad, kernel_initializer = 'he_normal', activation="relu")(x1)
        x1=BatchNormalization()(x1)
        x1=MaxPooling2D(config.P1_pool_size, config.P1_strides)(x1)
    
    x1 = Permute((2,1,3))(x1)
#    (_, h, w, c) = x1.shape
    bs, a, b, c = x1._keras_shape
    x1 = Reshape((a, b*c))(x1)   
    
    for i in range(len(config.P1_rnns)-1):
        x1 = Bidirectional(GRU(config.P1_rnns[i], return_sequences=True), merge_mode="concat")(x1)

    x1_embed = Bidirectional(GRU(config.P1_rnns[-1], return_sequences=True), merge_mode="concat", name='embed_1')(x1) 
    
    ##=====================CNN2
    input_tensor2 = Input(shape=input_shape2)
    x2=input_tensor2
    for i in range(len(config.P2_cnns)):
        x2 = Convolution2D(config.P2_cnns[i], config.P2_kernel[i], padding=config.P2_pad, kernel_initializer = 'he_normal', activation="relu")(x2)
        x2=BatchNormalization()(x2)
        x2=MaxPooling2D(config.P2_pool_size, config.P2_strides)(x2)
     
    x2 = Permute((2,1,3))(x2)
    bs, a, b, c = x2._keras_shape
    x2 = Reshape((a, b*c))(x2)    
    
    for i in range(len(config.P2_rnns)-1):
        x1 = Bidirectional(GRU(config.P2_rnns[i], return_sequences=True), merge_mode="concat")(x1)
    
    x2_embed = Bidirectional(GRU(config.P2_rnns[-1], return_sequences=True), merge_mode="concat", name='embed_2')(x2)      
    
    ##==================== Concatenate
    y = layers.concatenate([x1_embed, x2_embed],axis=-1, name='Concat_embed') # axis????
    
    
    #====================Classifier
    for i in range(len(config.P_rnns)-1):
        y = Bidirectional(GRU(config.P_rnns[i], return_sequences=True), merge_mode="concat")(y)
    
    y = Bidirectional(GRU(config.P_rnns[-1], return_sequences=False), merge_mode="concat")(y)    
    
    ##==================== Output    
    output_final = Dense(config.num_classes, activation="softmax", name='outfinal')(y)
    
    
    model=Model([input_tensor1, input_tensor2], output_final)
    
    return model
    



#input_shape1=(64,64,1)
#input_shape2=(12,64,1)
#import config_models as config
#model = create_model(input_shape1, input_shape2, config)
#model.summary()

  
    














