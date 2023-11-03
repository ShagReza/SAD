# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 13:51:26 2019

@author: Administrator
"""
import tensorflow as tf
from tensorflow.keras.layers import *

from tensorflow.keras.models import Model
# from tensorflow.keras import layers
from tensorflow.keras import Input
NAME = "Attention CNN"

def CAM(i):
  (_, h, w, c) = i.shape
  x = Conv2D(c, (1, 1), padding='same', kernel_initializer = 'he_normal')(i)
  xp = Permute((3,1,2))(x)

  x_rsh = Reshape((c,h*w))(xp)
  x_rsht = Permute((2,1))(x_rsh)

  z1 = tf.matmul(x_rsh,x_rsht)
  z2 = Activation('softmax')(z1)

  z3 = tf.matmul(z2,x_rsh)
  z3_rsh = Reshape((c,h,w))(z3)
  z3_rsht = Permute((2,3,1))(z3_rsh)

  z4 = Conv2D(c, (1, 1), padding='same', kernel_initializer = 'he_normal')(z3_rsht)
  o  = Add()([x, z4])
  return o

def create_model(input_shape1, config, is_training=True):
    
    #CNN1
    input_tensor1 = Input(shape=input_shape1)
    x1=input_tensor1
    for i in range(len(config.cnns)):        
        x1 = Conv2D(config.cnns[i], config.kernel[i], padding=config.pad, kernel_initializer = 'he_normal', activation="relu", dilation_rate=config.dil)(x1)
        x1=BatchNormalization()(x1)
        x1=MaxPooling2D(config.pool_size, config.strides)(x1)
    
    x1 = CAM(x1)
    
    x1=Flatten()(x1)       
    
    #Dense 
    for i in range(len(config.dnns)):
        x1=Dense(config.dnns[i], kernel_initializer = 'he_normal', activation="relu")(x1)
        
    #Output
    
    output=Dense(config.num_classes, activation="softmax")(x1)
    
    model=Model(input_tensor1, output)
    
    return model




