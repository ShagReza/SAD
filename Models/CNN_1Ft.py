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
from keras.layers.core import Dense, Flatten
from keras.layers import *
from keras.activations import *
#from keras.layers.pooling import MaxPooling2D

from keras.models import Model
from keras import layers
from keras import Input
#NAME = "ParallelCNN2"

##=====================================
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()


##=====================================    
def create_model(input_shape1, config, atn=True):
    
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
    
    if atn:
        att_out=attention()(x1)
    else:
        att_out = GlobalAveragePooling1D()(x1)
   
    for i in range(len(config.P_dnns)):
        att_out = Dense(config.P_dnns[i], kernel_initializer = 'he_normal', activation="relu")(att_out)

    ##==================== Output    
    output_final = Dense(config.num_classes, activation="softmax", name='outfinal')(att_out)
    
    
    model=Model([input_tensor1], output_final)
    
    return model
    



# input_shape1=(64,64,1)
# import config_models as config
# model = create_model(input_shape1, config, atn=True)
# model.summary()

  
    














