# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:04:52 2020

@author: Administrator
"""

from keras.models import Model
from keras.layers import LSTM,Dense,Input,Bidirectional





def seq2seq(fea_dim = 24 , output_dim=2) :
   inputs = Input((None,fea_dim), name='input')
   x = Bidirectional(LSTM(32, return_sequences=True ) , name='b1')(inputs)
   x = Bidirectional(LSTM(20, return_sequences=True ),name = 'b2')(x)
   x = Dense(40, activation='relu' , name='d1')(x)
   x = Dense(10, activation='relu',name='d2')(x)
   x = Dense(output_dim, activation='softmax',name='d3')(x)
   model = Model(inputs, x)
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   return model

model = seq2seq(24,2)
model.summary()