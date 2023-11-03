# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 13:57:44 2019

@author: Administrator
"""

import config_models as config


from Models.cnn2_aten import create_model
model = create_model(config.inp_shape, config)
model.summary()