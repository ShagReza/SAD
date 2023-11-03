# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:16:03 2019

@author: Administrator
"""

import os

folder_model = "./Trained_MDLs"
#model_name = 'CNN_14_log_normChroma_SadFeats_Drug_C4D3_GAP'
model_name = 'CNN_1_log_[2C_1D]_winMVN2sec'
#model_name = 'CNN_1_log_C4D3_GAP_winMVN2sec_addMusanNos'

#model_name = 'CNN_8_Drug_C4D1_GAP_2'



folder_log = './Log_Model'
#folder_log = './Log_Model_MIMO'

if not os.path.exists(folder_model):
    os.mkdir(folder_model)

if not os.path.exists(folder_log):
    os.mkdir(folder_log)
    
folder_model = os.path.join(folder_model, model_name)
folder_log = os.path.join(folder_log, model_name)

num_classes = 3


## cnn2

#### mel-log
#inp_shape = (64,64,1)	
#cnns = (16, 32, 32, 32)	
#kernel = (7,5,3, 3)	
#pad = 'same'  ## 'same' or 'valid'	
#pool_size = (2,2)	
#strides = (2,2)	
#dil = 1	
#dnns = (512,)	

#### chroma and sad feats
#inp_shape = (12,64,1)	
#cnns = (16,32,32,32)	
#kernel = (5,3,3,3)	
#pad = 'same'  ## 'same' or 'valid'	
#pool_size = (2,2)	
#strides = (1,2)	
#dil = 1	
#dnns = (512,)	

### Drug
#inp_shape = (7,64,1)	
#cnns = (16,32,32,32)	
#kernel = (5,3,3,3)	
#pad = 'same'  ## 'same' or 'valid'	
#pool_size = (2,)	
#strides = (1,)	
#dil = 1	
#dnns = (512,)	

## crnn2

#cnns = (16, 32, 32, 32, 32)
#kernel = (7,5,3,3,3)
#
#pad = 'same'  ## 'same' or 'valid'
#pool_size = (2,2)
#strides = (2,1)
#
#rnns = (256,128) 
#


##-------------------------چ
#CNN1
#P1_cnns = (16, 32, 32, 32)
#P1_kernel = (7,5,3,3)
#P1_pad = 'same'  ## 'same' or 'valid'
#P1_pool_size = (2,2)
#P1_strides = (2,2)
#
#P2_cnns = (16,32,32,32)
#P2_kernel = (5,3,3,3)
#P2_pad = 'same'  ## 'same' or 'valid'
#P2_pool_size = (2)
#P2_strides = (1)
#
#P_dnns = (512,256,128)

#-----------------------------Prallel CNN (Pcnn2)----------------------------
#CNN1
#P1_cnns = (16, 32, 32, 32)
#P1_kernel = (7,5,3,3)
#P1_pad = 'same'  ## 'same' or 'valid'
#P1_pool_size = (2,2)
#P1_strides = (2,2)
##P1_strides = (2,1)  ## for rnn
#
##CNN2
#P2_cnns = (16,32,32,32)
#P2_kernel = (5,3,3,3)
#P2_pad = 'same'  ## 'same' or 'valid'
#P2_pool_size = (2,2)
#P2_strides = (1,2)
##P2_strides = (1,1)   ## for rnn
#
#P3_cnns = (16,32,32,32)
#P3_kernel = (5,3,3,3)
#P3_pad = 'same'  ## 'same' or 'valid'
#P3_pool_size = (2,2)
#P3_strides = (1,2)
#
#P4_cnns = (16,32,32,32)
#P4_kernel = (5,3,3,3)
#P4_pad = 'same'  ## 'same' or 'valid'
#P4_pool_size = (2)
#P4_strides = (1)
#
##DNN
#P1_dnns = (512,)
#P2_dnns = P1_dnns
#
#P_dnns = (512,256,128)
##P_dnns = (256,128)
#P_rnns = (256,)


#-----------------------------Prallel CNN-RNN (Pcnn2)----------------------------
#CNN1
P1_cnns = (16, 32)
P1_kernel = (7,5)
P1_pad = 'same'  ## 'same' or 'valid'
P1_pool_size = (2,2)
#P1_strides = (2,2)
P1_strides = (2,1)  ## for rnn

#CNN2
P2_cnns = (16,32,32,32)
P2_kernel = (5,3,3,3)
P2_pad = 'same'  ## 'same' or 'valid'
P2_pool_size = (2,2)
#P2_strides = (1,2)
P2_strides = (1,1)   ## for rnn

#P3_cnns = (16,32,32,32)
#P3_kernel = (5,3,3,3)
#P3_pad = 'same'  ## 'same' or 'valid'
#P3_pool_size = (2,2)
#P3_strides = (1,2)
#
#P4_cnns = (16,32,32,32)
#P4_kernel = (5,3,3,3)
#P4_pad = 'same'  ## 'same' or 'valid'
#P4_pool_size = (2)
#P4_strides = (1)

#DNN
P1_rnns = (128,)
P2_rnns = P1_rnns

P_rnns = (128,)
P_dnns = (256,)
#P_rnns = (256,)



