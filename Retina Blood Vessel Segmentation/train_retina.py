# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019

@author: Reza winchester
"""
from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import models as M
import numpy as np
from help_functions import *
from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau
from keras import callbacks

#========= Load settings from Config file
#patch to the datasets
path_data = './DRIVE_datasets_training_testing/'
#Experiment name
name_experiment = 'test'
#training settings

batch_size = 8

####################################  Load Data #####################################3
patches_imgs_train  = np.load('patches_imgs_train.npy')
patches_masks_train = np.load('patches_masks_train.npy')

patches_imgs_train = np.einsum('klij->kijl', patches_imgs_train)
patches_masks_train = np.einsum('klij->kijl', patches_masks_train)


print('Patch extracted')

#model = M.unet2_segment(input_size = (64,64,1))
model = M.BCDU_net_D3(input_size = (64,64,1))
model.summary()

print('Training')

nb_epoch = 50

mcp_save = ModelCheckpoint('weight_lstm.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

history = model.fit(patches_imgs_train,patches_masks_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_split=0.2, callbacks=[mcp_save, reduce_lr_loss] )




