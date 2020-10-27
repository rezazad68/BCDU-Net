from __future__ import division
import nibabel as nib
import numpy as np
import Reza_functions as RF
import nibabel as nib
import glob
import os

# Define Train data and mask
Data_train   = []
Mask_train   = []
Maska_train  = []
FOV_train    = []

idx_count =1
Tr_add = '3d_images'

Tr_list = glob.glob(Tr_add+'/*.gz')

for idx in range(len(Tr_list)):
    b = Tr_list[idx]
    a = b[len(Tr_add)+1:len(Tr_add)+4]
    if a=='IMG':
       print(idx_count)
       a = b[len(Tr_add)+5:len(b)]
       add = (Tr_add+'/MASK_' + a) 
       vol = nib.load(Tr_list[idx])
       seg = nib.load(add)
       # Get the axials images and corresponding masks
       vol_ims, lung, around_lung, FOV = RF.return_axials(vol, seg)          
       segmentation  = seg.get_data()
       # Insert samples to the Train data, which has the segmentation label
       for idx in range(vol.shape[0]):
           if ~( np.sum(np.sum(np.sum(segmentation[idx, :, :]))) == 0): 
               Data_train.append(vol_ims [idx, :, :])
               Mask_train.append(lung[idx, :, :])
               Maska_train.append(around_lung[idx, :, :])               
               FOV_train.append(FOV[idx, :, :])               
       idx_count += 1
        
Data_train  = np.array(Data_train)
Mask_train  = np.array(Mask_train)
Maska_train = np.array(Maska_train)
FOV_train   = np.array(FOV_train)

# We use 70% of the data for training and 30% for test
alpha = np.int16(np.floor(Data_train.shape[0]* 0.7))
en_d  = Data_train.shape[0]

Train_img      = Data_train[0:alpha,:,:]
Test_img       = Data_train[alpha:en_d,:,:]

Train_mask     = Mask_train[0:alpha,:,:]
Test_mask      = Mask_train[alpha:en_d,:,:]

Train_maska     = Maska_train[0:alpha,:,:]
Test_maska      = Maska_train[alpha:en_d,:,:]

FOV_tr     = FOV_train[0:alpha,:,:]
FOV_te      = FOV_train[alpha:en_d,:,:]

folder = './processed_data/'
if not os.path.exists(folder):
    os.makedirs(folder)
    
np.save(folder+'data_train' , Train_img)
np.save(folder+'data_test'  , Test_img)
np.save(folder+'mask_train' , Train_mask)
np.save(folder+'mask_test'  , Test_mask)

np.save(folder+'Train_maska' , Train_maska)
np.save(folder+'Test_maska'  , Test_maska)
np.save(folder+'FOV_tr'      , FOV_tr)
np.save(folder+'FOV_te'      , FOV_te)


