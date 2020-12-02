from __future__ import division
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
#############################################################
# Se extraen las caracteristicas con el modelo anterioremente entrenado, y un conjunto de imagenes
# distinto al anterior.
# Se guardan las features en un fichero hdf
############################################################ 
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import sys

from networks import largest as Net
import h5py

from sklearn.decomposition import PCA
from utils_clust import normalizing_samples_L2

import random
random.seed(42)

import time

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from skcuda import linalg
import skcuda
# setting parameters
exp = '083'
#############################################
classes_path = '../surrogate_dataset/train_set/'
labels = os.listdir(classes_path)
# number of classes with which the network was trained
nb_classes = len(labels) 
##############################################

mean = [0.383661700858527, 0.3819784115384924, 0.3588786631614881]
std=[0.2167717755518767, 0.21201058526724945, 0.21143164036556178]

#mean = [0.5, 0.5, 0.5]
#std = [0.5, 0.5, 0.5]

epochs = ['10']
model = '../saving_model/exp' + exp + '/exp' + exp + '_epoch_' + epochs[0] + '.pth.tar'
# some variables
feat_side = 2
test_number = 'maxpool_allConv'

# setting paths
# 1st iteration path
images_path = '../surrogate_dataset/unlab_set/'

output = './less_collisions/'
image_names = output + 'features_' + test_number +'.txt'
output_path = output + 'features_' + test_number +'.hdf5'

print ("Extracting features from model in: " + exp)
# Load the images
image_list = os.listdir(images_path)
image_list.sort()
nb_images = len(image_list)
#nb_images = len(image_list[:64000])
#image_list = image_list[:64000]

# saving the labels of the images in the labels_path file
f_names = open(image_names, 'w')
for i in image_list:
    f_names.write(i + '\n')    
f_names.close()
        
# instantiating the net
net = Net(nb_classes).cuda()
# defining transformations:
normalize = transforms.Normalize(mean = mean, std=std)
transf = transforms.Compose([transforms.ToTensor(), normalize])

# loading the model
checkpoint = torch.load(model)
net.load_state_dict(checkpoint['state_dict'])

# We fix the size of the features that we will then flat and concatenate
pooling_to_fixed_value = nn.AdaptiveMaxPool2d(feat_side)

def process_layers(out_conv1, out_conv2, out_conv3):
    # set the layer config for extracting the features 
    out_c1 = pooling_to_fixed_value(out_conv1)
    out_c2 = pooling_to_fixed_value(out_conv2)
    out_c3 = pooling_to_fixed_value(out_conv3)
    out_c = torch.cat([out_c1, out_c2, out_c3], dim = 1)
    return out_c

def process_features_batch(out_conv1, out_conv2, out_conv3, batch_size):
    # process the extracteed features batch by batch (batch_size, nb_feats, height, length)
    out_c = process_layers(out_conv1, out_conv2, out_conv3)
    out_f = out_c.squeeze().cpu().data.view(batch_size, -1).numpy()
    return out_f

batch_size = 9
idx = 0
##############################################3
images = torch.zeros((batch_size, 3, 96, 96))
f_h5py = h5py.File(output_path, 'w')

for iteration in tqdm(range(int(len(image_list)/batch_size))):
    
    for num, image in enumerate(image_list[idx:idx+batch_size]):
        image = Image.open(images_path + image)
        image = transf(image)
        images[num] = image.float()

    samples = Variable(images).cuda()

    net.train(False)
    (out_conv1, out_conv2, out_conv3) = net.forward_all_conv_feat(samples)

    out_f = process_features_batch(out_conv1, out_conv2, out_conv3, batch_size)
    
    if "/features" not in f_h5py:
        f_h5py.create_dataset('features', shape = (nb_images, out_f.shape[1]), dtype = np.float32)

    for feat in out_f:
        f_h5py['features'][idx, ...] = feat
        idx += 1


f_h5py.close()
print ('Features extracted from ' + str(model) + '!!')
print ('Features saved in: ' + output_path)
