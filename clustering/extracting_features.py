from __future__ import division
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image

from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import sys

#from networks import largest as Net
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
#Añadida esta NET por mí, porque hacia referencia a una lib networks que no encuentro
def flatten(x):
    """Flattens a tensor."""
    return x.view(x.size(0), -1)

class Net(nn.Module):
    def __init__(self,nb_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,92,5,1,2)
        self.pool1 = nn.MaxPool2d(3,2)
        self.drop1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(92,256,5,1,2)
        self.relu = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3,2)
        self.drop2 = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(256,512,5,padding=2)
        self.drop3 = nn.Dropout2d(0.25)
        self.flat = flatten
        self.fc1 = nn.Linear(25088,1024)
        self.dropDense1 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(1024,16000)
        self.sfmx = nn.Softmax(1)

    def forward(self, x):
        out = self.pool1(self.relu(self.conv1(x)))
        out = self.drop1(out)
        out = self.pool2(self.relu(self.conv2(out)))
        out = self.drop2(out)
        out = self.drop3(self.relu(self.conv3(out)))
        out = self.flat(out)
        out = self.relu(self.fc1(out))
        out = self.dropDense1(out)
        out = self.relu(self.fc2(out))
        out = self.sfmx(out)
        return out
        
# instantiating the net
net = Net(nb_classes)
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

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3,92,5,1,2)
        self.pool1 = nn.MaxPool2d(3,2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.pool1(self.relu(self.conv1(x)))
        return out
        
net1 = Net1()

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3,92,5,1,2)
        self.pool1 = nn.MaxPool2d(3,2)
        self.drop1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(92,256,5,1,2)
        self.relu = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3,2)
       
    def forward(self, x):
        out = self.pool1(self.relu(self.conv1(x)))
        out = self.drop1(out)
        out = self.pool2(self.relu(self.conv2(out)))
        return out
        
net2 = Net2()

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3,92,5,1,2)
        self.pool1 = nn.MaxPool2d(3,2)
        self.drop1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(92,256,5,1,2)
        self.relu = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3,2)
        self.drop2 = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(256,512,5,padding=2)

    def forward(self, x):
        out = self.pool1(self.relu(self.conv1(x)))
        out = self.drop1(out)
        out = self.pool2(self.relu(self.conv2(out)))
        out = self.drop2(out)
        out = self.relu(self.conv3(out))
        return out
        
net3 = Net3()


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

    samples = Variable(images)

    net.train(False)
    out_conv1 = net1(samples)
    out_conv2 = net2(samples)
    out_conv3 = net3(samples)
    out_f = process_features_batch(out_conv1, out_conv2, out_conv3, batch_size)

    if "/features" not in f_h5py:
        f_h5py.create_dataset('features', shape = (nb_images, out_f.shape[1]), dtype = np.float32)

    for feat in out_f:
        f_h5py['features'][idx, ...] = feat
        idx += 1


f_h5py.close()
print ('Features extracted from ' + str(model) + '!!')
print ('Features saved in: ' + output_path)
