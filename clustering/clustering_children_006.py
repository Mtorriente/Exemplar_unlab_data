import numpy as np
import os
from shutil import copyfile, copytree
from tqdm import tqdm

import sys
from chainercv import utils
from torchvision import transforms

from PIL import Image
from matplotlib import pyplot as plt

import shutil
from os.path import join

from utils_clust import normalizing_samples_L2, loading_images, searching_similar_images

# To load pairs of children and...
iteration_nb = '001_retrieval'
childrens_nb = '001'
path_pairs_out = './less_collisions/image_pairs' + childrens_nb
path_pairs_out_larger_clusters = './less_collisions/image_pairs_mixed_nodes' + childrens_nb
# other paths
path_target_dset_s_cl = './less_collisions/dset' + iteration_nb + '_short_cl/'
path_target_dset_l_cl = './less_collisions/dset' + iteration_nb + '_large_cl/'
path_new_classes = './less_collisions/new_classes_' + iteration_nb + '/'
path_target_dset_single = './less_collisions/dset' + iteration_nb + '_clusters_from_single_images'
path_single_old_classes = './less_collisions/old_classes_' + iteration_nb + '/'
#Este lo añado yo que no aparecia definido
path_target_dset = './less_collisions'
# source images
images_path = '../surrogate_dataset/unlab_set/'

short_clusters = []
larger_clusters = []

if os.path.exists(path_pairs_out  + '.npy'):
    short_clusters = np.load(path_pairs_out +'.npy')
if os.path.exists(path_pairs_out_larger_clusters  + '.npy'):
    larger_clusters = np.load(path_pairs_out_larger_clusters + '.npy',allow_pickle=True)
    
print ('Arrays loaded stats: ')
print("children_array:",len(short_clusters), type(short_clusters), short_clusters.shape)

# We need a set of all the images selected for clustering. 
# From there we will remove each image when it is added to a cluster.

samples_set_l = set()
for cluster_l in larger_clusters:
    for sample_l in cluster_l:
        samples_set_l.add(sample_l)
print ("Larger clusters introduced. Length: ",len(samples_set_l))
        
# I introduce the samples from the simple clusters as well
samples_set_s = set()
for cluster_s in short_clusters:
    for sample_s in cluster_s:
        samples_set_s.add(sample_s)
        
print ("Short clusters (pairs) introduced. Length: ",len(samples_set_s))

print("sub_child_mixed_array:",len(larger_clusters), type(larger_clusters), larger_clusters.shape)


# we need a set with all the 100k images from the dataset
samples_full = os.listdir(images_path)
nb_samples = len(samples_full)
samples_full_set = set([int(sample_i[:-4]) for sample_i in samples_full])
print (len(samples_full), len(samples_full_set), type(samples_full_set))


# first we invert the larger_cluster array. This is to start clustering the larger groups.
# This is needed because the tree structure made clusters that are inside other clusters...
larger_clusters = larger_clusters[::-1]
# Note that we do not need to do that with the simple clusters


cluster_number = 0    # variable to count clusters

for cluster_l in larger_clusters:
    nb_img = len(cluster_l)
    num = 1
    temp = 0    # variable to count clusters
    for sample_l in cluster_l: 
           
        if sample_l in samples_set_l:
            temp +=1    # variable to count clusters
            
            ## defining paths 
            if not os.path.exists(join(path_target_dset_l_cl, 'cl_l_' + str(cluster_number).zfill(4))):
                os.makedirs(join(path_target_dset_l_cl, 'cl_l_' + str(cluster_number).zfill(4)))
                # 'cl_l' stands for clusters large
            
            src_path = join(images_path, str(sample_l).zfill(4) + '.png')
            dst_path = join(path_target_dset_l_cl, 'cl_l_' + str(cluster_number).zfill(4), str(sample_l).zfill(4) + '.png')
            
            ## moving files to the new clusterred dataset
            shutil.copyfile(src_path, dst_path)
            
            #### ploting...
            #image = Image.open(src_path)
            #plt.subplot(1,nb_img, num)
            #plt.imshow(np.asarray(image))
            #num += 1
            
            ### removing processed samples
            samples_set_l.remove(sample_l)
            samples_full_set.remove(sample_l)
    if temp > 0:
        cluster_number += 1   # variable to count clusters
    plt.show()
print ("Number of larger clusters:",cluster_number)

print ("Set length after processing the large clusters: ",len(samples_set_l))
print ("Set length of the full set after processing the large clusters: ",len(samples_full_set))

cluster_number = 0    # variable to count clusters
for cluster_s in short_clusters:
    nb_img = len(cluster_s)
    num = 1
    temp = 0    # variable to count clusters
    for sample_s in cluster_s:
        if sample_s in samples_set_s:
            temp +=1    # variable to count clusters
            
            ## defining paths 
            if not os.path.exists(join(path_target_dset_s_cl, 'cl_s_' + str(cluster_number).zfill(4))):
                os.makedirs(join(path_target_dset_s_cl, 'cl_s_' + str(cluster_number).zfill(4)))
                # 'cl_l' stands for clusters large
            
            src_path = join(images_path, str(sample_s).zfill(4) + '.png')
            dst_path = join(path_target_dset_s_cl, 'cl_s_' + str(cluster_number).zfill(4), str(sample_s).zfill(4) + '.png')
            
            ## moving files to the new clusterred dataset
            shutil.copyfile(src_path, dst_path)
            
            ### ploting...
            #image = Image.open(images_path + str(sample_i).zfill(6) + '.png')
            #plt.subplot(1,nb_img, num)
            #plt.imshow(np.asarray(image))
            #num += 1.
            
            ### removing processed samples
            samples_set_s.remove(sample_s)
            #print(samples_full_set)
            #samples_full_set.remove(sample_s)
    if temp > 0:
        cluster_number += 1   # variable to count clusters
    plt.show()
print ("Number of smaller clusters:",cluster_number)


print ("Set length after processing the short clusters: ",len(samples_set_s))
print ("Set length of the full set after processing the large clusters: ",len(samples_full_set))

# first we extract the images that we need to reach the full dataset, in this case 8000
max_classes = 8000
nb_new_classes = max_classes - (len(os.listdir(path_target_dset_l_cl)) + len(os.listdir(path_target_dset_s_cl)))
print ("Checking numbers...",nb_new_classes)

# I will get the images from the unsup set
single_samples = [image_i for image_i in list(samples_full_set)[:nb_new_classes]]

# Save the images in a folder 

if not os.path.exists(path_new_classes):
    os.makedirs(path_new_classes)           

for idx in tqdm(single_samples):
    path = os.path.join(path_new_classes, str(idx).zfill(4))
    image = Image.open(images_path + str(idx).zfill(4) + '.png')
    samples_full_set.remove(idx)
    image.save(path + '.png') 
print ("Set length after processing the short clusters: ",len(samples_set_s))
print ("Set length of the full set after processing the large clusters: ",len(samples_full_set))


single_images = os.listdir(path_new_classes)
single_images.sort()

# We need the features to do the query search
# paths
features_path = './less_collisions/features_maxpool_allConv.hdf5'
# load images
samples = loading_images(features_path)
# normalize images
samples_L2 = normalizing_samples_L2(samples)

query_mtx = np.zeros([len(single_images), len(samples_L2[0])])
query_names = []
#single_images.pop(0)
for num, sample_i in enumerate(single_images):
    query = samples_L2[int(sample_i[:-4])]
    query_names.append(sample_i)
    query_mtx[num] = query
    
# compute cosine similarity
sim = np.dot(query_mtx, samples_L2.T)

# sort ranking
ranks = np.argsort(sim, axis=1)[:,::-1]


cluster_size = 8

for num, sample_i in enumerate(query_names):
    cluster_i = []
    cluster_i.append(sample_i)
    instance_num = 1
    while len(cluster_i)<cluster_size and instance_num<15:
        image_nb = ranks[num][instance_num]
        image_name = str(image_nb).zfill(4) + '.png'
        if image_nb in samples_full_set:
            cluster_i.append(image_name)
            samples_full_set.remove(image_nb)
        instance_num += 1
    
    # move the images fom the cluster_i to the new folder
    
    src_path = join(path_new_classes, cluster_i[0])
    dst_folder =  join(path_target_dset_single, cluster_i[0][:-4])

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    dst_path = join(dst_folder, cluster_i[0])
    shutil.copyfile(src_path, dst_path)
    
    for num_cl, img_i in enumerate(cluster_i[1:]):
        src_path = join(images_path, cluster_i[num_cl+1])
        dst_path = join(dst_folder, img_i)
        shutil.copyfile(src_path, dst_path)
   
print ("Set length after processing the short clusters: ",len(samples_set_s))
print ("Set length of the full set after processing the large clusters: ",len(samples_full_set))

# after that i should do the same with the other clusters with 2 images each...
print ('Similarity shape:',sim.shape)

#print (np.amax(sim))
#print (np.amin(sim))
#sim_ord = np.argsort(sim)


...
# finally we move all the single classes to a folder inside the "dataset00?" folder
list_files = os.listdir(path_new_classes)
for sample_i in tqdm(list_files):
    src_path = join(path_new_classes, sample_i)
    dst_folder =  join(path_target_dset, sample_i[:-4])
    
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    dst_path = join(dst_folder, sample_i)
    shutil.copyfile(src_path, dst_path)
    
    
    
