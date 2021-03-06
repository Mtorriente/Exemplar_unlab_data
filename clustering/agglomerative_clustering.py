#######################################################
# Se cargan las caracteristicas de cada imagen anterioremente calculadas
# Se agrupan los hojas del arbol teniendo en cuenta el umbral
########################################################
from __future__ import print_function
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering_dist
from sklearn.decomposition import PCA

import time
import os
from utils_clust import normalizing_samples_L2, loading_images, searching_similar_images
import csv

# To save pairs of children...
iteration_nb = '001'
path_pairs_out = './less_collisions/image_pairs' + iteration_nb
#path_pairs_out_int_nodes = './image_int_nodes' + iteration_nb
path_pairs_out_larger_clusters = './less_collisions/image_pairs_mixed_nodes' + iteration_nb

# Define the paths
features_path = './less_collisions/features_maxpool_allConv.hdf5'
image_names_path = './less_collisions/features_maxpool_allConv.txt'

# source images
images_path = '../surrogate_dataset/unlab_set/'

crimefile = open(image_names_path, 'r')
reader = csv.reader(crimefile)
allRows = [row for row in reader]

# load images #hdf5
samples = loading_images(features_path)
# normalize images
samples_L2 = normalizing_samples_L2(samples)
# Una feature por imagen del conjunto de datos unlab_set

# defining and fitting the PCA
pca_red = PCA(n_components=9, whiten = True)
pca_red.fit(samples_L2)

# "reducing" the samples
samples_L2_pca = pca_red.transform(samples_L2)
# normalizing the samples again
samples_L2_pca_L2 = normalizing_samples_L2(samples_L2_pca)

# agglomerative clustering from scikit learn - 50000 SAMPLES 
linkage='complete' # ward, complete or average
clustering = AgglomerativeClustering_dist(linkage=linkage, n_clusters=1, compute_full_tree = True, return_distance = True)
t0 = time.time()
clustering.fit(samples_L2_pca_L2)
print("Clustering time with %s and %d samples: %.2fs" % (linkage, len(samples_L2_pca_L2), time.time() - t0))
children = clustering.children_
distances = clustering.distances
# The leaves correspond to the samples introduced
leaves = clustering.n_leaves_
#plt.plot(distances) 
#plt.show()

# visualization of what we save in the file
# El umbral se establece para dejar de agrupar imagenes que tienen una gran distancia.
# En este caso que no vemos un umbral claro que elgir, tomamos un punto intermedio
threshold = 1.25
start = 0 #1100#880
nb_im = len(children)#200

sub_child_int = []
sub_child_mixed = []
num =1
for idx, child in enumerate(children[start:start+nb_im]):
    #print (distances[start+idx], end = ' * ')
    if (child<leaves).all() and distances[start+idx]<threshold:
        # if both children from a particular node are leaves and 
        # the associated distance between them is less than thereshold 
        # the images associated to those leaves are clustered.
        #print (distances[start+idx], end = ' * ')
        '''
        image = Image.open(images_path + str(child[0]).zfill(6) + '.png')
        plt.subplot(nb_im,2, num)
        num += 1
        plt.imshow(np.asarray(image))
        plt.axis('off')
        image = Image.open(images_path + str(child[1]).zfill(6) + '.png')
        plt.subplot(nb_im,2, num)
        num += 1
        plt.imshow(np.asarray(image))
        plt.axis('off')
        '''
        num += 1
        num += 1
        
    elif (child>=leaves).all() and distances[start+idx]<threshold:
        # if both children from a particular node are internal nodes
        sub_child_int.append(child)

        
    elif (child>=leaves).any() and distances[start+idx]<threshold:
        # if one children is a node and the other is a leave
        sub_child_mixed.append(child)

print("Nodes requested: ",nb_im)
print("Nodes showed: ", (num-1)/2)
#plt.tight_layout()
print('Number of children made by internal nodes:',len(sub_child_int))
print('Number of children made by mixed node and leave:',len(sub_child_mixed))


larger_clusters = []
num_3 = 0
num_4 = 0
num_5 = 0
num_6 = 0
num_7 = 0

for child in sub_child_mixed:
    plot_leaf = []
    #plt.figure()
    plot_leaf.append(min(child)) #Se queda con la hoja del hijo
    
    new_idx = max(child)-leaves #Nuevo indice para tratar los nodos internos
    if (children[new_idx]<leaves).all():
        plot_leaf.append(children[new_idx][0])
        plot_leaf.append(children[new_idx][1])
        
    elif(children[new_idx]<leaves).any():
        plot_leaf.append(min(children[new_idx]))
        new_idx = max(children[new_idx])-leaves
        
        if (children[new_idx]<leaves).all():
            plot_leaf.append(children[new_idx][0])
            plot_leaf.append(children[new_idx][1])
            
        elif(children[new_idx]<leaves).any():
            plot_leaf.append(min(children[new_idx]))
            #print("####### Another internal node ######")
            
            new_idx = max(children[new_idx])-leaves

            if (children[new_idx]<leaves).all():
                plot_leaf.append(children[new_idx][0])
                plot_leaf.append(children[new_idx][1])

            elif(children[new_idx]<leaves).any():
                plot_leaf.append(min(children[new_idx]))
                #print("####### Another internal node ######")
                
                new_idx = max(children[new_idx])-leaves

                if (children[new_idx]<leaves).all():
                    plot_leaf.append(children[new_idx][0])
                    plot_leaf.append(children[new_idx][1])
                    
                elif(children[new_idx]<leaves).any():
                    plot_leaf.append(min(children[new_idx]))
                    #print("####### Another internal node ######")
                    
                    new_idx = max(children[new_idx])-leaves

                    if (children[new_idx]<leaves).all():
                        plot_leaf.append(children[new_idx][0])
                        plot_leaf.append(children[new_idx][1])

                    elif(children[new_idx]<leaves).any():
                        plot_leaf.append(min(children[new_idx]))
                        print("####### Another internal node ######")
    
    nb_img = len(plot_leaf)
    num = 1
    
    if len(plot_leaf) == 3:
        num_3 += 1
    
    if len(plot_leaf) == 4:
        num_4 += 1
    
    if len(plot_leaf) == 5:
        num_5 += 1

    if len(plot_leaf) == 6:
        num_6 += 1
        
    if len(plot_leaf) == 7:
        num_7 += 1
    for img in plot_leaf:
        name = ''.join(allRows[img])
        image = Image.open(images_path + name)
        #plt.subplot(1,nb_img, num)
        #plt.imshow(np.asarray(image))
        num += 1
    larger_clusters.append(plot_leaf)
#plt.show()


larger_clusters_2 = []
num_3 = 0
num_4 = 0
num_5 = 0
num_6 = 0
num_7 = 0

for child in sub_child_int:
    plot_leaf = []
#    plt.figure()

    new_idx = min(child)-leaves
    if (children[new_idx]<leaves).all():
        plot_leaf.append(children[new_idx][0])
        plot_leaf.append(children[new_idx][1])
        
    elif(children[new_idx]<leaves).any():
        plot_leaf.append(min(children[new_idx]))
        new_idx = max(children[new_idx])-leaves
        
        if (children[new_idx]<leaves).all():
            plot_leaf.append(children[new_idx][0])
            plot_leaf.append(children[new_idx][1])
            
        elif(children[new_idx]<leaves).any():
            plot_leaf.append(min(children[new_idx]))
            #print("####### Another internal node ######")
            
            new_idx = max(children[new_idx])-leaves

            if (children[new_idx]<leaves).all():
                plot_leaf.append(children[new_idx][0])
                plot_leaf.append(children[new_idx][1])

            elif(children[new_idx]<leaves).any():
                plot_leaf.append(min(children[new_idx]))
                #print("####### Another internal node ######")
                
                new_idx = max(children[new_idx])-leaves

                if (children[new_idx]<leaves).all():
                    plot_leaf.append(children[new_idx][0])
                    plot_leaf.append(children[new_idx][1])
                    
                elif(children[new_idx]<leaves).any():
                    plot_leaf.append(min(children[new_idx]))
                    #print("####### Another internal node ######")
                    
                    new_idx = max(children[new_idx])-leaves

                    if (children[new_idx]<leaves).all():
                        plot_leaf.append(children[new_idx][0])
                        plot_leaf.append(children[new_idx][1])

                    elif(children[new_idx]<leaves).any():
                        plot_leaf.append(min(children[new_idx]))
                        print("####### Another internal node ######")
 
    new_idx = max(child)-leaves
    if (children[new_idx]<leaves).all():
        plot_leaf.append(children[new_idx][0])
        plot_leaf.append(children[new_idx][1])
        
    elif(children[new_idx]<leaves).any():
        plot_leaf.append(min(children[new_idx]))
        new_idx = max(children[new_idx])-leaves
        
        if (children[new_idx]<leaves).all():
            plot_leaf.append(children[new_idx][0])
            plot_leaf.append(children[new_idx][1])
            
        elif(children[new_idx]<leaves).any():
            plot_leaf.append(min(children[new_idx]))
            #print("####### Another internal node ######")
            
            new_idx = max(children[new_idx])-leaves

            if (children[new_idx]<leaves).all():
                plot_leaf.append(children[new_idx][0])
                plot_leaf.append(children[new_idx][1])

            elif(children[new_idx]<leaves).any():
                plot_leaf.append(min(children[new_idx]))
                #print("####### Another internal node ######")
                
                new_idx = max(children[new_idx])-leaves

                if (children[new_idx]<leaves).all():
                    plot_leaf.append(children[new_idx][0])
                    plot_leaf.append(children[new_idx][1])
                    
                elif(children[new_idx]<leaves).any():
                    plot_leaf.append(min(children[new_idx]))
                    #print("####### Another internal node ######")
                    
                    new_idx = max(children[new_idx])-leaves

                    if (children[new_idx]<leaves).all():
                        plot_leaf.append(children[new_idx][0])
                        plot_leaf.append(children[new_idx][1])

                    elif(children[new_idx]<leaves).any():
                        plot_leaf.append(min(children[new_idx]))
                        print("####### Another internal node ######")
    
    nb_img = len(plot_leaf)
    num = 1
    
    if nb_img == 3:
        num_3 += 1
    
    if nb_img == 4:
        num_4 += 1
    
    if nb_img == 5:
        num_5 += 1

    if nb_img == 6:
        num_6 += 1
        
    if nb_img == 7:
        num_7 += 1

    for img in plot_leaf:
        name = ''.join(allRows[img])
        image = Image.open(images_path + name)
#        plt.subplot(1,nb_img, num)
#        plt.imshow(np.asarray(image))
        num += 1
    larger_clusters_2.append(plot_leaf)
#plt.show()

num_3, num_4, num_5, num_6, num_7

# writing into a matrix only the node names we want
threshold = 1.25
num =1
child_list = []

for idx, child in enumerate(children):
    if (child<leaves).all() and distances[idx]<threshold:
        # if both children from a particular node are leaves and 
        # the associated distance between them is less than thereshold 
        # the images associated to those leaves are clustered.
        child_list.append(child)

child_array = np.vstack(child_list)
np.save(path_pairs_out + '.npy', child_array)

### REMEMBER TO BUILD TEH CLUSTERS AGAIN WITH ALL THE CHILDRE!!!! ALL OF THEM!!!
if len(sub_child_mixed) > 0:
    #sub_child_mixed_array = np.vstack(sub_child_mixed)
    np.save(path_pairs_out_larger_clusters + '.npy', larger_clusters+larger_clusters_2)