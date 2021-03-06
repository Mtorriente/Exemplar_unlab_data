import os
import numpy as np
from PIL import Image
from skimage import color
from skimage.filters import sobel
from tqdm import tqdm
################################################
# Crea una carpeta con los contornos de todas las imagenes de entrada
################################################
out_path = './surrogate_dataset/edge_masks/'
in_path = './surrogate_dataset/classes/'

classes = os.listdir(in_path)
classes.sort()

try:
    os.mkdir(out_path)
except:
    pass

for img in tqdm(classes):
    image = np.asarray(Image.open(in_path + img))
    edge_mask = sobel(color.rgb2gray(image))
    im = Image.fromarray(np.uint8(edge_mask*255))
    im.save(out_path + img[:-4] + '_edge.png')
