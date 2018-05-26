'''
Adapted from nbviewer.jupyter.org/gist/embanner/6149bba89c174af3bfd69537b72bca74

Used to convert 3d array from ImageDataGenerator to 4D for pre-processing
for ImageNet pre-trained models.
'''

import numpy as np

# Currently using ResNet & Inception networks
from keras.applications.inception_v3 import preprocess_input as pre_inception
from keras.applications.resnet50 import preprocess_input as pre_resnet

def label_normalizer(x):
    x = np.array(x, dtype=np.float64)
    x /= 255.
    return x

def inception_preprocess_input(x):
    x = np.expand_dims(x, axis=0)
    x = pre_inception(x)
    return x[0]

def resnet_preprocess_input(x):
    x = np.expand_dims(x, axis=0)
    x = pre_resnet(x)
    return x[0]
