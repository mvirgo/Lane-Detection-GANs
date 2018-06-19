'''	
Adapted from nbviewer.jupyter.org/gist/embanner/6149bba89c174af3bfd69537b72bca74	
	
Used to convert 3d array from ImageDataGenerator to 4D for pre-processing	
for ImageNet pre-trained models.	
'''

import numpy as np
from keras.applications.imagenet_utils import preprocess_input

def adj_preprocess_input(x):	
	x = np.expand_dims(x, axis=0)	
	x = preprocess_input(x)	
	return x[0]