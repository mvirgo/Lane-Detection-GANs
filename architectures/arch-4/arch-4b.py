"""
This architecture uses Inception as a feature extractor,
followed by the decoder in my previous architecture from
https://github.com/mvirgo/MLND-Capstone.
To match Inception with the decoder, ZeroPadding is used to
match the desired 10x10 shape. Inception is fully re-trained,
but uses ImageNet pre-trained weights as a starting point.
Uses a 299x299x3 input to a 112x112x1 output.
"""

# Import necessary items from Keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, ZeroPadding2D

def get_model(input_shape, final_activation):

	# Set pooling size
	pool_size = (2, 2)

	# Using Inception with ImageNet pre-trained weights
	inception = InceptionV3(weights='imagenet')

	# Get rid of final two layers - global average pooling, FC
	for i in range(2):
		inception.layers.pop()

	# Grab input and output in order to make a new model
	inp = inception.input
	out = inception.layers[-1].output

	# Output above should be 8x8
	# Use zero padding to get up to a 10x10 like previous architectures
	x = ZeroPadding2D(padding=((1, 1),(1, 1)))(out)

	# Upsample 1
	x = UpSampling2D(size=(2,2))(x)

	# Deconv 1
	x = Conv2DTranspose(128, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1')(x)
	x = Dropout(0.2)(x)

	# Deconv 2
	x = Conv2DTranspose(128, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2')(x)
	x = Dropout(0.2)(x)

	# Upsample 2
	x = UpSampling2D(size=pool_size)(x)

	# Deconv 3
	x = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3')(x)
	x = Dropout(0.2)(x)

	# Deconv 4
	x = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4')(x)
	x = Dropout(0.2)(x)

	# Deconv 5
	x = Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5')(x)
	x = Dropout(0.2)(x)

	# Upsample 3
	x = UpSampling2D(size=pool_size)(x)

	# Deconv 6
	x = Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6')(x)

	# Final layer - only including one channel so 1 filter
	predictions = Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = final_activation, name = 'Final')(x)

	# Create model
	model = Model(inputs=inp, outputs=predictions)

	return model