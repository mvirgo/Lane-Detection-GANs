"""
This architecture uses my previous architecture from
https://github.com/mvirgo/MLND-Capstone, just changed
to using a 112x112x3 input, and with a 112x112x1 output,
along with adding skip layers that jump from similar
layers on the encoder to decoder sides.
Also, adds batch normalization after each
convolutional or transpose convolutional layer.
Fairly standard shrinking convolutional layers into
expanding transpose convolutions.
"""

# Import necessary items from Keras
from keras.models import Model
from keras.layers import Add, Input, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def get_model(input_shape, final_activation):

	# Set pooling size
	pool_size = (2, 2)

	inputs = Input(shape=input_shape)
	# Normalizes incoming inputs. First layer needs the input shape to work
	x = BatchNormalization()(inputs)

	# Below layers were re-named for easier reading of model summary; this not necessary
	# Conv Layer 1
	x = Conv2D(16, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv1')(x)
	x = BatchNormalization()(x)

	# Conv Layer 2
	x = Conv2D(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv2')(x)
	x = BatchNormalization()(x)

	# Pooling 1
	x1 = MaxPooling2D(pool_size=pool_size)(x)

	# Conv Layer 3
	x1 = Conv2D(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv3')(x1)
	x1 = BatchNormalization()(x1)
	x1 = Dropout(0.2)(x1)

	# Conv Layer 4
	x2 = Conv2D(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv4')(x1)
	x2 = BatchNormalization()(x2)
	x2 = Dropout(0.2)(x2)

	# Conv Layer 5
	x3 = Conv2D(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv5')(x2)
	x3 = BatchNormalization()(x3)
	x3 = Dropout(0.2)(x3)

	# Pooling 2
	x4 = MaxPooling2D(pool_size=pool_size)(x3)

	# Conv Layer 6
	x4 = Conv2D(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv6')(x4)
	x4 = BatchNormalization()(x4)
	x4 = Dropout(0.2)(x4)

	# Conv Layer 7
	x5 = Conv2D(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv7')(x4)
	x5 = BatchNormalization()(x5)
	x5 = Dropout(0.2)(x5)

	# Pooling 3
	x6 = MaxPooling2D(pool_size=pool_size)(x5)

	# Upsample 1
	x6 = UpSampling2D(size=pool_size)(x6)

	# Deconv 1
	x6 = Conv2DTranspose(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv1')(x6)
	x6 = BatchNormalization()(x6)
	x6 = Add()([x6, x5])
	x6 = Dropout(0.2)(x6)

	# Deconv 2
	x6 = Conv2DTranspose(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv2')(x6)
	x6 = BatchNormalization()(x6)
	x6 = Add()([x6, x4])
	x6 = Dropout(0.2)(x6)

	# Upsample 2
	x6 = UpSampling2D(size=pool_size)(x6)

	# Deconv 3
	x6 = Conv2DTranspose(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv3')(x6)
	x6 = BatchNormalization()(x6)
	x6 = Add()([x6, x3])
	x6 = Dropout(0.2)(x6)

	# Deconv 4
	x6 = Conv2DTranspose(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv4')(x6)
	x6 = BatchNormalization()(x6)
	x6 = Add()([x6, x2])
	x6 = Dropout(0.2)(x6)

	# Deconv 5
	x6 = Conv2DTranspose(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv5')(x6)
	x6 = BatchNormalization()(x6)
	x6 = Add()([x6, x1])
	x6 = Dropout(0.2)(x6)

	# Upsample 3
	x6 = UpSampling2D(size=pool_size)(x6)

	# Deconv 6
	x6 = Conv2DTranspose(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv6')(x6)
	x6 = BatchNormalization()(x6)
	x6 = Add()([x6, x])

	# Final layer - only including one channel so 1 filter
	predictions = Conv2DTranspose(1, (3, 3), padding='same', strides=(1,1), activation = final_activation, name = 'Final')(x6)

	# Create model
	model = Model(inputs = inputs, outputs = predictions)

	return model
