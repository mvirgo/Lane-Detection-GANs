"""
This architecture uses my previous architecture from
https://github.com/mvirgo/MLND-Capstone, just changed
to using a 112x112x3 input, and with a 112x112x1 output,
along with adding skip layers that jump over one
convolutional layer at a time.
Also, removes pooling layers and instead uses
2x2 strides, and adds batch normalization after each
convolutional or transpose convolutional layer.
Fairly standard shrinking convolutional layers into
expanding transpose convolutions.
"""

# Import necessary items from Keras
from keras.models import Model
from keras.layers import Add, Input, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D
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

	# Increase number of filters for above for skip layer usage with 1x1 conv
	x1 = Conv2D(32, (1, 1), padding='same', strides=(2,2), activation = 'relu', name = 'Conv1_1x1')(x)
	x1 = BatchNormalization()(x1)

	# Conv Layer 2
	x = Conv2D(32, (3, 3), padding='same', strides=(2,2), activation = 'relu', name = 'Conv2')(x)
	x = BatchNormalization()(x)

	# Conv Layer 3
	x = Conv2D(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv3')(x)
	x = BatchNormalization()(x)
	x = Add()([x, x1])
	x = Dropout(0.2)(x)

	# Increase number of filters for above for skip layer usage with 1x1 conv
	x2 = Conv2D(64, (1, 1), padding='same', strides=(2,2), activation = 'relu', name = 'Conv3_1x1')(x)
	x2 = BatchNormalization()(x2)

	# Conv Layer 4
	x = Conv2D(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv4')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)

	# Conv Layer 5
	x = Conv2D(64, (3, 3), padding='same', strides=(2,2), activation = 'relu', name = 'Conv5')(x)
	x = BatchNormalization()(x)
	x = Add()([x, x2])
	x = Dropout(0.2)(x)

	# Increase number of filters for above for skip layer usage with 1x1 conv
	x3 = Conv2D(128, (1, 1), padding='same', strides=(2,2), activation = 'relu', name = 'Conv5_1x1')(x)
	x3 = BatchNormalization()(x3)

	# Conv Layer 6
	x = Conv2D(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv6')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)

	# Conv Layer 7
	x = Conv2D(128, (3, 3), padding='same', strides=(2,2), activation = 'relu', name = 'Conv7')(x)
	x = BatchNormalization()(x)
	x = Add()([x, x3])
	x = Dropout(0.2)(x)

	# Have to upsample Conv7 to pass forward in the skip layer
	x4 = UpSampling2D(size=pool_size)(x)

	# Upsample 1
	x = UpSampling2D(size=pool_size)(x)

	# Deconv 1
	x = Conv2DTranspose(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv1')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)

	# Deconv 2
	x = Conv2DTranspose(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv2')(x)
	x = BatchNormalization()(x)
	x = Add()([x, x4])
	x = Dropout(0.2)(x)

	# Decrease number of filters for above for skip layer usage with 1x1 conv
	x5 = Conv2D(64, (1, 1), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv2_1x1')(x)
	x5 = BatchNormalization()(x5)
	x5 = UpSampling2D(size=pool_size)(x5)

	# Upsample 2
	x = UpSampling2D(size=pool_size)(x)

	# Deconv 3
	x = Conv2DTranspose(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv3')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)

	# Deconv 4
	x = Conv2DTranspose(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv4')(x)
	x = BatchNormalization()(x)
	x = Add()([x, x5])
	x = Dropout(0.2)(x)

	# Decrease number of filters for above for skip layer usage with 1x1 conv
	x6 = Conv2D(32, (1, 1), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv4_1x1')(x)
	x6 = BatchNormalization()(x6)
	x6 = UpSampling2D(size=pool_size)(x6)

	# Deconv 5
	x = Conv2DTranspose(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv5')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)

	# Upsample 3
	x = UpSampling2D(size=pool_size)(x)

	# Deconv 6
	x = Conv2DTranspose(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv6')(x)
	x = BatchNormalization()(x)
	x = Add()([x, x6])

	# Final layer - only including one channel so 1 filter
	predictions = Conv2DTranspose(1, (3, 3), padding='same', strides=(1,1), activation = final_activation, name = 'Final')(x)

	# Create model
	model = Model(inputs = inputs, outputs = predictions)

	return model
