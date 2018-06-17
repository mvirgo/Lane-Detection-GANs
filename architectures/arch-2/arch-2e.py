""" This file contains code for a fully convolutional
(i.e. contains zero fully connected layers) neural network
for detecting lanes. This version assumes the inputs
to be road images in the shape of 112 x 112 x 3 (RGB) with
the labels as 112 x 112 x 1 (just the G channel with a
re-drawn lane). Note that in order to view a returned image,
the predictions is later stacked with zero'ed R and B layers
and added back to the initial road image.
"""

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.io_utils import HDF5Matrix
from adj_preprocess import label_normalizer

# Load training images
X_train = HDF5Matrix('data112.h5', 'images')
X_val = HDF5Matrix('challenge_pics112.h5', 'images')

# Load image labels
y_train = HDF5Matrix('labels112.h5', 'labels', normalizer=label_normalizer)
y_val = HDF5Matrix('challenge_lane_labels112.h5', 'labels', normalizer=label_normalizer)

# Batch size, epochs and pool size below are all parameters to fiddle with for optimization
batch_size = 64
epochs = 50
pool_size = (2, 2)
input_shape = X_train.shape[1:]

### Here is the actual neural network ###
model = Sequential()
# Normalizes incoming inputs. First layer needs the input shape to work
model.add(BatchNormalization(input_shape=input_shape))

# Below layers were re-named for easier reading of model summary; this not necessary
# Conv Layer 1
model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))
model.add(BatchNormalization())

# Conv Layer 2
model.add(Conv2D(32, (3, 3), padding='valid', strides=(2,2), activation = 'relu', name = 'Conv2'))
model.add(BatchNormalization())

# Conv Layer 3
model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Conv Layer 4
model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Conv Layer 5
model.add(Conv2D(64, (3, 3), padding='valid', strides=(2,2), activation = 'relu', name = 'Conv5'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Conv Layer 6
model.add(Conv2D(128, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Conv Layer 7
model.add(Conv2D(128, (3, 3), padding='valid', strides=(2,2), activation = 'relu', name = 'Conv7'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Upsample 1
model.add(UpSampling2D(size=pool_size))

# Deconv 1
model.add(Conv2DTranspose(128, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Deconv 2
model.add(Conv2DTranspose(128, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Upsample 2
model.add(UpSampling2D(size=pool_size))

# Deconv 3
model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Deconv 4
model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Deconv 5
model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Upsample 3
model.add(UpSampling2D(size=pool_size))

# Deconv 6
model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))
model.add(BatchNormalization())

# Final layer - only including one channel so 1 filter
model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'sigmoid', name = 'Final'))

### End of network ###


# Using a generator to help the model use less data
# Channel shifts help with shadows slightly
datagen = ImageDataGenerator(channel_shift_range=0.2)

# Save down only the best result
checkpoint = ModelCheckpoint(filepath='arch-2.5.h5', 
                               monitor='val_loss', save_best_only=True)
# Stop early when improvement ends
stopper = EarlyStopping(monitor='val_acc', min_delta=0.0003, patience=5)

# Compiling and training the model
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)/batch_size,
                    epochs=epochs, verbose=1, callbacks=[checkpoint, stopper],
                    validation_data=(X_val, y_val))

# Show summary of model
model.summary()

