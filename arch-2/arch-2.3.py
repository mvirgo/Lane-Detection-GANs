""" This file contains code for a fully convolutional
(i.e. contains zero fully connected layers) neural network
for detecting lanes. This version assumes the inputs
to be road images in the shape of 112 x 112 x 3 (RGB) with
the labels as 112 x 112 x 1 (just the G channel with a
re-drawn lane). Note that in order to view a returned image,
the predictions is later stacked with zero'ed R and B layers
and added back to the initial road image.
"""

import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Import necessary items from Keras
from keras.models import Model
from keras.layers import Add, Input, Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Load training images
X_train = np.array(pickle.load(open("full_CNN_train112.p", "rb" )))
X_val = np.array(pickle.load(open("challenge_pics112.p", "rb" )))

# Load image labels
y_train = np.array(pickle.load(open("full_CNN_labels112.p", "rb" )))
y_val = np.array(pickle.load(open("challenge_lane_labels112.p", "rb" )))

# Normalize labels - training images get normalized to start in the network
y_train = y_train / 255
y_val = y_val / 255

# Shuffle training images along with their labels
seed_num = 123456789
X_train, y_train = shuffle(X_train, y_train, random_state=seed_num)

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 64
epochs = 50
pool_size = (2, 2)
input_shape = X_train.shape[1:]

### Here is the actual neural network ###
inputs = Input(shape=input_shape)
# Normalizes incoming inputs. First layer needs the input shape to work
x = BatchNormalization()(inputs)

# Below layers were re-named for easier reading of model summary; this not necessary
# Conv Layer 1
x = Conv2D(16, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv1')(x)

# Increase number of filters for above for skip layer usage with 1x1 conv
x1 = Conv2D(32, (1, 1), padding='same', strides=(2,2), activation = 'relu', name = 'Conv1_1x1')(x)

# Conv Layer 2
x = Conv2D(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv2')(x)

# Pooling 1
x = MaxPooling2D(pool_size=pool_size)(x)

# Conv Layer 3
x = Conv2D(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv3')(x)
x = Add()([x, x1])
x = Dropout(0.2)(x)

# Increase number of filters for above for skip layer usage with 1x1 conv
x2 = Conv2D(64, (1, 1), padding='same', strides=(1,1), activation = 'relu', name = 'Conv3_1x1')(x)

# Conv Layer 4
x = Conv2D(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv4')(x)
x = Dropout(0.2)(x)

# Conv Layer 5
x = Conv2D(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv5')(x)
x = Add()([x, x2])
x = Dropout(0.2)(x)

# Increase number of filters for above for skip layer usage with 1x1 conv
x3 = Conv2D(128, (1, 1), padding='same', strides=(2,2), activation = 'relu', name = 'Conv5_1x1')(x)

# Pooling 2
x = MaxPooling2D(pool_size=pool_size)(x)

# Conv Layer 6
x = Conv2D(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv6')(x)
x = Dropout(0.2)(x)

# Conv Layer 7
x = Conv2D(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv7')(x)
x = Add()([x, x3])
x = Dropout(0.2)(x)

# No change in size by next skip layer
x4 = x

# Pooling 3
x = MaxPooling2D(pool_size=pool_size)(x)

# Upsample 1
x = UpSampling2D(size=pool_size)(x)

# Deconv 1
x = Conv2DTranspose(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv1')(x)
x = Dropout(0.2)(x)

# Deconv 2
x = Conv2DTranspose(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv2')(x)
x = Add()([x, x4])
x = Dropout(0.2)(x)

# Decrease number of filters for above for skip layer usage with 1x1 conv
x5 = Conv2D(64, (1, 1), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv2_1x1')(x)
x5 = UpSampling2D(size=pool_size)(x5)

# Upsample 2
x = UpSampling2D(size=pool_size)(x)

# Deconv 3
x = Conv2DTranspose(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv3')(x)
x = Dropout(0.2)(x)

# Deconv 4
x = Conv2DTranspose(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv4')(x)
x = Add()([x, x5])
x = Dropout(0.2)(x)

# Decrease number of filters for above for skip layer usage with 1x1 conv
x6 = Conv2D(32, (1, 1), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv4_1x1')(x)
x6 = UpSampling2D(size=pool_size)(x6)

# Deconv 5
x = Conv2DTranspose(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv5')(x)
x = Dropout(0.2)(x)

# Upsample 3
x = UpSampling2D(size=pool_size)(x)

# Deconv 6
x = Conv2DTranspose(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Deconv6')(x)
x = Add()([x, x6])

# Final layer - only including one channel so 1 filter
predictions = Conv2DTranspose(1, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Final')(x)

### End of network ###


# Using a generator to help the model use less data
# Channel shifts help with shadows slightly
datagen = ImageDataGenerator(channel_shift_range=0.2)
datagen.fit(X_train)

# Save down only the best result
checkpoint = ModelCheckpoint(filepath='arch-2.3.h5', 
                               monitor='val_loss', save_best_only=True)
# Stop early when improvement ends
stopper = EarlyStopping(monitor='val_acc', min_delta=0.0003, patience=5)

# Compiling and training the model
model = Model(inputs = inputs, outputs = predictions)
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)/batch_size,
                    epochs=epochs, verbose=1, callbacks=[checkpoint, stopper],
                    validation_data=(X_val, y_val))

# Freeze layers since training is done
model.trainable = False
model.compile(optimizer='Adam', loss='binary_crossentropy')

# Show summary of model
model.summary()

