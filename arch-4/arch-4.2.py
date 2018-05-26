# Import necessary items from Keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, UpSampling2D, Flatten
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.io_utils import HDF5Matrix
from adj_preprocess import label_normalizer, inception_preprocess_input

# Load training images
X_train = HDF5Matrix('data299.h5', 'images')
X_val = HDF5Matrix('challenge_pics299.h5', 'images')

# Load image labels
y_train = HDF5Matrix('labels112.h5', 'labels', normalizer=label_normalizer)
y_val = HDF5Matrix('challenge_lane_labels112.h5', 'labels', normalizer=label_normalizer)

# Batch size, epochs and pool size below are all parameters to fiddle with for optimization
batch_size = 16
epochs = 50
pool_size = (2, 2)

### Here is the actual neural network ###
# Using ResNet with ImageNet pre-trained weights
inception = InceptionV3(weights='imagenet')

# Get rid of final fully-connected layer
inception.layers.pop()

# Grab input and output in order to make a new model
inp = inception.input
out = inception.layers[-1].output

# Note that now final layer of inception is `GlobalAveragePooling2D`
# Use FC layer to get back to desired size
x = Dense(5*5*64, activation = 'relu', name = 'Middle')(out)
x = Dropout(0.5)(x)

# Reshape to use in convolutional layer
x = Reshape((5, 5, 64))(x)

# Upsample 1
x = UpSampling2D(size=(4,4))(x)

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
predictions = Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final')(x)

### End of network ###


# Using a generator to help the model use less data
# Channel shifts help with shadows slightly
datagen = ImageDataGenerator(channel_shift_range=0.2, preprocessing_function=inception_preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=inception_preprocess_input)

# Save down only the best result
checkpoint = ModelCheckpoint(filepath='arch-4.2.h5', 
                               monitor='val_loss', save_best_only=True)
# Stop early when improvement ends
stopper = EarlyStopping(monitor='val_acc', min_delta=0.0003, patience=5)

# Compiling and training the model
model = Model(inputs=inp, outputs=predictions)
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)/batch_size,
                    epochs=epochs, verbose=1, callbacks=[checkpoint, stopper],
                    validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
                    validation_steps=len(X_val)/batch_size)

# Show summary of model
model.summary()
