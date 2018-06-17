import numpy as np
import time
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix


# Load Keras model
model = load_model('arch-1a.h5')

# Load test dataset
images = HDF5Matrix('challenge_pics112.h5', 'images')
labels = HDF5Matrix('challenge_lane_labels112.h5', 'labels')

# Evaluate speed
times = []

for i in range(images.shape[0]):
    img = images[i][None,:,:,:]
    
    start_time = time.time()
    # Make prediction with neural network)
    prediction = model.predict(img)
    times.append(time.time() - start_time)

# Speed in ms
speed = (sum(times)/len(times))*1000

# Evaluate accuracy
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
score, acc = model.evaluate(images, labels)

print("{0:.2f}".format(speed), "ms average inference")
print('{:.2%}'.format(acc), "accuracy")
print(model.count_params(), "model parameters")
