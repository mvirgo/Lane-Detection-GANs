import numpy as np
import pickle
from keras.models import load_model
import time

# Load Keras model
model = load_model('full_CNN_model112.h5')

# Load test dataset
images = np.array(pickle.load(open("challenge_pics112.p", "rb" )))
labels = np.array(pickle.load(open("challenge_lane_labels112.p", "rb" )))

# Evaluate speed
times = []

for i in range(images.shape[0]):
    img = np.array(images[i])
    img = img[None,:,:,:]
    
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
