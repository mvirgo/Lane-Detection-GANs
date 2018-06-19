import argparse
import numpy as np
import time
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.io_utils import HDF5Matrix
from keras.models import load_model
from train_net import get_data_size

def get_args():
	'''Gets the arguments from the command line'''

	parser = argparse.ArgumentParser("Obtain location of network model and (optional) ImageNet model name")

	# -- Create the descriptions for the commands
	a_desc = "The location of the trained .h5 model file, e.g. 'architectures/arch-2/arch-2a.h5'"
	m_desc = "The ImageNet pre-trained model used as encoder"

	# -- Create the arguments
	parser.add_argument("-a", help = a_desc)
	parser.add_argument("-m", help = m_desc, default = "")
	args = parser.parse_args()

	return args


def check_performance(args):
	# Load Keras model
	model = load_model(args.a)

	data_size = get_data_size(args)

	# Load test dataset
	images = HDF5Matrix('challenge_pics'+data_size+'.h5', 'images')
	labels = HDF5Matrix('challenge_lane_labels112.h5', 'labels')

	# Evaluate speed
	times = []
	# Perform pre-processing if ImageNet
	if args.m.lower() == "resnet" or args.m.lower() == "inception":
		for i in range(images.shape[0]):
			img = images[i][None,:,:,:]
			img = np.array(img, dtype=float)

			start_time = time.time()
			img = preprocess_input(img)
			# Make prediction with neural network
			prediction = model.predict(img)
			times.append(time.time() - start_time)
	else:
		for i in range(images.shape[0]):
			img = images[i][None,:,:,:]
			
			start_time = time.time()
			# Make prediction with neural network
			prediction = model.predict(img)
			times.append(time.time() - start_time)

	# Speed in ms
	speed = (sum(times)/len(times))*1000

	# Evaluate accuracy
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	if args.m.lower() == "resnet" or args.m.lower() == "inception":
		score, acc = model.evaluate(preprocess_input(np.array(images, dtype=float)), labels)
	else:
		score, acc = model.evaluate(images, labels)

	print("{0:.2f}".format(speed), "ms average inference")
	print('{:.2%}'.format(acc), "accuracy")
	print(model.count_params(), "model parameters")


def main():
	'''
	Pull in args and check speed and accuracy of trained network.
	'''
	args = get_args()
	check_performance(args)


if __name__ == "__main__":
	main()