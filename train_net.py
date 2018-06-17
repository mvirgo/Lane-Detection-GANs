import argparse
import importlib
import logging
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.io_utils import HDF5Matrix

# Local imports
from adj_preprocess import *

def get_args():
	'''Gets the arguments from the command line'''

	parser = argparse.ArgumentParser("Obtain location of network model and (optional) training details")

	# -- Create the descriptions for the commands
	a_desc = "The location of the model to be used for training, e.g. 'architectures.arch-1.arch-1a'"
	b_desc = "The batch size for training"
	e_desc = "The max amount of epochs to train for"
	fa_desc = "The activation function used on the final layer (must be from Keras)"
	l_desc = "The loss function to train on (must be from Keras)"
	m_desc = "The ImageNet pre-trained model used as encoder"

	# -- Create the arguments
	parser.add_argument("-a", help = a_desc)
	parser.add_argument("-b", help = b_desc, default = 16)
	parser.add_argument("-e", help = e_desc, default = 50)
	parser.add_argument("-fa", help = fa_desc, default = "sigmoid")
	parser.add_argument("-l", help = l_desc, default = "binary_crossentropy")
	parser.add_argument("-m", help = m_desc, default = "")
	args = parser.parse_args()

	return args


def get_data_size(args):
	'''
	Determines the pre-made input data size to pull from.
	Supports Resnet & Inception architectures, as well as
	default 112x112.
	'''
	
	if args.m != "":
		if args.m.lower() == "resnet":
			return "224"
		elif args.m.lower() == "inception":
			return "299"
		else:
			print("Invalid entry. Use 'm_desc' for help.")
	else:
		return "112"


def get_generators(args):
	'''
	Determines whether pre-processing is needed based on model type,
	and creates generators for training and validation data.
	Channel shifts are used to help with shadows.
	'''

	if args.m.lower() == "resnet":
		datagen = ImageDataGenerator(channel_shift_range=0.2, preprocessing_function=resnet_preprocess_input)
		val_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess_input)
	elif args.m.lower() == "inception":
		datagen = ImageDataGenerator(channel_shift_range=0.2, preprocessing_function=inception_preprocess_input)
		val_datagen = ImageDataGenerator(preprocessing_function=inception_preprocess_input)
	else:
		datagen = ImageDataGenerator(channel_shift_range=0.2)
		val_datagen = ImageDataGenerator()

	return datagen, val_datagen


def train_model(args, data_size):
	'''
	Loads in training data and trains on the given neural network architecture.
	Also, saves the trained model each time the model improves.
	'''

	# Load training and validation data
	print("Loading training data.")
	X_train = HDF5Matrix('data'+data_size+'.h5', 'images')
	X_val = HDF5Matrix('challenge_pics'+data_size+'.h5', 'images')
	# Load image labels
	y_train = HDF5Matrix('labels112.h5', 'labels', normalizer=label_normalizer)
	y_val = HDF5Matrix('challenge_lane_labels112.h5', 'labels', normalizer=label_normalizer)
	print("Training data loaded.")
	# Get input_shape to feed into model
	input_shape = X_train.shape[1:]

	# Load "module" containing our neural network architecture
	m = importlib.import_module(args.a)
	# Load model from that "module"
	print("Loading model.")
	model = m.get_model(input_shape, args.fa)
	print("Model loaded.")

	# Compile model
	print("Compiling model.")
	model.compile(optimizer='Adam', loss=args.l, metrics=['accuracy'])
	print("Model compiled, training initializing.")

	# Save down only the best result
	save_path = args.a.replace('.', '/')
	checkpoint = ModelCheckpoint(filepath=save_path+'.h5', 
                                 monitor='val_loss', save_best_only=True)
	# Stop early when improvement ends
	stopper = EarlyStopping(monitor='val_acc', min_delta=0.0003, patience=5)

	# Using a generator to help the model use less data
	datagen, val_datagen = get_generators(args)

	# Train the model
	args.b = int(args.b)
	args.e = int(args.e)
	model.fit_generator(datagen.flow(X_train, y_train, batch_size=args.b),
                    	steps_per_epoch=len(X_train)/args.b,
                    	epochs=args.e, verbose=1, callbacks=[checkpoint, stopper],
                    	validation_data=val_datagen.flow(X_val, y_val, batch_size=args.b),
                    	validation_steps=len(X_val)/args.b)
	
	# Show summary of model at conclusion of training
	model.summary()


def main():
	'''
	Pull in args and train a given neural network architecture
	over certain semantic segmentation data.
	'''
	args = get_args()
	logging.basicConfig(level=logging.INFO)
	data_size = get_data_size(args)
	train_model(args, data_size)


if __name__ == "__main__":
	main()
