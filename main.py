### YOUR CODE HERE
import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
from ImageUtils import visualize
import tensorflow as tf


def configure():
	parser = argparse.ArgumentParser()
	parser.add_argument("mode",help="train, test or predict")
	parser.add_argument("--save_dir", help="path to save the model and dataset")
	parser.add_argument("--data_dir", help="path to data")
	args = parser.parse_args()
	return args


def main(config):
	model_configs['save_dir'] = config.save_dir
	model = MyModel(model_configs).cuda()
	data_dir = os.path.join(config.data_dir,"cifar-10-batches-py")
	if config.mode == 'train':
		# Training and validation on public training dataset
		x_train, y_train, x_test, y_test = load_data(data_dir)
		x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)
		model.model_setup(training_configs)
		model.train(x_train_new, y_train_new, training_configs,x_valid,y_valid)
		model.evaluate(x_valid, y_valid,[10, 20, 30, 40, 50, 60])

	elif config.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(data_dir)
		model.evaluate(x_test, y_test,[60])

	elif config.mode == 'predict':
		#Predicting on private testing dataset
		path = os.path.join(config.data_dir,'private_test_images_2024.npy')
		# Loading private testing dataset
		x_unseen_test = load_testing_images(path)
		# visualizing the first testing image to check your image shape
		visualize(x_unseen_test[0], 'test.png')
		# Predicting and storing results on private testing dataset
		predictions = model.predict_prob(x_unseen_test,60)
		print("Shape of prediction:", predictions.shape, predictions)
		np.save('../predictions.npy', predictions)

if __name__ == '__main__':
	print(torch.cuda.is_available())
	os.environ['CUDA_VISIBLE_DEVICES'] = '9'
	config = configure()
	main(config)

# ### END CODE HERE


