import numpy as np 
import matplotlib.pyplot as plt 
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

"""
THIS IS A BINARY IMAGE CLASSIFIER INSPIRED BY SILICON VALLEY'S HOTDOG - NOT HOTDOG APP

It can be trained to recognise whether a desired object is present or not in a labelled dataset of images

Some parts (importing the dataset) have been left out, feel free to change it up a bit to suit your dataset

However I've left the imports here
"""

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

train_set_x_flatten = train_set_x_orig.reshaape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmoid(z):
	"""
	Compute the sigmoid of z - a scalar or numpy array of any size
	"""
	return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
	"""
	Creates a vector of zeros of shape (dim, 1) for the weights and initializes the bias to 0
	"""
	w = np.zeros((dim, 1))
	b = 0

	return w, b


def propagate(w, b, X, Y):
	"""
	Implements forward propagation
	"""

	m = X.shape[1]

	A = sigmoid(np.dot(w.T, X) + b)
	cost = -np.sum(((Y * np.log(A)) + ((1-Y) * np.log(1-A)))) / m

	dw = np.dot(X, (A-Y).T) / m
	db = np.sum(A-Y) / m

	cost = np.squeeze(cost)

	grads = {"dw": dw,
			 "db": db}

	return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
	"""
	This optimizes w and b by running a gradient descent algorithm (backpropagation)
	"""

	costs = []

	for i in range(num_iterations):

		grads, cost = propagate(w, b, X, Y)


		dw = grads["dw"]
		db = grads["db"]

		w = w - learning_rate * dw
		b = b - learning_rate * db

		if i % 100 == 0:
			costs.append(cost)

		if print_cost and i % 100 == 0:
			print ("Cost after iteration {}: {}".format(i, cost))


	params = {"w": w,
			  "b": b}

	grads = {"dw": dw,
			 "db": db}

	return params, grads, costs


def predict(w, b, X):
	"""
	Predict whether the label is 0 or 1 using leared logistic regression parameters (w, b)
	"""

	m = X.shape[1]
	Y_prediction = np.zeros((1, m))
	w = w.reshape(X.shape[0], 1)

	# Compute the vector "A" predicting the probabilities of something being present in the image
	A = sigmoid(np.dot(w.T, X) + b)

	for i in range(A.shape[1]):
		
		pred = A[0][i]
		if pred > 0.5:
			Y_prediction[0][i] = 1
		else:
			Y_prediction[0][i] = 0

	return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
	"""
	Builds the logistic regression model using the functions above
	"""

	w, b = initialize_with_zeros(X_train.shape[0])

    	parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    	w = parameters["w"]
    	b = parameters["b"]
    
    	Y_prediction_test = predict(w, b, X_test)
    	Y_prediction_train = predict(w, b, X_train)

    	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


    	d = {"costs": costs,
		"Y_prediction_test": Y_prediction_test, 
		"Y_prediction_train" : Y_prediction_train, 
		"w" : w, 
	 	"b" : b,
		"learning_rate" : learning_rate,
	 	"num_iterations": num_iterations}

    	return d

