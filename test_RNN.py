# import required packages

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import numpy as np


from sklearn.preprocessing import MinMaxScaler

import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


from train_RNN import readInput, scaleData


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


def plotGraph():
	# plt.title('Accuracy vs Epochs for ' + kind + '.')
	plt.figure(figsize=(20, 5))
	plt.xlabel('Indices in Test Data')
	plt.ylabel('Opening Prices')
	indices = [i for i in range(len(y_test_unscaled))]
	plt.plot(indices, y_test_unscaled, 'oc', label='Actual Values')
	plt.plot(indices, y_pred_unscaled, '.-r', label='Predicted Values')
	plt.legend()
	plt.savefig('data/q2plot.png')
	plt.show()


# This method calculates and returns the RMSE value.
def rmse():
	print("RMSE Value is = ", np.sqrt(np.mean(((y_pred_unscaled - y_test_unscaled)**2))))


# Main method.
if __name__ == "__main__":

	# Load the saved model.
	model = load_model("models/20829490_RNN_model.model")

	# Load the test data and normalize it.
	X_test_unscaled, y_test_unscaled = readInput("data/test_data_RNN.csv")
	X_test, y_test, scalar = scaleData(X_test_unscaled, y_test_unscaled)

	# Prediction on the scaled data.
	y_pred = model.predict(X_test)

	# Inverse scaling the output of the NN.
	y_pred_unscaled = scalar.inverse_transform(y_pred)

	# Printing the RMSE value.
	rmse()
	plotGraph()
	# 1. Load your saved model

	# 2. Load your testing data

	# 3. Run prediction on the test data and output required plot and loss