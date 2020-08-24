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
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")





# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


# This function brings date from mmddyy form to mmddyyyy form.
def processDate(date):
	mmddyy = date.split("/")
	if len(mmddyy[2]) == 2:
		mmddyy[2] = "20" + mmddyy[2]
	final_date = mmddyy[0] + "/" + mmddyy[1] + "/" + mmddyy[2]
	# print(mmddyy, final_date)
	return final_date



def convertDataset(data):
	
	data_list = list()
	
	# For each data, get the previous 3 dates (i.e. next three rows) and prepare the new dataset.
	for i in range(0, len(data) - 3):
		temp = data.iloc[i : i + 4]
		
		line = list()

		line.append(temp.loc[i + 3][' Volume'])
		line.append(temp.loc[i + 3][' Open'])
		line.append(temp.loc[i + 3][' High'])
		line.append(temp.loc[i + 3][' Low'])


		line.append(temp.loc[i + 2][' Volume'])
		line.append(temp.loc[i + 2][' Open'])
		line.append(temp.loc[i + 2][' High'])
		line.append(temp.loc[i + 2][' Low'])


		line.append(temp.loc[i + 1][' Volume'])
		line.append(temp.loc[i + 1][' Open'])
		line.append(temp.loc[i + 1][' High'])
		line.append(temp.loc[i + 1][' Low'])
		
		line.append(temp.loc[i][' Open'])
		line.append(processDate(temp.loc[i]['Date']))

		line = np.array(line)
		# print(line)

		data_list.append(line)  
	# print(temp.shape)
	data_list = np.array(data_list)
	return data_list


# Main function to prepare dataset and write train and test csv files.
def prepareDataset():
	
	data = pd.read_csv("data/q2_dataset.csv")
	# data = pd.read_csv("q2_dataset.csv")
	data_processed = convertDataset(data)

	# Shuffle the dataset.
	np.random.seed(2)
	np.random.shuffle(data_processed)

	# Split into train and test data.
	train = data_processed[:879]
	test = data_processed[879:]

	# Prepare dataframes to write into CSV Files.
	df_train = pd.DataFrame(train, columns=all_features)
	df_test = pd.DataFrame(test, columns=all_features)

	# Write into CSV Files.
	df_train.to_csv("data/train_data_RNN.csv", index=False)
	df_test.to_csv("data/test_data_RNN.csv", index=False)



# Read the input CSV files.
def readInput(fileName):

	df_inp = pd.read_csv(fileName)

	print("NULL VALUES IN DATA SET = ", df_inp.isnull().sum().sum(), sep='')

	inp = np.array(df_inp.values)

	x = inp[:, :12]
	y = inp[:, 12]

	y = y.reshape(-1, 1)

	# print(x.shape, y.shape)

	return x, y


# Normalize the data using MinMaxScaler.
def scaleData(x1, y1):
	f_scalar = MinMaxScaler(feature_range=(0, 1))
	l_scalar = MinMaxScaler(feature_range=(0, 1))

	x = f_scalar.fit_transform(x1)
	y = l_scalar.fit_transform(y1)

	x = np.reshape(x, (x.shape[0], x.shape[1], 1))
	y = y.reshape(-1, 1)

	return x, y, l_scalar


# This method creates and returns the model.
def prepareModel():

	tf.keras.backend.clear_session()
	model = Sequential()

	model.add(LSTM(units=32, input_shape=(X_train.shape[1],1)))
	model.add(Dropout(0.25))
	model.add(Dense(units=1))

	model.summary()

	model.compile(optimizer='adam', loss='mean_squared_error')

	return model



# Main method that first creates the model, then trains it on the training data and saves the model.
def trainModel():
	model = prepareModel()
	model.fit(X_train, y_train, batch_size = 20, epochs = 100)
	model.save("models/20829490_RNN_model.model")
	print("Tranining Loss:", model.evaluate(X_train, y_train))






# Main method.
if __name__ == "__main__":

	all_features = ['V-3', 'O-3', 'H-3', 'L-3', 'V-2', 'O-2', 'H-2', 'L-2', 'V-1', 'O-1', 'H-1', 'L-1', 'Actual', 'Date']
	inputs = all_features[:12]
	output = all_features[12]
	outputs = all_features[12:]

	# Uncomment this line to prepare data csv files.
	prepareDataset()


	# Read the data.
	X_train_unscaled, y_train_unscaled = readInput("data/train_data_RNN.csv")

	# Scale the data.
	X_train, y_train, scalar = scaleData(X_train_unscaled, y_train_unscaled)

	# Create, train and save the model.
	trainModel()


	# 1. load your training data

	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

	# 3. Save your model










