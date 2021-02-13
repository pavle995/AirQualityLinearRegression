import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from datetime import date
from sklearn.linear_model import LinearRegression
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split
import pandas as pd


def replace_condition(data_frame):
    unique_cond = set()
    for index, row in data_frame.iterrows():
        unique_cond.add(row['Conditions'])

    cond_dict = {}
    i=0
    for cond in unique_cond:
        cond_dict[cond] = i
        i+=1
    for index, row in data_frame.iterrows():
        data_frame.at[index, 'Conditions'] = cond_dict[row['Conditions']]
    return data_frame

def replace_date(data_frame):
	for index, row in data_frame.iterrows():
		date_mdY = row['Date time'].split('/')
		date_mdY = [int(i) for i in date_mdY]
		data_frame.at[index, 'Date time'] = date(date_mdY[2], date_mdY[0], date_mdY[1])

	return data_frame

def replace_date_Ymd(data_frame):
    for index, row in data_frame.iterrows():
        date_Ymd = row['date'].split('/')
        date_Ymd = [int(i) for i in date_Ymd]
        data_frame.at[index, 'date'] = date(date_Ymd[0], date_Ymd[1], date_Ymd[2])

    return data_frame


def replace_y_date(data):
	for row in data:
		date_Ymd = row[0].split('/')
		date_Ymd = [int(i) for i in date_Ymd]
		row[0] = date(date_Ymd[0], date_Ymd[1], date_Ymd[2])
	return data

def convert_2_np2d(data):
	data_2d = np.empty([0, len(data[0])-1], np.float32)
	i = 0
	for row in data:
		np_array = np.array([row[1:]], np.float32)
		data_2d = np.append(data_2d, np_array, axis=0)
	return data_2d

def handle_empty_data(data_frame):
	for index, row in data_frame.iterrows():
		if pd.isna(row['Wind Chill']):
			data_frame.at[index, 'Wind Chill'] = 0
		if pd.isna(row['Heat Index']):
			data_frame.at[index, 'Heat Index'] = 0
		if pd.isna(row['Wind Gust']):
			data_frame.at[index, 'Wind Gust'] = 0
		if pd.isna(row['Cloud Cover']):
			data_frame.at[index, 'Cloud Cover'] = 0
	data_frame = data_frame.drop('Snow Depth', 1)
	return data_frame

def feature_normaliaztion(data):
	min = [None] * 8
	max = [None] * 8
	data = np.delete(data,7,1)
	data = np.delete(data,5,1)
	data = np.delete(data,4,1)
	for i in range(0, 8):
		min[i] = np.amin(data[:,i])
		max[i] = np.amax(data[:,i])

	for row in data:
		for i in range(0, 8):
			row[i] = (row[i] - min[i])/(max[i] - min[i])

	return data

def generate_dataset():
    np.set_printoptions(threshold=sys.maxsize)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    #pd.set_option("display.max_columns", None)
    data_frame = pd.read_csv('weather_data.csv', sep=',')
    data_frame = replace_condition(data_frame)
    data_frame = replace_date(data_frame)
    data_frame = data_frame.sort_values(by=['Date time'])
    data_frame = handle_empty_data(data_frame)
    data_frame = data_frame.drop('Date time', 1)
    data_np = data_frame.to_numpy(dtype='float32')
#############################
    y_data = list()
    with open('y_data.csv', newline='') as csvfile:
        weather_data_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in weather_data_reader:
        	y_data.append(row)
    y_data_frame = pd.read_csv('y_data.csv', sep=',')
    y_data_frame = replace_date_Ymd(y_data_frame)
    y_data_frame = y_data_frame.sort_values(by=['date'])
    y_data_frame = y_data_frame.drop(['date', ' pm10', ' o3', ' no2', ' so2', ' co'], axis=1)
    y_np = y_data_frame.to_numpy(dtype='float32')
    
    return data_np, y_np

def run():
	X, y = generate_dataset()
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.1)

	model = LinearRegression().fit(X_train, y_train)
	#r_sq = model.score(X_train, y_train)
	y_pred = model.predict(X_test)
	print(y_pred)
	print(y_test)


if __name__ == "__main__":
	run()