# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:13:38 2020

@author: Truong
"""

import os
import gdown

from keras.models import  Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
import zipfile

def loadModel():
	
	num_classes = 7
	
	model = Sequential()

# 1
	model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
	model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

# 2
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

# 3
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	model.add(Flatten())


	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(num_classes, activation='softmax'))
	


	if os.path.isfile('facial_expression_model_weights.h5') != True:
		print("facial_expression_model_weights.h5 will be downloaded...")
		

		url = 'https://drive.google.com/uc?id=13iUHHP3SlNg53qSuQZDdHDSDNdBP9nwy'
		output = 'facial_expression_model_weights.zip'
		gdown.download(url, output, quiet=False)
		

		with zipfile.ZipFile(output, 'r') as zip_ref:
			zip_ref.extractall()
		
	model.load_weights('facial_expression_model_weights.h5')
	
	return model
	
	
	return 0