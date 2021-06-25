# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:38 2021

@author: alaaf
"""
import keras  
import numpy as np 
import pandas as pd
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.datasets import mnist
from keras.models import load_model
from sklearn.model_selection import KFold
from matplotlib import image
# example of cropping an image
from PIL import Image
# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# load dataset train set and test set
(X_train ,Y_train) , (X_test , Y_test) = mnist.load_data()
# summarize loaded dataset
print ("X_train Shape = " , X_train.shape)
print ("Y_train Shape = " , Y_train.shape)
print ("X_test Shape = " , X_test.shape)
print ("Y_test Shape = " , Y_test.shape)

# preprossing 
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))
# convert to one-hot-encoding , is a representation of categorical variables as binary vectors
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
# plot first few images
numberofImages = 9 
for i in range(numberofImages):
	plt.subplot(330 + 1 + i)
	plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
plt.show()

def nomalization(X_trainset ,X_testset):
    newtrain = X_trainset.astype('float32')
    newtest = X_testset.astype('float32')
    newtrain = newtrain / 255.0
    newtest = newtest / 255.0
	# return images after normalization step.
    return newtrain, newtest

new_X_train,new_X_test=nomalization(X_train ,X_test)


def CNNMODEL():
    cnnmodel = Sequential()
    cnnmodel.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
   	
    cnnmodel.add(MaxPooling2D((2, 2)))
    cnnmodel.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    cnnmodel.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	 
    cnnmodel.add(MaxPooling2D((2, 2)))
    cnnmodel.add(Flatten())
    cnnmodel.add(Dense(100,activation=('relu'),kernel_initializer='he_uniform'))
    cnnmodel.add(Dense(10, activation='softmax'))
	# compile model
    opt = SGD(lr=0.01, momentum=0.9)
    cnnmodel.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return cnnmodel

def crossValidation(X, Y,numberoffolds ):
    s = KFold(numberoffolds, shuffle=True, random_state=1)
    accuracy_, fitting_ = list(), list()
    
    for indexofX_train,indexofX_test in s.split(X):
        cnnModel = CNNMODEL()
        X_train , Y_train , X_test,Y_test = X[indexofX_train],Y[indexofX_train],X[indexofX_test],Y[indexofX_test]
        modelfit = cnnModel.fit(X_train,Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test), verbose=0)
        _,accuracy= cnnModel.evaluate(X_test, Y_test, verbose=0)
        print('Accuracy %.3f' % (accuracy * 100))
        accuracy_.append(accuracy)
        fitting_.append(modelfit)
        
    return accuracy_,fitting_

numberoffolds=5

acc , fit = crossValidation(new_X_train,Y_train,numberoffolds)
print('Accuracy: mean=%.3f' % (mean(acc)*100))

# load and prepare the image
def load_image(filename):
	img = load_img(filename,grayscale=True , target_size=(28, 28))
	new_img = img_to_array(img)
	new_img = new_img.reshape(1,28, 28, 1)
	new_img = new_img.astype('float32')
	new_img = new_img / 255.0
	return new_img

image = load_image('E:\FCAI\Year4\Machine learning\image.png')
model = CNNMODEL() 
digit = model.predict_classes(image)
print(digit[0])
    
