# Load useful library
import os
import random
import numpy as np 
import pylab as plt
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
#from celluloid import Camera # getting the camera
from tensorflow.keras import layers
#from IPython.display import HTML


def load_data():
     
    """"  Data reading """

    params = np.load("INPUT.npy")
    output = np.load("OUTPUT.npy")
    return params, output

params, output = load_data()


def splitinto_train_test(data):

     """  
      The purpose of this function is to separate our data into training and testing.
      It returns the train and test data and also their indices
     
     """
     
     index = [k for k in range (len(data))] # We collect the indexes from the videos
     train_index = random.sample(index, k = 1250) # We take randomly 1250 indexes for the video trains
     test_index = list(set (index)-set(train_index)) # the remaining 250 indexes for the test

     train = [video for video in data[train_index]] # The 1250 videos of train
     test = [video for video in data[test_index]] # The 250 videos of test 
     # Transform train and test into numpy
     train = np.array(train) 
     test = np.array(test)
     return train, test, train_index, test_index

train, test, train_index, test_index = splitinto_train_test(output)

def nb_train_test(x, y, nb_video_train, nb_video_test, nb_frame):
    x = x[:nb_video_train, :nb_frame , : , : ]
    y = y[:nb_video_test, :nb_frame , : , : ]
    return x, y

train, test = nb_train_test(train,test,50, 20, 10)
""" 
print(train.shape)
print(test.shape)
 """



def change_shape(X,Y, frame = None, width = 40, length = 40):
    
    """ changing the shape of the train """

    N1 = len(X) 
    N2=len(Y)
    X = X.reshape([N1, frame , width , length , 1]) 
    Y = Y.reshape([N2, frame , width , length, 1])
    return X, Y

train, test = change_shape(train, test,frame = 10, width = 40, length = 40)
""" print(train.shape)
print(test.shape) """

def processing_data(X):

    """ train data processing for the training of the model """

    X_= X[:,:-1,:,:,:] 
    X_shifted = X[:,1:,:,:,:] 
    return X_, X_shifted

x_train, y_train = processing_data(train)
x_test, y_test = processing_data(test)

""" for i in [x_train, y_train, x_test, y_test]:
  print(i.shape)
 """
def normalize(x,y):

    """ Normalize the value of all frames between 0 and 1 """

    for video in range(len(x)):
        for frame in range(len(x[video,:,:,:,:])):
            x[video,frame,:,:,:] = x[video,frame,:,:,:] / np.linalg.norm(x[video,frame,:,:,:])
            y[video,frame,:,:,:] = y[video,frame,:,:,:] / np.linalg.norm(y[video,frame,:,:,:])
    return x, y

#x_train, y_train = normalize (x_train, y_train)
#x_test, y_test = normalize(x_test, y_test)


def normalize_max(x,y):

  """ min-max normalize the value of all frames """

  x = x / x.max(axis=0)
  y = y / y.max(axis=0)

  return x, y 

x_train, y_train = normalize_max(x_train, y_train)
x_test, y_test = normalize_max(x_test, y_test)


# Model 



model = keras.Sequential()

input_shape = (None, 40, 40, 1)

model.add(keras.Input(shape=(None, 40, 40, 1)))

model.add(keras.layers.Conv2D(
            filters=32, kernel_size=(5,5), strides=(1,1), activation="tanh", padding="same"
        ))


model.add(keras.layers.Conv2D(
            filters= 64, kernel_size=(5,5), strides=(1,1), activation="tanh", padding="same"
        ))

model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2)))

# BGRU
model.add(
    keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True),input_shape=(20,20,64))
)

model.add(
    keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True), input_shape=(None,None,40,40, 1 ))
)

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2, 3), strides=(2,2,2)))

#BGRU
model.add(
    keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True),input_shape=(None,None,40,40, 1 ))
)
model.add(
    keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True),input_shape=(None,None,40,40, 1 ))
)

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2, 3), strides=(2,2,2)))
          
model.add(
    keras.layers.Bidirectional(keras.layers.GRU(512, return_sequences=True),input_shape=(None,None,40,40, 1 ))
)
          
model.add(
    keras.layers.Bidirectional(keras.layers.GRU(512, return_sequences=True), input_shape=(None,None,40,40, 1 ))
)

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2, 3), strides=(2,2,2)))
model.add(
    keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True),input_shape = (None,None,40,40, 1 ))
)
model.add(
    keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True),input_shape=(None,None,40,40, 1 ))
)


model.summary()
model.compile(loss="mean_squared_error", optimizer="adam")
epochs = 10
model.fit(
    x_train,
    y_train,
    batch_size=10,
    epochs=epochs,
    verbose=2,
    validation_split=0.2,
) 











    # Build a model

seq = keras.Sequential(
    [
        keras.Input(
            shape=(None, 40, 40, 1)
        ),  # Variable-length sequence of 40x40x1 frames
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.Conv3D(
            filters=1, kernel_size=(3, 3, 3), activation="tanh", padding="same"
        ),
    ]
)
seq.compile(loss="mean_squared_error", optimizer="adam")


# Training of the model
epochs = 50
seq.fit(
    x_train,
    y_train,
    batch_size=10,
    epochs=epochs,
    verbose=2,
    validation_split=0.2,
) 



