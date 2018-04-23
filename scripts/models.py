# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:24 2018

@author: Sandiagle
"""

# GRADED FUNCTION: HappyModel

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import regularizers

def YannLeCun(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=(9, 9),
                     strides=1, activation='relu', kernel_regularizer =regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=8, kernel_size=(7, 7),
                     strides=1, activation='relu', kernel_regularizer =regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=16, kernel_size=(5, 5),
                     strides=1, activation='relu', kernel_regularizer =regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     strides=1, activation='relu', kernel_regularizer =regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # the model so far outputs 3D feature maps (height, width, features)

    # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(units=32,activation='relu', kernel_regularizer =regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=16, activation='sigmoid',kernel_regularizer =regularizers.l2(0.01)))


    return model
