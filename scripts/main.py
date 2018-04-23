# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:24 2018

@author: Sandiagle
"""

import numpy as np
from models import YannLeCun
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalMaxPooling2D, Flatten
from keras.models import Model
import keras.optimizers as op
import matplotlib.pyplot as plt
import keras.applications
from keras.utils.vis_utils import plot_model

shape = (224, 224)

model = keras.applications.mobilenet.MobileNet(input_shape=(
    224, 224, 3), alpha=0.5, include_top=None, weights='imagenet', input_tensor=None, pooling='max')
plot_model(model, to_file='model.png', show_shapes=True)

x = model.output
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(5, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=model.input, outputs=predictions)

model.compile(optimizer=keras.optimizers.rmsprop(lr=0.001), loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    '../data/train-pre',  # this is the target directory
    target_size=shape,  # all images will be resized to 150x150
    batch_size=32)

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    '../data/validation',
    target_size=shape,
    batch_size=32)

history = model.fit_generator(
    generator=train_generator,
    verbose=2,
    epochs=5,
    validation_data=validation_generator)
# always save your weights after training or during training
model.save_weights('first_try.h5')

# %%
# 打印loss记录
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('train val loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')


