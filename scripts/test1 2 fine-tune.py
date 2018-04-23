# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:52:21 2018

@author: Sandiagle
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras import optimizers
import matplotlib.pyplot as plt
import keras.applications
from keras.utils import plot_model

shape = (224, 224)

base_model = keras.applications.mobilenet.MobileNet(input_shape=(
    224, 224, 3), alpha=0.5, include_top=None, weights='imagenet',
    input_tensor=None, pooling='max')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Dense(256, activation='relu',
                    input_shape=base_model.output_shape[1:]))
top_model.add(Dropout(0.5))
top_model.add(Dense(5, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights('bottleneck-fc.h5')

# add the model on top of the convolutional base
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
plot_model(model, to_file='MobileNet+finetune.png', show_shapes=True)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:-12]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '../data/train',  # this is the target directory
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
    epochs=10,
    validation_data=validation_generator)
model.save_weights('MobileNet+fc.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('train val loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
