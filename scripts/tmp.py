# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:45:02 2018

@author: Sandiagal
"""

import database_IO as io
import matplotlib.pyplot as plt
import numpy as np
import models
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
import keras.optimizers as op
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical

# %%

path_orig = "../data/set/"
incl_agmt = False
now_splits = 3
n_splits = 3
shape = (224, 224, 3)
batch_size = 32
epochs = 10

# %%

imgs_orig, imgs_name_orig, labels_orig, imgs_agmt, imgs_name_agmt, labels_agmt, classes = io.load_all(
    path_orig, incl_agmt)
imgs_train, imgs_test, labels_train, labels_test = io.data_split(
    imgs_orig, labels_orig, imgs_agmt, labels_agmt, n_splits, now_splits, incl_agmt)
del imgs_orig, imgs_name_orig, labels_orig, imgs_agmt, imgs_name_agmt, labels_agmt

imgs_train, labels_train = shuffle(imgs_train, labels_train)
imgs_vld, imgs_test, labels_vld, labels_test = io.train_test_split(
    imgs_test, labels_test, test_size=0.5)

labels_train = np.vectorize(classes.get)(labels_train)
labels_vld = np.vectorize(classes.get)(labels_vld)
labels_test = np.vectorize(classes.get)(labels_test)
labels_train = to_categorical(labels_train, len(classes))
labels_vld = to_categorical(labels_vld, len(classes))
labels_test = to_categorical(labels_test, len(classes))

# %%

model = models.YannLeCun(shape)
model.compile(optimizer=op.adam(decay=1e-6), loss='categorical_crossentropy',
              metrics=['accuracy'])
plot_model(model, to_file='YannLeCun.png', show_shapes=True)
model.summary()

# %%

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
#    zca_whitening=True,
#    zca_epsilon=1e-6,
    rescale=1./255)

datagen.fit(imgs_train)

train_generator = datagen.flow(imgs_train, labels_train, batch_size=batch_size)

vld_generator = datagen.flow(imgs_vld, labels_vld, batch_size=batch_size)

# %%

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(labels_train) / batch_size,
    epochs=epochs,
    validation_data=vld_generator,
    validation_steps=len(labels_vld) / batch_size)

# %%

model.save_weights('first_try.h5')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('train val loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
