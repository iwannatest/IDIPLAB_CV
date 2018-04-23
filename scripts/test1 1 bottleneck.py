# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:40:57 2018

@author: Sandiagle
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras.optimizers
from keras import metrics
import keras.applications
from keras.utils.vis_utils import plot_model


# %%
# 导入模型
model = keras.applications.mobilenet.MobileNet(input_shape=(
    224, 224, 3), alpha=0.5, include_top=None, weights='imagenet', input_tensor=None, pooling='max')
plot_model(model, to_file='model.png', show_shapes=True)

shape = (224, 224)

train_datagen = ImageDataGenerator(
    rescale=1./255)

# %%
# 记录bottleneck

# this is the augmentation configuration we will use for training

generator = train_datagen.flow_from_directory(
    '../data/train',
    target_size=shape,
    batch_size=32,
    class_mode=None,  # this means our generator will only yield batches of data, no labels
    shuffle=False)  # our data will be in order, so all first 1000 images will be cats, then 1000 dogs
# the predict_generator method returns the output of a model, given
# a generator that yields batches of numpy data
train_data = model.predict_generator(generator)
# save the output as a Numpy array
#np.save(open('bottleneck_features_train.npy', 'ab'), bottleneck_features_train)

generator = train_datagen.flow_from_directory(
    '../data/validation',
    target_size=shape,
    batch_size=32,
    class_mode=None,
    shuffle=False)
validation_data = model.predict_generator(generator)
#np.save(open('bottleneck_features_validation.npy', 'ab'), bottleneck_features_validation)

# %%
# 利用bottleneck

#train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
# the features were saved in order, so recreating the labels is easy
train_labels = np.array([0] * 581 + [1] * 168 + [2]
                        * 104 + [3] * 30 + [4] * 149)
train_labels = to_categorical(train_labels, num_classes=5)

#validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
validation_labels = np.array([0] * 20 + [1] * 7 + [2] * 4 + [3] * 2 + [4] * 6)
validation_labels = to_categorical(validation_labels, num_classes=5)

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=train_data.shape[1:]))
model.add(Dropout(0.5))
model.add(Dense(5, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train_data, train_labels,
                    nb_epoch=500, batch_size=128,
                    validation_data=(validation_data, validation_labels))
model.save_weights('bottleneck_fc_model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('train val loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
