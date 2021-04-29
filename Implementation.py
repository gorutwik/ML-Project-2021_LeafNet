
import numpy as np
import cv2
import os
from os import listdir
import random
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import image

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/My Drive/data')
!ls

data_folder = 'train data'
imageSize = tuple((256,256))
batch_size = 32

import tensorflow as tf
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
height = 256
width = 256
depth = 3
classifier = Sequential()
inputShape = (height, width, depth)
chanDim = -1

if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1

classifier.add(tf.keras.layers.InputLayer(input_shape=inputShape))
classifier.add(Conv2D(24, (11, 11), activation='relu', padding='same', strides=4))
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))
classifier.add(Conv2D(64, (5, 5), activation='relu', padding='same', strides=1))
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))
classifier.add(Conv2D(96, (3, 3), activation='relu', padding='same', strides=1))
classifier.add(Conv2D(96, (5, 5), activation='relu', padding='same', strides=1))
classifier.add(Conv2D(64, (5, 5), activation='relu', padding='same', strides=1))
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))
classifier.add(Dense(500,activation='relu'))
classifier.add(Flatten())
classifier.add(Dense(100,activation='relu'))
classifier.add(Dense(3,activation='softmax'))

classifier.summary()

from keras.preprocessing.image import img_to_array
def convert_img_array(image_dir):
    try:
        photo = cv2.imread(image_dir)
        if photo is not None:
            photo = cv2.resize(photo, imageSize)
            return img_to_array(photo)
        else:
            return np.array([])
    except Exception as e:
        print(f"{e}")
        return None

imageList = [] 
labelList = []

try:
    folder_list = listdir(data_folder)

    for folder in folder_list:
        print(f"{folder}")
        
        image_directory = listdir(f"{data_folder}/{folder}/")
        for every_image in image_directory:
            image = f"{data_folder}/{folder}/{every_image}"
            if image.endswith(".jpg")==True or image.endswith(".JPG")==True:
              imageList.append(convert_img_array(image))
              labelList.append(folder)
              
    print("Images Loaded!")  
except Exception as e:
    print(f"Error : {e}")
image_len = len(imageList)
print(f"Total number of images: {image_len}")

np_image_list = np.array(imageList, dtype=np.float16) / 225.0
print()

from sklearn.preprocessing import MultiLabelBinarizer
label_b = LabelBinarizer()
image_labels = label_b.fit_transform(labelList)
numClasses = len(label_b.classes_)
print("Number of classes: ", numClasses)

from keras.preprocessing.image import ImageDataGenerator
imgDG = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.3, random_state = 35)

from keras import optimizers
classifier.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.005),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

history = classifier.fit_generator(imgDG.flow(x_train, y_train, batch_size=batch_size),
                                   steps_per_epoch=len(x_train)//batch_size,
                                   validation_data=(x_test, y_test),
                                   epochs=55,
                                   validation_steps=len(x_train)//batch_size)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

accuracy = classifier.evaluate(x_test, y_test)
print(accuracy)