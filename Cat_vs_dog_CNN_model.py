import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

# Data preprocessing , Image augmentations
# Augmenting only the training sets

train_datagen = ImageDataGenerator(rescale= 1./255, shear_range= 0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory('E:/dataset/training_set', target_size= (64,64), batch_size=32,class_mode='binary')

#preprocessing to test set
test_datagen = ImageDataGenerator(rescale= 1./255)
test_set = test_datagen.flow_from_directory('E:/dataset/test_set', target_size= (64,64), batch_size=32,class_mode='binary')

#Convulation
cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size= 3, activation= 'relu', input_shape= [64,64,3]))

#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 2))

#Second convulation layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size= 3, activation= 'relu'))

#Second pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 2))

#flattening
cnn.add(tf.keras.layers.Flatten())

#full connection
cnn.add(tf.keras.layers.Dense(units=128, activation= 'relu' ))

#output
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#compiling the cnn
cnn.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

cnn.summary()

#Traingin the network
cnn.fit(x = training_set, validation_data= test_set, epochs= 20)

# #Making a single prediction
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('E:/dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64,64))
print(test_image.shape())
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)


result = cnn.predict(test_image)

print(training_set.class_indices)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

#making multiple prediction:
import os
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

list = os.listdir('E:/Udemy/Section 40 - Convolutional Neural Networks (CNN)/dataset/single_prediction')
print(list)
print(len(list))
print(training_set.class_indices)
for label in list:
    path = "E:/dataset/single_prediction"
    file = path + '/' + label
    test_image = image.load_img(file, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = preprocess_input(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)

    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

    print(prediction)
