from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
#import pickle
import matplotlib.pyplot as plt
import numpy as np

tf.test.gpu_device_name()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# dataset of handwritten digits 28x28 images
mnist = tf.keras.datasets.mnist

# load dataset values separated into training and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# plot and print original sample from training
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
print(x_train[0])

# normalize dataset values to scale given in axis
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# plot and print normalized sample from training
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
print(x_train[0])

# build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# final output layer as the number of classes (in this case 10)
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model training parameters
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model training
model.fit(x_train, y_train, epochs=3)

# test model, verbose is 0 due to progress bar printing bug
val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
print(val_loss, val_acc)

# save model
model.save('num_reader.model')

# load model
loaded_model = tf.keras.models.load_model('num_reader.model')

# predict using model
# predictions will have probability distributions for each element of the list given to predict
predictions = loaded_model.predict([x_test])

# use np.argmax to show the prediction in a readable manner
print(np.argmax(predictions[0]))

# plot the image associated with shown prediction
plt.imshow(x_test[0], cmap = plt.cm.binary)
plt.show()


#x = pickle.load(open("X.pickle", "rb"))
#y = pickle.load(open("y.pickle", "rb"))

#model = Sequential()