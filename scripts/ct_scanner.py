from __future__ import absolute_import, division, print_function, unicode_literals
import utils

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

def main():
    csvLines = utils.readCsv("../trainset_csv/trainNodules_gt.csv")
    last_ID = 0
    scan = 0
    spacing = 0
    origin = 0
    # ignore header
    for line in csvLines[1:]:
        current_ID = line[0]
        if last_ID != current_ID:
            print(getFileID(current_ID))
            scan,spacing,origin,_ = utils.readMhd('../LNDb dataset/dataset/LNDb-' + getFileID(current_ID) + '.mhd')
            spacing = [float(spacing[i]) for i in range(3)]

        finding_coords = line[4:7]
        
        nodule_x = (float(finding_coords[0]) - float(origin[0])) / float(spacing[0])
        nodule_y = (float(finding_coords[1]) - float(origin[1])) / float(spacing[1])
        nodule_z = (float(finding_coords[2]) - float(origin[2])) / float(spacing[2])
        real_coords = [nodule_x, nodule_y, nodule_z]
        print("real_coords", real_coords)

        scan_cube = utils.extractCube(scan, spacing, real_coords)
        _, axs = plt.subplots(2,3)
        axs[0,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:], cmap=plt.cm.binary)
        axs[1,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:], cmap=plt.cm.binary)
        axs[0,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:], cmap=plt.cm.binary)
        axs[1,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:], cmap=plt.cm.binary)
        axs[0,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)], cmap=plt.cm.binary)
        axs[1,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)], cmap=plt.cm.binary)    
        plt.show()

        #nodule_coords
        last_ID = current_ID
            

def getFileID(id):
    id_digits = len(id)
    zeros = 4 - id_digits
    file_name = ''
    for i in range(zeros):
        file_name += '0'

    return file_name + id
#print(scan,spacing,origin,transfmat)

#TODO: passar so cubo
#input_shape=(512,512,328)
#model = models.Sequential()
#model.add(layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
#                 activation='relu',
 #                input_shape=input_shape))
#model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(layers.Conv2D(64, (5, 5), activation='relu'))
#model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(layers.Flatten())
#model.add(layers.Dense(1000, activation='relu'))
#model.add(layers.Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.SGD(lr=0.01),
              # metrics=['accuracy'])

# model.fit(x_train, y_train,
          # batch_size=batch_size,
          # epochs=epochs,
          # verbose=1,
          # validation_data=(x_test, y_test),
          # callbacks=[history])

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# class AccuracyHistory(keras.callbacks.Callback):
    # def on_train_begin(self, logs={}):
        # self.acc = []
    # def on_epoch_end(self, batch, logs={}):
        # self.acc.append(logs.get('acc'))

# history = AccuracyHistory()

# plt.plot(range(1,11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()
main()