from __future__ import absolute_import, division, print_function, unicode_literals
import utils

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv3D, MaxPooling3D

import matplotlib.pyplot as plt
import numpy as np

def getCubes():
    csvLines = utils.readCsv("../trainset_csv/trainNodules_gt.csv")
    last_ID = 0
    scan = 0
    spacing = 0
    origin = 0
    
    cubeList = []    
    textures = [row[-1] for row in csvLines]

    # delete 1st element
    del textures[0]

    # limit to 200 nodules out of ~1200
    max_nodules = 50
    del textures[max_nodules:]

    count = 0

    # ignore header
    for line in csvLines[1:]:
        
        # limit to 200 nodules out of ~1200
        count += 1
        if count > max_nodules:
            break
        
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

        scan_cube = utils.extractCube(scan, spacing, real_coords, cube_size=60)
        
        #np.append(cubeList, scan_cube)
        cubeList.append(scan_cube)
        
        #_, axs = plt.subplots(2,3)
        #axs[0,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:], cmap=plt.cm.binary)
        #axs[1,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:], cmap=plt.cm.binary)
        #axs[0,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:], cmap=plt.cm.binary)
        #axs[1,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:], cmap=plt.cm.binary)
        #axs[0,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)], cmap=plt.cm.binary)
        #axs[1,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)], cmap=plt.cm.binary)    
        #plt.show()

        #nodule_coords
        last_ID = current_ID
    
    return cubeList, textures
            

def getFileID(id):
    id_digits = len(id)
    zeros = 4 - id_digits
    file_name = ''
    for i in range(zeros):
        file_name += '0'

    return file_name + id


def parseTrainingData(cubeList, textures):
    # put data in correct format
    valid_cube_list = np.array(cubeList).reshape(-1, 60, 60, 60, 1)
    valid_cube_list = valid_cube_list.astype(float)

    #print(valid_cube_list[0])
    #valid_cube_list = (valid_cube_list - np.min(valid_cube_list))/np.ptp(valid_cube_list)
    #print(valid_cube_list[0])

    # put labels in correct format
    textures = np.array(textures)
    textures = textures.astype(float)

    valid_textures = np.zeros((textures.__len__(), 3), dtype=int)

    for i in range(textures.__len__()):
        texture_value = textures[i]
        if texture_value < 2.33:
            valid_textures[i, 0] = 1
        elif texture_value < 3.66:
            valid_textures[i, 1] = 1
        else:
            valid_textures[i, 2] = 1


    return valid_cube_list, valid_textures

def createModel():
    model = Sequential()

    '''#model.add(Conv3D(256, (8, 8, 8), input_shape=(60,60,60,1)))
    model.add(Conv3D(32, (8, 8, 8), input_shape=(60,60,60,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    #model.add(Conv3D(256, (8, 8, 8)))
    model.add(Conv3D(16, (8, 8, 8)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(16, (8, 8, 8)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))'''

    model.add(Conv3D(64, (3, 3, 3), input_shape=(60,60,60,1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv3D(128, (3, 3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv3D(256, (3, 3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))

    print(model.output_shape)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    print(model.output_shape)

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))

    # Output layer
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    return model


#print(scan,spacing,origin,transfmat)


# list available gpus in order to limit memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            #tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

cubeList, textures = getCubes()

valid_cube_list, valid_textures = parseTrainingData(cubeList, textures)

model = createModel()

model.fit(valid_cube_list, valid_textures, batch_size=1, epochs=16, validation_split=0.3)


to_predict = np.array([valid_cube_list[0]]).reshape(-1, 60, 60, 60, 1)
predictions = model.predict(to_predict)

print(predictions[0])
print(np.argmax(predictions[0]))
print(valid_textures[0])