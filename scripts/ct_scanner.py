from __future__ import absolute_import, division, print_function, unicode_literals
import utils

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

import time
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np

def getCubes(cubeSize):
    csvLines = utils.readCsv("../trainset_csv/trainNodules_gt.csv")
    last_ID = 0
    scan = 0
    spacing = 0
    origin = 0
    
    cubeList = []    
    textures = [row[-1] for row in csvLines]

    # delete 1st element (header)
    del textures[0]

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

        scan_cube = utils.extractCube(scan, spacing, real_coords, cube_size=cubeSize)
        
        cubeList.append(scan_cube)

        # nodule_coords
        last_ID = current_ID
    
    return cubeList, textures

def getTextures(count):
    csvLines = utils.readCsv("../trainset_csv/trainNodules_gt.csv")
    textures = [row[-1] for row in csvLines]
    # delete 1st element
    del textures[0]

    if count != -1:
        del textures[count:]

    return textures

def getMaskedCubes():

    csvLines = utils.readCsv("../trainset_csv/trainNodules_gt.csv")
    last_ID = 0
    scan = 0
    spacing = 0
    origin = 0

    maskedCubeList = []  

    # ignore header
    for line in csvLines[1:]:
        
        # get image of this patient only one time (there are repeated patient ids)
        current_ID = line[0]
        if last_ID != current_ID:
            print(getFileID(current_ID))
            scan,spacing,origin,_ = utils.readMhd('../LNDb dataset/dataset/LNDb-' + getFileID(current_ID) + '.mhd')
            spacing = [float(spacing[i]) for i in range(3)]

        # find the coordinates of the current nodule (it is done for every line of the csv)
        finding_coords = line[4:7]
        
        nodule_x = (float(finding_coords[0]) - float(origin[0])) / float(spacing[0])
        nodule_y = (float(finding_coords[1]) - float(origin[1])) / float(spacing[1])
        nodule_z = (float(finding_coords[2]) - float(origin[2])) / float(spacing[2])
        real_coords = [nodule_x, nodule_y, nodule_z]

        # get a mask for the image of this patient (from one of the radiologists that found the current nodule)
        radiologists = line[1] # list of radiologists that found the current nodule
        radId = str(radiologists[0]) # always choose the mask from the first radiologist in the list
        mask,_,_,_ = utils.readMhd('../LNDb dataset/masks/LNDb-' + getFileID(current_ID) + '_rad' + radId + '.mhd')

        # filter the whole image scan by the mask
        short_min = -32768 # lowest signed short value
        masked_scan = np.where(mask == 0, short_min, scan)
    
        # extract mini cube of the current nodule on the masked scan
        masked_cube = utils.extractCube(masked_scan, spacing, real_coords, cube_size=80)

        # add masked cubed to the list
        maskedCubeList.append(masked_cube)

        last_ID = current_ID
    
    return maskedCubeList

def saveMaskedCubes(maskedCubeList):
    for i in range(maskedCubeList.__len__()):
        masked_cube = maskedCubeList[i]
        filename = "../LNDb dataset/mini_masked_cubes/masked_cube" + str(i)
        with open(filename, 'wb') as outfile:
            pickle.dump(masked_cube, outfile)

def loadMaskedCubes(count):
    if count == -1:
        count = 1219

    maskedCubeList = []
    for i in range(count):
        filename = "../LNDb dataset/mini_masked_cubes/masked_cube" + str(i)
        with open(filename, 'rb') as infile:
            masked_cube = pickle.load(infile)
            maskedCubeList.append(masked_cube)
    
    return maskedCubeList

def saveMiniCubes(cubeList):
    for i in range(cubeList.__len__()):
        cube = cubeList[i]
        filename = "../LNDb dataset/mini_cubes/cube" + str(i)
        with open(filename, 'wb') as outfile:
            pickle.dump(cube, outfile)

def loadMiniCubes(count):
    if count == -1:
        count = 1219

    cubeList = []
    for i in range(count):
        filename = "../LNDb dataset/mini_cubes/cube" + str(i)
        with open(filename, 'rb') as infile:
            cube = pickle.load(infile)
            cubeList.append(cube)
    
    return cubeList

def getFileID(id):
    id_digits = len(id)
    zeros = 4 - id_digits
    file_name = ''
    for i in range(zeros):
        file_name += '0'

    return file_name + id


def parseTrainingData(cubeList, textures, validationSplit, cubeSize):
    texture_counter = [0, 0, 0]

    # put labels in correct format
    textures = np.array(textures)
    textures = textures.astype(float)

    valid_textures = np.zeros((textures.__len__(), 3), dtype=int)

    for i in range(textures.__len__()):
        texture_value = textures[i]
        if texture_value < 2.33:
            valid_textures[i, 0] = 1
            texture_counter[0] += 1
        elif texture_value < 3.66:
            valid_textures[i, 1] = 1
            texture_counter[1] += 1
        else:
            valid_textures[i, 2] = 1
            texture_counter[2] += 1
    
    print(texture_counter)

    training_size = math.ceil(textures.__len__() * (1 - validationSplit))
    validation_size = textures.__len__() - training_size

    valid_cube_list_training = []
    valid_cube_list_validation = []
    valid_textures_training = []
    valid_textures_validation = []

    count_0 = 0
    count_1 = 0
    count_2 = 0

    count_t = 0
    count_v = 0

    threshold_0 = math.floor(validationSplit * texture_counter[0])
    threshold_1 = math.floor(validationSplit * texture_counter[1])
    threshold_2 = validation_size - (threshold_0 + threshold_1)

    for i in range(textures.__len__()):
        texture_value = textures[i]
        if texture_value < 2.33:
            if count_0 < threshold_0:
                valid_cube_list_validation.append(cubeList[i])
                valid_textures_validation.append([1, 0, 0])
                count_v += 1
            else:
                valid_cube_list_training.append(cubeList[i])
                valid_textures_training.append([1, 0, 0])
                count_t += 1
            count_0 += 1
        elif texture_value < 3.66:
            if count_1 < threshold_1:
                valid_cube_list_validation.append(cubeList[i])
                valid_textures_validation.append([0, 1, 0])
                count_v += 1
            else:
                valid_cube_list_training.append(cubeList[i])
                valid_textures_training.append([0, 1, 0])
                count_t += 1
            count_1 += 1
        else:
            if count_2 < threshold_2:
                valid_cube_list_validation.append(cubeList[i])
                valid_textures_validation.append([0, 0, 1])
                count_v += 1
            else:
                valid_cube_list_training.append(cubeList[i])
                valid_textures_training.append([0, 0, 1])
                count_t += 1
            count_2 += 1
    

    print("Total size: ", textures.__len__())
    print("validation size: ", validation_size)
    print("training size: ", training_size)
    print("validation + training size: ", (validation_size + training_size))
    print("count_0 + count_1 + count_2: ",  (count_0 + count_1 + count_2))
    print("threshold_0: ", threshold_0)
    print("threshold_1: ", threshold_1)
    print("threshold_2: ", threshold_2)

    # put textures data in correct format
    valid_textures_training = np.array(valid_textures_training)
    valid_textures_training = valid_textures_training.astype(float)

    valid_textures_validation = np.array(valid_textures_validation)
    valid_textures_validation = valid_textures_validation.astype(float)

    # put cube data in correct format
    valid_cube_list_training = np.array(valid_cube_list_training).reshape(-1, cubeSize, cubeSize, cubeSize, 1)
    valid_cube_list_training = valid_cube_list_training.astype(float)

    valid_cube_list_validation = np.array(valid_cube_list_validation).reshape(-1, cubeSize, cubeSize, cubeSize, 1)
    valid_cube_list_validation = valid_cube_list_validation.astype(float)

    # substitute lowest short value with -1500 (better for input normalization)
    #valid_cube_list_training = np.where(valid_cube_list_training == -32768, -1500, valid_cube_list_training)
    #valid_cube_list_validation = np.where(valid_cube_list_validation == -32768, -1500, valid_cube_list_validation)
    
    # Input normalization for cube data (-1 to 1, mean 0)
    valid_cube_list_training -= np.mean(valid_cube_list_training)
    valid_cube_list_training /= np.std(valid_cube_list_training)

    valid_cube_list_validation -= np.mean(valid_cube_list_validation)
    valid_cube_list_validation /= np.std(valid_cube_list_validation)

    return valid_cube_list_training, valid_cube_list_validation, valid_textures_training, valid_textures_validation

# for cubes size 80
def createModel():
    # create a sequential model with a layout similar to the vgg16 model
    model = Sequential()
    model.add(Conv3D(12, (3, 3, 3), input_shape=(80,80,80,1), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(24, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(48, (3, 3, 3), activation='relu'))
    model.add(Dropout(0.50))
    model.add(Conv3D(48, (3, 3, 3), activation='relu'))
    model.add(Dropout(0.50))
    model.add(Conv3D(48, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    print(model.output_shape)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    print(model.output_shape)

    model.add(Dropout(0.50))
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(0.50))
    
    model.add(Dense(3, activation='softmax'))

    model.summary()

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# for masked cubes size 80
def createModelA1():
    model = Sequential()

    model.add(Conv3D(64, (3, 3, 3), input_shape=(80,80,80,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv3D(128, (3, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv3D(256, (3, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))

    print(model.output_shape)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    print(model.output_shape)

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.50))

    # Output layer
    model.add(Dense(3), activation='softmax')

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# for masked cubes size 80
def createModelA2():
    model = Sequential()

    model.add(Conv3D(12, (3, 3, 3), input_shape=(80,80,80,1), activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv3D(24, (3, 3, 3), activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv3D(48, (3, 3, 3), activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.25))

    print(model.output_shape)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    print(model.output_shape)

    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.50))

    # Output layer
    model.add(Dense(3, activation='softmax'))

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def main():
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

    # run once to get mini cubes size 80, create "mini_cubes" folder inside 'LNDb dataset' folder

    #cubeList, textures = getCubes(80)
    #saveMiniCubes(cubeList)
    #print("done")
    #time.sleep(20)

    # run once to get mini MASKED cubes, create "mini_masked_cubes" folder inside 'LNDb dataset' folder

    #maskedCubeList = getMaskedCubes()
    #saveMaskedCubes(maskedCubeList)
    #print("done")
    #time.sleep(20)

    cubeList = loadMiniCubes(-1) # change between loadMiniCubes and loadMaskedCubes
    textures = getTextures(-1)

    validationSplit = 0.3
    cubeSize = 80

    valid_cube_list_training, valid_cube_list_validation, valid_textures_training, valid_textures_validation = parseTrainingData(cubeList, textures, validationSplit, cubeSize)

    model = createModel()
    #model = tf.keras.models.load_model('../models/model')


    model.fit(valid_cube_list_training, valid_textures_training, batch_size=16, epochs=32, validation_data=(valid_cube_list_validation, valid_textures_validation))

    # folder needs to exist before instruction is ran
    model.save('../models/model')

    to_predict = np.array([valid_cube_list_validation[0]]).reshape(-1, cubeSize, cubeSize, cubeSize, 1)
    predictions = model.predict(to_predict)

    print(predictions[0])
    print(np.argmax(predictions[0]))
    print(valid_textures_validation[0])

main()