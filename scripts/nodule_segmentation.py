from __future__ import absolute_import, division, print_function, unicode_literals
import utils

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, Conv3DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

import time
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import block_reduce

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

def getMaskVolumes():

    csvLines = utils.readCsv("../trainset_csv/trainNodules_gt.csv")
    last_ID = 0
    scan = 0
    spacing = 0
    origin = 0

    maskVolumesList = []  

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

        # extract mini cube of the current nodule on the masked scan
        mask_volume = utils.extractCube(mask, spacing, real_coords, cube_size=80)

        # add mask volumes to the list
        maskVolumesList.append(mask_volume)

        last_ID = current_ID
    
    return maskVolumesList

def saveMaskVolumes(maskVolumesList):
    for i in range(maskVolumesList.__len__()):
        mask_volume = maskVolumesList[i]
        filename = "../LNDb dataset/mask_volumes/mask_volume" + str(i)
        with open(filename, 'wb') as outfile:
            pickle.dump(mask_volume, outfile)

def loadMaskVolumes(count):
    if count == -1:
        count = 1219

    maskVolumesList = []
    for i in range(count):
        filename = "../LNDb dataset/mask_volumes/mask_volume" + str(i)
        with open(filename, 'rb') as infile:
            mask_volume = pickle.load(infile)
            maskVolumesList.append(mask_volume)
    
    return maskVolumesList

# run only one time to write the files. then always use load
def getAndSaveMaskVolumes():
    maskVolumesList = getMaskVolumes()
    saveMaskVolumes(maskVolumesList)

def obtainCubeList(num_cubes, reshape_size):
    # load 80x80x80 cubes
    cubeList = loadMiniCubes(num_cubes)

    # put cube list in correct format
    cubeList = np.array(cubeList).reshape(-1, 80, 80, 80, 1)
    cubeList = cubeList.astype(float)

    # Resize cubes to reshape_size x reshape_size x reshape_size
    lower = (80 - reshape_size) // 2
    upper = lower + reshape_size
    cubeList = cubeList[:,lower:upper, lower:upper, lower:upper]

    return cubeList

def obtainMaskVolumesList(num_mask_volumes, reshape_size, final_size):
    # load 80x80x80 masks
    maskVolumesList = loadMaskVolumes(num_mask_volumes)

    '''newMaskVolumeList = []#np.zeros([num_mask_volumes, math.pow(reshape_size, 3)])

    cube_size = 80 // reshape_size
    for mask in maskVolumesList:
        new_mask = []
        for x in range(reshape_size):
            for y in range(reshape_size):
                for z in range(reshape_size):
                    value = 0
                    for real_x in range(cube_size):
                        for real_y in range(cube_size):
                            for real_z in range(cube_size):
                                if mask[x * reshape_size + real_x, y * reshape_size + real_y, z * reshape_size + real_z] == 1:
                                    value = 1
                    new_mask.append(value)
        
        newMaskVolumeList.append(new_mask)

    newMaskVolumeList = np.array(newMaskVolumeList)
    newMaskVolumeList = newMaskVolumeList.astype(float)'''

    # put masks in correct format
    maskVolumesList = np.array(maskVolumesList).reshape(-1, 80, 80, 80)
    maskVolumesList = maskVolumesList.astype(float)

    # Resize cubes to reshape_size x reshape_size x reshape_size
    lower = (80 - reshape_size) // 2
    upper = lower + reshape_size
    maskVolumesList = maskVolumesList[:,lower:upper, lower:upper, lower:upper]

    block_size = reshape_size // final_size

    reshaped_masks = []
    for mask in maskVolumesList:
        reshaped_masks.append(block_reduce(mask, block_size=(block_size, block_size, block_size), func=np.amax))

    ret = []
    for mask in reshaped_masks:
        flat_mask = []
        for x in range(final_size):
            for y in range(final_size):
                for z in range(final_size):
                    flat_mask.append(mask[x][y][z])
        ret.append(flat_mask)
        
    ret = np.array(ret).reshape(-1, int(math.pow(final_size, 3)))
    ret = ret.astype(float)

    ret = np.where(ret == 0, 0, 1)

    return ret

def inputNormalization(input_list):
    # Input normalization for cube data (-1 to 1, mean 0)
    input_list -= np.mean(input_list)
    input_list /= np.std(input_list)

    return input_list

def createModel(cube_dimension, output_dimention):
    model = Sequential()

    block_size = cube_dimension // output_dimention

    model.add(AveragePooling3D(pool_size=(block_size, block_size, block_size), input_shape=(cube_dimension,cube_dimension,cube_dimension,1)))

    print(model.output_shape)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    print(model.output_shape)

    #model.add(Dense(128, activation='relu'))

    # Output layer
    output_layer_dimention = int(math.pow(output_dimention, 3))
    model.add(Dense(output_layer_dimention, activation='sigmoid'))

    model.compile(Adam(lr=.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
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

    num_examples = -1 # number of cubes for training and validation (-1 means use all available cubes)
    cube_dimension = 20 # consider only the centered sub cube with this dimention, discard the outter cube data
    output_dimention = 10 # the final dimension to predict will be (output_dimention x output_dimention x output_dimention)

    # Obtain input and output lists
    cubeList = obtainCubeList(num_examples, cube_dimension)
    maskVolumesList = obtainMaskVolumesList(num_examples, cube_dimension, output_dimention)

    # Normalize inputs
    cubeList = inputNormalization(cubeList)

    # Create and train model
    model = createModel(cube_dimension, output_dimention)
    model.fit(cubeList, maskVolumesList, batch_size=1, epochs=64, validation_split=0.3)

main()

