from __future__ import absolute_import, division, print_function, unicode_literals
from utils import readCsv

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

csvLines = readCsv("../trainset_csv/trainNodules_gt.csv")
print(csvLines)