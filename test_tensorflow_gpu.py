from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.test.gpu_device_name()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))