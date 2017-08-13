from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.merge import Multiply
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
from time import time
import sys

backend_name = K.backend()

is_tf = False
if 'tensorflow' in backend_name.lower():
    is_tf = True

from bigan_root import BIGAN_ROOT
from bigan_conv import BIGAN

class intep(BIGAN_ROOT):

