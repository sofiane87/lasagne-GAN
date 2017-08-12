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

backend_name = K.backend()

is_tf = False
if 'tensorflow' in backend_name.lower():
    is_tf = True

from bigan_root import BIGAN_ROOT

class BIGAN(BIGAN_ROOT):
    def __init__(self,reload_model = False):
        super(BIGAN, self).__init__(reload_model=reload_model)

    def build_encoder(self):

        img = Input(shape=self.img_shape)

        model = Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same")(img)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.25)(model)
        model = Conv2D(64, kernel_size=3, strides=2, padding="same")(model)
        model = ZeroPadding2D(padding=((0,1),(0,1)))(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.25)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Conv2D(128, kernel_size=3, strides=2, padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.25)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Conv2D(256, kernel_size=3, strides=1, padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.25)(model)
        model = Flatten()(model)
        z = Dense(self.latent_dim)(model)



        return Model(img, z)

    def build_generator(self):


        z = Input(shape=(self.latent_dim,))

        model = Dense(128 * 7 * 7, activation="relu")(z)
        model = Reshape((7, 7, 128))(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = UpSampling2D()(model)
        model = Conv2D(128, kernel_size=3, padding="same")(model)
        model = Activation("relu")(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = UpSampling2D()(model)
        model = Conv2D(64, kernel_size=3, padding="same")(model)
        model = Activation("relu")(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Conv2D(1, kernel_size=3, padding="same")(model)
        model = Activation("tanh")(model)

        return Model(z, model)

    def build_discriminator(self):


        img = Input(shape=self.img_shape)
        model_image = Conv2D(32, kernel_size=3, strides=2, padding="same")(img)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = Dropout(0.25)(model_image)
        model_image = Conv2D(64, kernel_size=3, strides=2, padding="same")(model_image)
        model_image = ZeroPadding2D(padding=((0,1),(0,1)))(model_image)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = Dropout(0.25)(model_image)
        model_image = BatchNormalization(momentum=0.8)(model_image)
        model_image = Conv2D(128, kernel_size=3, strides=2, padding="same")(model_image)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = Dropout(0.25)(model_image)
        model_image = BatchNormalization(momentum=0.8)(model_image)
        # model_image = Conv2D(256, kernel_size=3, strides=1, padding="same")(model_image)
        # model_image = LeakyReLU(alpha=0.2)(model_image)
        # model_image = Dropout(0.25)(model_image)
        
        model_image = Flatten()(model_image)
        model_image = Dense(self.latent_dim)(model_image)



        z = Input(shape=(self.latent_dim, ))
        model_z = Dense(self.latent_dim)(z)
        # d_in = concatenate([model_image,model_z,multiply([model_image,model_z])])
        d_in = concatenate([model_image,model_z])

        model = Dense(100)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        # model = Dense(1024)(model)
        # model = LeakyReLU(alpha=0.2)(model)
        # model = Dropout(0.5)(model)
        # model = Dense(1024)(model)
        # model = LeakyReLU(alpha=0.2)(model)
        # model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)


        return Model([z, img], validity)




if __name__ == '__main__':
    reload_bool = True
    bigan = BIGAN(reload_model = reload_bool)
    bigan.run(epochs=30001, batch_size=32, save_interval=100)






