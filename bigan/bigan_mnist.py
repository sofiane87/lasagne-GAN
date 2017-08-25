from __future__ import print_function

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.merge import Multiply
from keras.models import Sequential, Model
from keras.optimizers import Adam,RMSprop, SGD
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import numpy as np
from time import time
import sys


backend_name = K.backend()

is_tf = False
if 'tensorflow' in backend_name.lower():
    is_tf = True


import platform
print('platform : ', platform.node().lower())

is_sofiane = False

if 'alison' in  platform.node().lower():
    mnist_path = '/Users/pouplinalison/Documents/skin_analytics/code_dcgan/inData/celeba.npy'
elif 'desktop' in  platform.node().lower():
    is_sofiane = True
    mnist_path = 'D:\Code\data\sceleba.npy'
elif 'sofiane' in platform.node().lower():
    mnist_path = '/Users/sofianemahiou/Code/data/mnist.npy'
else:
    mnist_path = '/data/users/amp115/skin_analytics/inData/celeba.npy'

from bigan_root import BIGAN_ROOT


class BIGAN(BIGAN_ROOT):
    def __init__(self,example_bool = False, test_model = False,interpolate_bool=False,mnist_path=mnist_path,preload=False,start_iteration=0,train_bool=True):
        super(BIGAN, self).__init__(example_bool = example_bool, train_bool=train_bool, test_model=test_model,interpolate_bool=interpolate_bool,
                                    img_rows=28,img_cols=28,channels=1, save_folder='bigan/mnist/'
                                    ,latent_dim=100,preload=preload)
        
        self.dataPath = mnist_path

   


    def build_generator(self):


        z = Input(shape=(self.latent_dim,))

        model = Dense(1024 * 2 * 2)(z)
        # print(model.shape)
        model = LeakyReLU(alpha=0.2)(model)
        model = Reshape((2, 2, 1024))(model)
        # print(model.shape)
        model = BatchNormalization(momentum=0.8)(model)
        # print(model.shape)
        model = Deconv(512,kernel_size=4,strides=(2,2),padding="valid")(model)        
        # print(model.shape)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Cropping2D(1)(model)
        # print(model.shape)
        model = Deconv(256,kernel_size=4,strides=(2,2))(model)        
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        # print(model.shape)
        model = Cropping2D(1)(model)
        # print(model.shape)
        model = Deconv(128,kernel_size=4,strides=(2,2))(model)        
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        # print(model.shape)
        model = Cropping2D(1)(model)
        # print(model.shape)
        model = Deconv(1,kernel_size=4,strides=(2,2))(model)
        model = Activation("tanh")(model)
        # print(model.shape)
        model = Cropping2D(3)(model)
        # print(model.shape)

        # model = UpSampling2D()(model)
        # model = Conv2D(128, kernel_size=3, padding="same")(model)
        # model = Activation("relu")(model)
        # model = BatchNormalization(momentum=0.8)(model)
        # model = UpSampling2D()(model)
        # model = Conv2D(64, kernel_size=3, padding="same")(model)
        # model = Activation("relu")(model)
        # model = BatchNormalization(momentum=0.8)(model)
        # model = Conv2D(1, kernel_size=3, padding="same")(model)
        # model = Activation("tanh")(model)

        return Model(z, model)


    def build_encoder(self):

        # model.add(Flatten(input_shape=self.img_shape))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(self.latent_dim))

        # model.summary()


        img = Input(shape=self.img_shape)
        model = ZeroPadding2D(3)(img)
        model = Conv2D(128, kernel_size=4, strides=2, input_shape=self.img_shape, padding="valid")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = AveragePooling2D()(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = ZeroPadding2D(1)(model)
        model = Conv2D(256, kernel_size=4, strides=2, input_shape=self.img_shape, padding="valid")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = AveragePooling2D()(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = ZeroPadding2D(1)(model)
        model = Conv2D(256, kernel_size=4, strides=2, input_shape=self.img_shape, padding="valid")(model)        
        model = Activation("tanh")(model)


        # model = Dropout(0.25)(model)
        # model = Conv2D(64, kernel_size=3, strides=2, padding="same")(model)
        # model = ZeroPadding2D(padding=((0,1),(0,1)))(model)
        # model = LeakyReLU(alpha=0.2)(model)
        # model = Dropout(0.25)(model)
        # model = BatchNormalization(momentum=0.8)(model)
        # model = Conv2D(128, kernel_size=3, strides=2, padding="same")(model)
        # model = LeakyReLU(alpha=0.2)(model)
        # model = Dropout(0.25)(model)
        # model = BatchNormalization(momentum=0.8)(model)
        # model = Conv2D(256, kernel_size=3, strides=1, padding="same")(model)
        # model = LeakyReLU(alpha=0.2)(model)
        # model = Dropout(0.25)(model)
        
        model = Flatten()(model)
        z = Dense(self.latent_dim)(model)
        
        return Model(img, z)
    def build_discriminator(self):


        # img = Input(shape=self.img_shape)
        # model_image = Conv2D(32, kernel_size=3, strides=2, padding="same")(img)
        # model_image = LeakyReLU(alpha=0.2)(model_image)
        # model_image = Dropout(0.25)(model_image)
        # model_image = Conv2D(64, kernel_size=3, strides=2, padding="same")(model_image)
        # model_image = ZeroPadding2D(padding=((0,1),(0,1)))(model_image)
        # model_image = LeakyReLU(alpha=0.2)(model_image)
        # model_image = Dropout(0.25)(model_image)
        # model_image = BatchNormalization(momentum=0.8)(model_image)
        # # model_image = Conv2D(128, kernel_size=3, strides=2, padding="same")(model_image)
        # # model_image = LeakyReLU(alpha=0.2)(model_image)
        # # model_image = Dropout(0.25)(model_image)
        # # model_image = BatchNormalization(momentum=0.8)(model_image)
        # # model_image = Conv2D(256, kernel_size=3, strides=1, padding="same")(model_image)
        # # model_image = LeakyReLU(alpha=0.2)(model_image)
        # # model_image = Dropout(0.25)(model_image)
        
        # # z_shape = int(np.prod(model_image.shape[1:]))
        # model_image = Flatten()(model_image)



        img = Input(shape=self.img_shape)

        model_image = ZeroPadding2D(3)(img)
        model_image = Conv2D(68, kernel_size=4, strides=2, padding="valid")(model_image)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = BatchNormalization(momentum=0.8)(model_image)

        model_image = ZeroPadding2D(2)(model_image)
        model_image = Conv2D(128, kernel_size=5, strides=2, padding="valid")(model_image)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = BatchNormalization(momentum=0.8)(model_image)

        model_image = ZeroPadding2D(2)(model_image)
        model_image = Conv2D(256, kernel_size=5, strides=2, padding="valid")(model_image)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = BatchNormalization(momentum=0.8)(model_image)

        model_image = ZeroPadding2D(2)(model_image)
        model_image = Conv2D(512, kernel_size=5, strides=2, padding="valid")(model_image)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = BatchNormalization(momentum=0.8)(model_image)


        model_image = BatchNormalization(momentum=0.8)(model_image)

        model_image = ZeroPadding2D(2)(model_image)
        model_image = Conv2D(1024, kernel_size=5, strides=2, padding="valid")(model_image)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = BatchNormalization(momentum=0.8)(model_image)
        model_image = Flatten()(model_image)
        model_image = BatchNormalization(momentum=0.8)(model_image)
        model_image = Dense(self.latent_dim)(model_image)
        model_image = BatchNormalization(momentum=0.8)(model_image)


        # model_image = Conv2D(64, kernel_size=3, strides=2, padding="valid")(model_image)
        # model_image = ZeroPadding2D(padding=((0,1),(0,1)))(model_image)
        # model_image = LeakyReLU(alpha=0.2)(model_image)
        # model_image = Dropout(0.25)(model_image)
        # model_image = BatchNormalization(momentum=0.8)(model_image)
        # model_image = Conv2D(128, kernel_size=3, strides=2, padding="same")(model_image)
        # model_image = LeakyReLU(alpha=0.2)(model_image)
        # model_image = Dropout(0.25)(model_image)
        # model_image = BatchNormalization(momentum=0.8)(model_image)
        # model_image = Conv2D(256, kernel_size=3, strides=1, padding="same")(model_image)
        # model_image = LeakyReLU(alpha=0.2)(model_image)
        # model_image = Dropout(0.25)(model_image)
        


        z = Input(shape=(self.latent_dim, ))

        # d_in = concatenate([model_image,model_z,multiply([model_image,model_z])])
        d_in = concatenate([model_image,z])

        model = Dense(1024)(d_in)
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


    def load_data(self):
        print('---- loading MNIST -----')
        #X_train = np.load('/data/users/amp115/skin_analytics/inData/mnist.npy')
        X_train = np.load(self.dataPath).transpose([0,2,3,1])
        X_train = ((X_train - 127.5)/127.5).astype('float32')
        print('----- MNIST loaded ------')
        print (X_train.shape, X_train.min(), X_train.max())
        return X_train


if __name__ == '__main__':
    test_bool = False
    train_bool = True
    interpolate_bool = False
    preload=False
    start_iteration = 0
    example_bool = False
    if '-preload' in sys.argv[1:]:
        preload = True
    if '-test' in sys.argv[1:]:
        test_bool = True
        train_bool = False
    if '-interpolate' in sys.argv[1:]:
        interpolate_bool = True
        train_bool = False
    if '-start' in sys.argv[1:]:
        start_iteration = int(sys.argv[sys.argv.index('-start')+1])
        if start_iteration != 0:
            preload = True
    if '-example' in sys.argv[1:]:
        train_bool = False
        preload = True
        example_bool = True


    bigan = BIGAN(example_bool = example_bool, train_bool= train_bool, test_model = test_bool,interpolate_bool = interpolate_bool,preload=preload)
    bigan.run(epochs=50001, batch_size=64, save_interval=100,start_iteration=start_iteration)






