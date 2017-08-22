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
    celeba_path = '/Users/pouplinalison/Documents/skin_analytics/code_dcgan/inData/all_moles_128.npy'
elif 'desktop' in  platform.node().lower():
    is_sofiane = True
    celeba_path = 'D:\Code\data\moles.npy'
else:
    celeba_path = '/data/users/amp115/skin_analytics/inData/all_moles_128.npy'

from bigan_root import BIGAN_ROOT


class BIGAN(BIGAN_ROOT):
    def __init__(self,test_model = False,interpolate_bool=False,celeba_path=celeba_path,preload=False,start_iteration=0,train_bool=True):
        super(BIGAN, self).__init__(train_bool=train_bool, test_model=test_model,interpolate_bool=interpolate_bool,
                                    img_rows=128,img_cols=128,channels=3, save_folder='bigan/moles128/'
                                    ,latent_dim=200,preload=preload)
        
        self.dataPath = celeba_path

   


    def build_generator(self):


        noise_shape = (self.latent_dim,)
        
        model = Sequential()

        model.add(Dense(1024 * 4 * 4, activation="relu", input_shape=noise_shape))
        model.add(Reshape((4, 4, 1024)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(512, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_encoder(self):
        
        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(self.latent_dim))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_discriminator(self):


        img_shape = (self.img_rows, self.img_cols, self.channels)
        img = Input(shape=img_shape)

        model_image = Conv2D(4, kernel_size=3, strides=2, padding="same")(img)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = Dropout(0.25)(model_image)
        model_image = Conv2D(16, kernel_size=3, strides=2, padding="same")(model_image)
        model_image = ZeroPadding2D(padding=((0,1),(0,1)))(model_image)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = Dropout(0.25)(model_image)
        model_image = BatchNormalization(momentum=0.8)(model_image)
        model_image = Conv2D(32, kernel_size=3, strides=2, padding="same")(model_image)
        model_image = ZeroPadding2D(padding=((0,1),(0,1)))(model_image)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = Dropout(0.25)(model_image)
        model_image = BatchNormalization(momentum=0.8)(model_image)
        model_image = Conv2D(64, kernel_size=3, strides=2, padding="same")(model_image)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = Dropout(0.25)(model_image)
        model_image = BatchNormalization(momentum=0.8)(model_image)
        model_image = Conv2D(128, kernel_size=3, strides=1, padding="same")(model_image)
        model_image = LeakyReLU(alpha=0.2)(model_image)
        model_image = Dropout(0.25)(model_image)
        model_image = Flatten()(model_image)
        model_image = Dense(self.latent_dim)(model_image)



        z = Input(shape=(self.latent_dim, ))
        model_z = Dense(self.latent_dim)(z)
        # d_in = concatenate([model_image,model_z,multiply([model_image,model_z])])
        d_in = concatenate([model_image,model_z])

        model = Dense(512)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)

        model = Dense(512)(d_in)
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
        print('----- Loading moles -------')
        X_train = np.load(self.dataPath)
        print('------ Data moles : Preprocessing -----')
        # X_train = X_train.transpose([0,2,3,1])
        # Rescale -1 to 1
        if is_sofiane:
            print('initial max : {} , min : {}'.format(X_train.max() , X_train.min()))
            for i in range(X_train.shape[0]):
                print('advancement: {:.2f}%'.format(i/X_train.shape[0]*100),end='\r')
                temp = np.array(X_train[i].astype(np.float32))
                # temp[:,:,1] =  X_train[i,:,:,0]
                # temp[:,:,0] = X_train[i,:,:,1]
                X_train[i] = (temp - 127.5) / 127.5
        else:
            X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        print('moles shape:', X_train.shape, X_train.min(), X_train.max())
        print('------- moles loaded -------')
        
        return X_train
    
    def plot(self, fig, img):
            fig.imshow(np.array(img*255).astype(np.uint8))
            fig.axis('off')

if __name__ == '__main__':
    test_bool = False
    train_bool = True
    interpolate_bool = False
    preload=False
    start_iteration = 0
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

    bigan = BIGAN(train_bool= train_bool, test_model = test_bool,interpolate_bool = interpolate_bool,preload=preload)
    bigan.run(epochs=50001, batch_size=64, save_interval=100,start_iteration=start_iteration)






