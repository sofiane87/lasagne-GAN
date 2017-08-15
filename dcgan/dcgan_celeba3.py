from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, merge
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Lambda, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD

import keras.backend as K

import matplotlib.pyplot as plt

import sys
import os

import numpy as np

backend_name = K.backend()

is_tf = False
if 'tensorflow' in backend_name.lower():
    is_tf = True


class DCGAN():
    def __init__(self):
        self.img_rows = 64 
        self.img_cols = 64
        self.channels = 3
        self.save_img_folder = 'dcgan/images/'
        optimizer =  Adam(lr=1E-3, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
        optimizer_dis = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        self.latent_dim = 100


        ### generator params
        self.initial_filters = 512
        self.start_dim = int(self.img_cols / 16)
        self.nb_upconv = 4
        self.bn_axis = -1    
        self.initial_reshape_shape = (self.start_dim, self.start_dim, self.initial_filters)
        self.bn_mode = 2

        ### discriminator params
        self.list_f = [64, 128, 256]
        self.num_kernels = 100
        self.dim_per_kernel = 5
        self.use_mbd = True

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer_dis,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (self.latent_dim,)
        
        model = Sequential()

        model.add(Dense(self.initial_filters * self.start_dim * self.start_dim, input_shape=noise_shape))
        model.add(Reshape(self.initial_reshape_shape))
        model.add(BatchNormalization(axis=self.bn_axis))

        for i in range(self.nb_upconv):
            model.add(UpSampling2D(size=(2, 2)))
            nb_filters = int(self.initial_filters / (2 ** (i + 1)))
            model.add(Conv2D(nb_filters, (3, 3), border_mode="same"))
            model.add(BatchNormalization(axis=1))
            model.add(Activation("relu"))
            model.add(Conv2D(nb_filters, (3, 3), border_mode="same"))            
            model.add(Activation("relu"))


        model.add(Conv2D(self.channels, (3, 3), name="gen_convolution2d_final", border_mode="same", activation='tanh'))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        img = Input(shape=img_shape)

        # First conv
        x = Conv2D(32, (3, 3), strides=(2, 2), name="disc_convolution2d_1", border_mode="same")(img)
        x = BatchNormalization(axis=self.bn_axis)(x)
        x = LeakyReLU(0.2)(x)

        # Next convs
        for i, f in enumerate(self.list_f):
            name = "disc_convolution2d_%s" % (i + 2)
            x = Conv2D(f, (3, 3), strides=(2, 2), name=name, border_mode="same")(x)
            x = BatchNormalization(axis=self.bn_axis)(x)
            x = LeakyReLU(0.2)(x)

        x = Flatten()(x)

        def minb_disc(x):
            diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
            abs_diffs = K.sum(K.abs(diffs), 2)
            x = K.sum(K.exp(-abs_diffs), 2)

            return x

        def lambda_output(input_shape):
            return input_shape[:2]


        M = Dense(self.num_kernels * self.dim_per_kernel, bias=False, activation=None)
        MBD = Lambda(minb_disc, output_shape=lambda_output)

        if self.use_mbd:
            x_mbd = M(x)
            x_mbd = Reshape((self.num_kernels, self.dim_per_kernel))(x_mbd)
            x_mbd = MBD(x_mbd)
            x = concatenate([x, x_mbd])

        x = Dense(1, activation='sigmoid', name="disc_dense_2")(x)

        discriminator_model = Model(input=[img], output=[x], name='discriminator')

        discriminator_model.summary()


        return discriminator_model

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        X_train =  self.load_data()

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                self.save_imgs(epoch,imgs)

    def save_imgs(self, epoch,imgs):
            if not(os.path.exists(self.save_img_folder)):
                os.makedirs(self.save_img_folder)

            r, c = 5, 5
            z = np.random.normal(size=(25, self.latent_dim))
            gen_imgs = self.generator.predict(z)
            gen_imgs = 0.5 * gen_imgs + 0.5

            # z_imgs = self.encoder.predict(imgs)
            # gen_enc_imgs = self.generator.predict(z_imgs)
            # gen_enc_imgs = 0.5 * gen_enc_imgs + 0.5

            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    self.plot(axs[i,j],gen_imgs[cnt, :,:,:].squeeze())
                    cnt += 1
            
            print('----- Saving generated -----')
            if isinstance(epoch, str):
                fig.savefig(self.save_img_folder + "celeba_{}.png".format(epoch))
            else:
                fig.savefig(self.save_img_folder + "celeba_%d.png" % epoch)
            plt.close()


            # fig, axs = plt.subplots(r, c)
            # cnt = 0
            # for i in range(r):
            #     for j in range(c):
            #         self.plot(axs[i,j],gen_enc_imgs[cnt, :,:,:].squeeze())
            #         cnt += 1
            # print('----- Saving encoded -----')
            # if isinstance(epoch, str):
            #     fig.savefig(self.save_img_folder + "celeba_{}_enc.png".format(epoch))
            # else : 
            #     fig.savefig(self.save_img_folder + "celeba_%d_enc.png" % epoch)
            # plt.close()

            fig, axs = plt.subplots(r, c)
            cnt = 0
            imgs = imgs * 0.5 + 0.5
            for i in range(r):
                for j in range(c):
                    self.plot(axs[i,j],imgs[cnt, :,:,:].squeeze())
                    cnt += 1
            
            print('----- Saving real -----')
            if isinstance(epoch, str):
                fig.savefig(self.save_img_folder + "celeba_{}_real.png".format(epoch))
            else : 
                fig.savefig(self.save_img_folder + "celeba_%d_real.png" % epoch)
            plt.close()



    def load_data(self):
        self.dataPath = 'D:\Code\data\sceleba.npy'

        print('----- Loading CelebA -------')
        X_train = np.load(self.dataPath)
        X_train = X_train.transpose([0,2,3,1])
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        print('CelebA shape:', X_train.shape, X_train.min(), X_train.max())
        print('------- CelebA loaded -------')
        
        return X_train
    
    def plot(self, fig, img):
        if self.channels == 1:
            fig.imshow(img,cmap=self.cmap)
            fig.axis('off')
        else:
            fig.imshow(img*255)
            fig.axis('off')


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=50001, batch_size=64, save_interval=100)






