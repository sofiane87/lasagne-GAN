from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys
import os

import numpy as np

class DCGAN():
    def __init__(self):
        self.img_rows = 64 
        self.img_cols = 64
        self.channels = 3
        self.save_img_folder = 'dcgan/images/'
        optimizer = Adam(0.0002, 0.5)
        self.latent_dim = 200
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(200,))
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

        noise_shape = (200,)
        
        model = Sequential()

        model.add(Dense(512 * 4 * 4, activation="relu", input_shape=noise_shape))
        model.add(Reshape((4, 4, 512)))
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
        model.add(Conv2D(3, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
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
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

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
            noise = np.random.normal(0, 1, (half_batch, 200))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 200))

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
                fig.savefig(self.save_img_folder + "mnist_{}.png".format(epoch))
            else:
                fig.savefig(self.save_img_folder + "mnist_%d.png" % epoch)
            plt.close()


            # fig, axs = plt.subplots(r, c)
            # cnt = 0
            # for i in range(r):
            #     for j in range(c):
            #         self.plot(axs[i,j],gen_enc_imgs[cnt, :,:,:].squeeze())
            #         cnt += 1
            # print('----- Saving encoded -----')
            # if isinstance(epoch, str):
            #     fig.savefig(self.save_img_folder + "mnist_{}_enc.png".format(epoch))
            # else : 
            #     fig.savefig(self.save_img_folder + "mnist_%d_enc.png" % epoch)
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
                fig.savefig(self.save_img_folder + "mnist_{}_real.png".format(epoch))
            else : 
                fig.savefig(self.save_img_folder + "mnist_%d_real.png" % epoch)
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
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)






