from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Cropping2D
from keras.layers import MaxPooling2D, concatenate, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose as Deconv
from keras.layers.merge import Multiply
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import keras.backend as K

import matplotlib.pyplot as plt

import sys
from time import time

import numpy as np

class WBIGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.latent_dim = 100
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss, 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=self.wasserstein_loss, optimizer=optimizer)

        self.encoder = self.build_encoder()
        self.encoder.compile(loss=self.wasserstein_loss, optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=self.img_shape)
        z_ = self.encoder(img)


        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        self.combined = Model([z, img], [fake, valid])
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer, metrics=['accuracy'])


        # The discriminator takes generated images as input and determines validity
        # valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        # self.combined = Model(z, valid)
        # self.combined.compile(loss=self.wasserstein_loss, 
        #     optimizer=optimizer,
        #     metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        noise_shape = (100,)
        
        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_shape=noise_shape))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(1, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

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
        # model_image = Conv2D(128, kernel_size=3, strides=2, padding="same")(model_image)
        # model_image = LeakyReLU(alpha=0.2)(model_image)
        # model_image = Dropout(0.25)(model_image)
        # model_image = BatchNormalization(momentum=0.8)(model_image)
        # model_image = Conv2D(256, kernel_size=3, strides=1, padding="same")(model_image)
        # model_image = LeakyReLU(alpha=0.2)(model_image)
        # model_image = Dropout(0.25)(model_image)
        
        z_shape = int(np.prod(model_image.shape[1:]))
        model_image = Flatten()(model_image)


        z = Input(shape=(self.latent_dim, ))
        model_z = Dense(z_shape)(z)
        # d_in = concatenate([model_image,model_z,multiply([model_image,model_z])])
        d_in = concatenate([model_image,model_z])

        model = Dense(200)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        # model = Dense(1024)(model)
        # model = LeakyReLU(alpha=0.2)(model)
        # model = Dropout(0.5)(model)
        # model = Dense(1024)(model)
        # model = LeakyReLU(alpha=0.2)(model)
        # model = Dropout(0.5)(model)
        validity = Dense(1, activation="linear")(model)


        return Model([z, img], validity)



    # def build_discriminator(self):

    #     img_shape = (self.img_rows, self.img_cols, self.channels)
        
    #     model = Sequential()

    #     model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Dropout(0.25))
    #     model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    #     model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Dropout(0.25))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Dropout(0.25))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Dropout(0.25))

    #     model.add(Flatten())

    #     model.summary()

    #     img = Input(shape=img_shape)
    #     features = model(img)
    #     valid = Dense(1, activation="linear")(features)

    #     return Model(img, valid)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            start_time = time()

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                z = np.random.normal(size=(half_batch, self.latent_dim)).astype('float32')
                imgs_ = self.generator.predict(z)


                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                imgs = X_train[idx]
                z_ = self.encoder.predict(imgs)

                valid = -np.ones((half_batch, 1)).astype('float32')
                fake = np.ones((half_batch, 1)).astype('float32')

                # Train the discriminator (img -> z is valid, z -> img is fake)
                d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)
                d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


                # idx = np.random.randint(0, X_train.shape[0], half_batch)
                # z_dis = np.random.normal(size=(half_batch, self.latent_dim)).astype('float32')

                # imgs = X_train[idx]

                # noise = np.random.normal(0, 1, (half_batch, 100))

                # # Generate a half batch of new images
                # gen_imgs = self.generator.predict(noise)

                # # Train the discriminator
                # d_loss_real = self.discriminator.train_on_batch(imgs, -np.ones((half_batch, 1)))
                # d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.ones((half_batch, 1)))
                # d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip discriminator weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            z = np.random.normal(0, 1, (batch_size, self.latent_dim))
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            valid = -np.ones((batch_size, 1)).astype('float32')
            fake = np.ones((batch_size, 1)).astype('float32')

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.combined.train_on_batch([z, imgs], [valid, fake])

            end_time = time()

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f] [time : %.3f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0], end_time - start_time))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch,imgs)

    def save_imgs(self, epoch,imgs):
        r, c = 5, 5
        z = np.random.normal(size=(25, self.latent_dim))
        gen_imgs = self.generator.predict(z)
        gen_imgs = 0.5 * gen_imgs + 0.5

        z_imgs = self.encoder.predict(imgs)
        gen_enc_imgs = self.generator.predict(z_imgs)
        gen_enc_imgs = 0.5 * gen_enc_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        
        print('----- Saving generated -----')
        fig.savefig("wbigan/images/mnist_%d.png" % epoch)
        plt.close()


        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_enc_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        print('----- Saving encoded -----')
        fig.savefig("wbigan/images/mnist_%d_enc.png" % epoch)
        plt.close()

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        
        print('----- Saving real -----')
        fig.savefig("wbigan/images/mnist_%d_real.png" % epoch)
        plt.close()



if __name__ == '__main__':
    WBIGAN = WBIGAN()
    WBIGAN.train(epochs=4000, batch_size=32, save_interval=50)






