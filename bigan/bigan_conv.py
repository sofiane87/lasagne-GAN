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

import matplotlib.pyplot as plt

import numpy as np
from time import time


class BIGAN():
    def __init__(self):
        self.img_rows = 28 
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'], 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=['binary_crossentropy'], 
            optimizer=optimizer)

        # Build and compile the encoder
        self.encoder = self.build_encoder()
        self.encoder.compile(loss=['binary_crossentropy'], 
            optimizer=optimizer)

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from samples noise
        z = Input(shape=(self.latent_dim, ))
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=self.img_shape)
        z_ = self.encoder(img)

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        self.bigan_generator = Model([z, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            optimizer=optimizer)


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

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            start_time = time()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate img
            z = np.random.normal(size=(half_batch, self.latent_dim)).astype('float32')
            imgs_ = self.generator.predict(z)

            # Select a random half batch of images and encode
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            z_ = self.encoder.predict(imgs)

            valid = np.ones((half_batch, 1)).astype('float32')
            fake = np.zeros((half_batch, 1)).astype('float32')

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Sample gaussian noise
            z = np.random.normal(size=(batch_size, self.latent_dim)).astype('float32')

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            valid = np.ones((batch_size, 1)).astype('float32')
            fake = np.zeros((batch_size, 1)).astype('float32')

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch([z, imgs], [valid, fake])

            end_time = time()
            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f] [time : %.4f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0],end_time-start_time))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                # Select a random half batch of images
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        z = np.random.normal(size=(25, self.latent_dim)).astype('float32')
        gen_imgs = self.generator.predict(z)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("bigan/images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    bigan = BIGAN()
    bigan.train(epochs=40000, batch_size=32, save_interval=400)






