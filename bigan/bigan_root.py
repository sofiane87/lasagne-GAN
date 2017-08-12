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
from keras.models import load_model

import numpy as np
from time import time
import os


backend_name = K.backend()

is_tf = False
if 'tensorflow' in backend_name.lower():
    is_tf = True


class BIGAN_ROOT(object):
    def __init__(self,img_rows=28,img_cols=28,channels=1, optimizer = Adam, learningRate=0.0002,optimizer_params = {'beta_1' : 0.5}, reload_model = False,save_folder='bigan/'):
        self.img_rows =  img_rows 
        self.img_cols =  img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.reload = reload_model
        self.optimizer_params = optimizer_params
        self.optimizer_params['lr'] = learningRate
        self.optimizer = optimizer(**self.optimizer_params)
        self.save_model_folder = save_folder + 'saved_model/'
        self.save_img_folder = save_folder + 'images/'


        # Build and compile the generator
        if self.reload :
            self.model_load()
        else:
            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()            
            self.generator = self.build_generator()
            # Build and compile the encoder
            self.encoder = self.build_encoder()

        self.discriminator.summary()
        self.generator.summary()
        self.encoder.summary()

        self.discriminator.compile(loss=['binary_crossentropy'], 
        optimizer=self.optimizer,
        metrics=['accuracy'])

        self.generator.compile(loss=['binary_crossentropy'], 
            optimizer=self.optimizer)
        self.encoder.compile(loss=['binary_crossentropy'], 
        optimizer=self.optimizer)


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
            optimizer=self.optimizer)


    def build_encoder(self):
        raise NotImplementedError

    def build_generator(self):
        raise NotImplementedError

    def build_discriminator(self):
        raise NotImplementedError

    def model_save(self):
        if not(os.path.exists(self.save_model_folder)):
            os.makedirs(self.save_model_folder)

        print('--------------- saving model ----------------')
        self.encoder.save(self.save_model_folder + 'encoder.h5')
        self.generator.save(self.save_model_folder + 'generator.h5')
        self.discriminator.save(self.save_model_folder + 'discriminator.h5')
        print('--------------- saving done ----------------')


    def model_load(self):
        print('--------------- loading model ----------------')

        self.encoder = load_model(self.save_model_folder + 'encoder.h5')
        self.generator = load_model(self.save_model_folder + 'generator.h5') 
        self.discriminator = load_model(self.save_model_folder + 'discriminator.h5') 

        print('--------------- loading done ----------------')

    def load_data(self):
        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        return X_train


    def train(self, epochs, batch_size=128, save_interval=50):


        X_train = self.load_data()

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
                self.save_imgs(epoch,imgs)
                self.model_save()

    def test(self,batch_size=128):
        print('testing ...')
        X_train = self.load_data()
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        self.save_imgs('test',imgs)
        print('done...')

    def run(self,epochs=30001, batch_size=32, save_interval=100):
        if not(self.reload) :
            self.train(epochs=epochs, batch_size=batch_size, save_interval=save_interval)
        else:
            self.test(batch_size=batch_size)


    def save_imgs(self, epoch,imgs):
        if not(os.path.exists(self.save_img_folder)):
            os.makedirs(self.save_img_folder)

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
        if isinstance(epoch, str):
            fig.savefig(self.save_img_folder + "mnist_{}.png".format(epoch))
        else:
            fig.savefig(self.save_img_folder + "mnist_%d.png" % epoch)
        plt.close()


        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_enc_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        print('----- Saving encoded -----')
        if isinstance(epoch, str):
            fig.savefig(self.save_img_folder + "mnist_{}_enc.png".format(epoch))
        else : 
            fig.savefig(self.save_img_folder + "mnist_%d_enc.png" % epoch)
        plt.close()

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        
        print('----- Saving real -----')
        if isinstance(epoch, str):
            fig.savefig(self.save_img_folder + "mnist_{}_real.png".format(epoch))
        else : 
            fig.savefig(self.save_img_folder + "mnist_%d_real.png" % epoch)
        plt.close()



if __name__ == '__main__':
    reload_bool = True
    bigan = BIGAN_ROOT(reload_model = reload_bool)    
    bigan.run(epochs=30001, batch_size=32, save_interval=100)






