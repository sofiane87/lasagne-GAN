from __future__ import print_function

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
from keras.models import load_model


import numpy as np
from time import time


class BIGAN():
    def __init__(self,reload_model = False):
        self.img_rows = 28 
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.reload = reload_model
        optimizer = Adam(0.0002, 0.5)



        # Build and compile the generator
        if self.reload :
            self.model_load()
        else:
            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()            

            self.generator = self.build_generator()

            # Build and compile the encoder
            self.encoder = self.build_encoder()

        self.discriminator.compile(loss=['binary_crossentropy'], 
        optimizer=optimizer,
        metrics=['accuracy'])

        self.generator.compile(loss=['binary_crossentropy'], 
            optimizer=optimizer)
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


    # def build_discriminator(self):


    #     img = Input(shape=self.img_shape)
    #     model_image = Conv2D(32, kernel_size=3, strides=2, padding="same")(img)
    #     model_image = LeakyReLU(alpha=0.2)(model_image)
    #     model_image = Dropout(0.25)(model_image)
    #     model_image = Conv2D(64, kernel_size=3, strides=2, padding="same")(model_image)
    #     model_image = ZeroPadding2D(padding=((0,1),(0,1)))(model_image)
    #     model_image = LeakyReLU(alpha=0.2)(model_image)
    #     model_image = Dropout(0.25)(model_image)
    #     model_image = BatchNormalization(momentum=0.8)(model_image)
    #     # model_image = Conv2D(128, kernel_size=3, strides=2, padding="same")(model_image)
    #     # model_image = LeakyReLU(alpha=0.2)(model_image)
    #     # model_image = Dropout(0.25)(model_image)
    #     # model_image = BatchNormalization(momentum=0.8)(model_image)
    #     # model_image = Conv2D(256, kernel_size=3, strides=1, padding="same")(model_image)
    #     # model_image = LeakyReLU(alpha=0.2)(model_image)
    #     # model_image = Dropout(0.25)(model_image)
        
    #     z_shape = int(np.prod(model_image.shape[1:]))
    #     model_image = Flatten()(model_image)


    #     z = Input(shape=(self.latent_dim, ))
    #     model_z = Dense(z_shape)(z)
    #     # d_in = concatenate([model_image,model_z,multiply([model_image,model_z])])
    #     d_in = concatenate([model_image,model_z])

    #     model = Dense(200)(d_in)
    #     model = LeakyReLU(alpha=0.2)(model)
    #     model = Dropout(0.5)(model)
    #     # model = Dense(1024)(model)
    #     # model = LeakyReLU(alpha=0.2)(model)
    #     # model = Dropout(0.5)(model)
    #     # model = Dense(1024)(model)
    #     # model = LeakyReLU(alpha=0.2)(model)
    #     # model = Dropout(0.5)(model)
    #     validity = Dense(1, activation="sigmoid")(model)


    #     return Model([z, img], validity)

    def load_data(self):
        print('---- loading MNIST -----')
        X_train = np.load('/data/users/amp115/skin_analytics/inData/mnist.npy')
        print('----- MNIST loaded ------')
        print x_train.shape, X_train.min(), X_train.max()
        return X_train

    def model_save(self):
        self.encoder.save('bigan/saved_model/encoder.h5')
        self.generator.save('bigan/saved_model/generator.h5')
        self.discriminator.save('bigan/saved_model/discriminator.h5')


    def model_load(self):
        self.encoder = load_model('bigan/saved_model/encoder.h5')
        self.generator = load_model('bigan/saved_model/generator.h5') 
        self.discriminator = load_model('bigan/saved_model/discriminator.h5') 


    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        X_train = self.load_data()

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
                self.save_imgs(epoch,imgs)
                self.model_save()
    def montage(self, image):
        montage=[]
        rows = 10
        cols = 10
        for i in range(rows):
            col = image[i*cols:(i+1)*cols]
            col = np.hstack(col)
            montage.append(col)
        montage = np.vstack(montage)
        return montage

    def save_imgs(self, epoch,imgs):
        if not(os.path.exists(self.save_img_folder)):
            os.makedirs(self.save_img_folder)

        z = np.random.normal(size=(100, self.latent_dim))
        gen_imgs = self.generator.predict(z)
        gen_imgs = 0.5 * gen_imgs + 0.5

        z_imgs = self.encoder.predict(imgs)
        gen_enc_imgs = self.generator.predict(z_imgs)
        gen_enc_imgs = 0.5 * gen_enc_imgs + 0.5

        imgs = imgs * 0.5 + 0.5


        real_montage = self.montage(imgs)
        imsave(self.save_img_folder + '{}_0real.png'.format(epoch),real_montage)
        print('-- Saving real --')

        gen_montage = self.montage(gen_imgs)
        imsave(self.save_img_folder + '{}_2gen.png'.format(epoch),gen_montage)
        print('-- Saving generated --')

        enc_montage = self.montage(gen_enc_imgs)
        imsave(self.save_img_folder + '{}_1enc.png'.format(epoch),enc_montage)
        print('-- Saving encoded --')

        self.train_data = self.load_data()
        trueImgs = self.train_data[0:99]
        img_enc = encode_decode(trueImgs)

        enc_dec_montage = self.montage(img_enc)
        imsave(self.save_img_folder + '{}_3encDec.png'.format(epoch),enc_dec_montage)
        print('-- Saving enc/dec --')


if __name__ == '__main__':
    reload_bool = False
    bigan = BIGAN(reload_model = reload_bool)
    if not(reload_bool):
        bigan.train(epochs=40000, batch_size=32, save_interval=400)






