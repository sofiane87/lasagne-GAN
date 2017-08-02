from __future__ import print_function

from lasagne.layers import InputLayer as Input, DenseLayer as Dense, flatten as Flatten, DropoutLayer as Dropout, reshape as Reshape
#from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from lasagne.layers import batch_norm as BatchNormalization 
#from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from lasagne.layers import concat as concatenate
#from keras.layers import MaxPooling2D, concatenate
from lasagne.nonlinearities import LeakyRectify as lrelu, tanh, sigmoid
#from keras.layers.advanced_activations import LeakyReLU

#from keras.layers.convolutional import UpSampling2D, Conv2D
#from keras.models import Sequential, Model

from lasagne.updates import adam, sgd
# from keras.optimizers import Adam

from lasagne.objectives import binary_crossentropy as bce, binary_accuracy as accuracy
# from keras import losses
# from keras.utils import to_categorical
from lasagne.layers import get_output, get_all_params, get_output_shape, get_all_layers


import theano
from theano import tensor as T
# import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np



class BIGAN():
    def __init__(self):
        self.img_rows = 28 
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.optimizer = adam
        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        # Build and compile the generator
        self.generator = self.build_generator()
        # Build and compile the encoder
        self.encoder = self.build_encoder()

        self.prep_discriminant()
        self.prep_encoder()
        self.prep_generator()

        # # The part of the bigan that trains the discriminator and encoder
        # self.discriminator.trainable = False

        # # Generate image from samples noise
        # z = Input(shape=(self.latent_dim, ))
        # img_ = self.generator(z)

        # # Encode image
        # img = Input(shape=self.img_shape)
        # z_ = self.encoder(img)

        # # Latent -> img is fake, and img -> latent is valid
        # fake = self.discriminator([z, img_])
        # valid = self.discriminator([z_, img])

        # # Set up and compile the combined model
        # self.bigan_generator = Model([z, img], [fake, valid])
        # self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
        #     optimizer=optimizer)

    def prep_discriminant(self):
        z = T.matrix('z')  
        x = T.tensor4('x')
        target = T.matrix('targets')
        
        ### computing the loss
        D_output = get_output(self.discriminator,inputs={self.x_dis:x,self.z_dis:z})
        loss_dis = bce(D_output,target).mean()
        accuracy_dis = accuracy(D_output,target).mean()

        ### preparing the update
        params_d=get_all_params(self.discriminator, trainable=True) 
        grad_d=T.grad(loss_dis,params_d)
        update_D = self.optimizer(grad_d,params_d,learning_rate = self.learning_rate,beta1=self.beta1)

        self.train_batch_discriminant =  theano.function(inputs=[z,x,target], outputs=[loss_dis,accuracy_dis], updates=update_D)



    def prep_encoder(self):
        x = T.tensor4('x')
        target = T.matrix('targets')

        z_enc = get_output(self.encoder, inputs={self.x_enc:x})

        ### computing the loss
        D_output_enc = get_output(self.discriminator,inputs={self.x_dis:x,self.z_dis:z_enc})
        loss_enc = bce(D_output_enc,target).mean()
      
        ### preparing the update
        params_enc=get_all_params(self.encoder, trainable=True) 
        grad_enc=T.grad(loss_enc,params_enc)
        update_E = self.optimizer(grad_enc,params_enc,learning_rate = self.learning_rate,beta1=self.beta1)
        self.train_batch_encoder =  theano.function(inputs=[x,target], outputs=[loss_enc], updates=update_E)

    def prep_generator(self):
        z = T.matrix('z')  
        target = T.matrix('targets')

        x_gen = get_output(self.generator, inputs={self.z_gen:z})

        ### computing the loss
        D_output_gen = get_output(self.discriminator,inputs={self.x_dis:x_gen,self.z_dis:z})
        loss_gen = bce(D_output_gen,target).mean()
        ### preparing the update
       
        params_gen=get_all_params(self.generator, trainable=True) 
        grad_gen=T.grad(loss_gen,params_gen)
        update_G = self.optimizer(grad_gen,params_gen,learning_rate = self.learning_rate,beta1=self.beta1)
        self.train_batch_generator =  theano.function(inputs=[z,target], outputs=[loss_gen], updates=update_G)



    def load_data(self):
        from keras.datasets import mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        return (X_train, Y_train), (X_test, Y_test)

    def build_encoder(self):

        self.x_enc = Input(shape=(None,self.img_shape[0],self.img_shape[1],self.img_shape[2]))
        enc = Flatten(incoming = self.x_enc, outdim = 2)
        #model.add(Flatten(input_shape=self.img_shape))

        enc = BatchNormalization(layer = Dense(incoming = enc, num_units = 512, nonlinearity = lrelu(0.2)),alpha=0.8) 
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        enc = BatchNormalization(layer = Dense(incoming = enc, num_units = 512, nonlinearity = lrelu(0.2)),alpha=0.8) 
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        enc = Dense(incoming = enc, num_units = self.latent_dim, nonlinearity = None)

        # model.add(Dense(self.latent_dim))

        # model.summary()

        # img = Input(shape=self.img_shape)
        # z = model(img)

        # return Model(img, z)
        return enc
    def build_generator(self):
        #model = Sequential()
        self.z_gen = Input(shape=(None,self.latent_dim))
        
        gen = BatchNormalization(layer = Dense(incoming = self.z_gen, num_units = 512, nonlinearity = lrelu(0.2)),alpha=0.2) 
        # model.add(Dense(512, input_dim=self.latent_dim))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        gen = BatchNormalization(layer = Dense(incoming = gen, num_units = 512, nonlinearity = lrelu(0.2)),alpha=0.2) 
       

        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        gen = Dense(incoming = gen, num_units = int(np.prod(self.img_shape)), nonlinearity = tanh)
        # model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        gen = Reshape(incoming=gen, shape=(-1,self.img_shape[0],self.img_shape[1],self.img_shape[2]))
        # model.add(Reshape(self.img_shape))

        # model.summary()

        # z = Input(shape=(self.latent_dim,))
        # gen_img = model(z)

        # return Model(z, gen_img)
        return gen

    def build_discriminator(self):

        self.z_dis = Input(shape=(None,self.latent_dim))
        self.x_dis = Input(shape=(None,self.img_shape[0],self.img_shape[1],self.img_shape[2]))
        
        dis = concatenate(incomings = [self.z_dis, Flatten(incoming = self.x_dis, outdim = 2)], axis=-1, cropping=None)
        dis = Dropout(incoming = Dense(incoming = dis, num_units = 1024, nonlinearity = lrelu(0.2)),p=0.5) 

        # model = Dense(1024)(d_in)
        # model = LeakyReLU(alpha=0.2)(model)
        # model = Dropout(0.5)(model)
        
        dis = Dropout(incoming = Dense(incoming = dis, num_units = 1024, nonlinearity = lrelu(0.2)),p=0.5) 
 
        # model = Dense(1024)(model)
        # model = LeakyReLU(alpha=0.2)(model)
        # model = Dropout(0.5)(model)
        
        dis = Dropout(incoming = Dense(incoming = dis, num_units = 1024, nonlinearity = lrelu(0.2)),p=0.5) 

        # model = Dense(1024)(model)
        # model = LeakyReLU(alpha=0.2)(model)
        # model = Dropout(0.5)(model)

        dis = Dense(incoming = dis, num_units = 1, nonlinearity = sigmoid)

        # validity = Dense(1, activation="sigmoid")(model)

        # return Model([z, img], validity)

        return dis

    def train_batch_bigan(self,z, imgs,valid, fake):
        enc_loss = self.train_batch_encoder(imgs,fake)    
        gen_loss = self.train_batch_generator(z,valid)    
        g_loss = 0.5 * np.add(enc_loss,gen_loss)
        return g_loss
    def predict_generator(self,z):
        return get_output(self.generator, inputs={self.z_gen:z}).eval()

    def predict_encoder(self,x):
        return get_output(self.encoder, inputs={self.x_enc:x}).eval()


    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = self.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):


            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate img
            z = np.random.normal(size=(half_batch, self.latent_dim))
            imgs_ = self.predict_generator(z)

            # Select a random half batch of images and encode
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            z_ = self.predict_encoder(imgs)

            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real, d_acc_real = self.train_batch_discriminant(z_, imgs, valid)
            d_loss_fake, d_acc_fake = self.train_batch_discriminant(z, imgs_, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_acc = 0.5 * np.add(d_acc_real, d_acc_fake)
            # ---------------------
            #  Train Generator
            # ---------------------

            # Sample gaussian noise
            z = np.random.normal(size=(batch_size, self.latent_dim))

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.train_batch_bigan(z, imgs,valid, fake)

            # Plot the progress
            print ("{} [D loss: {}, acc: {:.2f}%] [G loss: {}]".format(epoch, d_loss, d_acc, g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                # Select a random half batch of images
                self.save_imgs(epoch,imgs)

    # def save_imgs(self, epoch):
    #     r, c = 5, 5
    #     z = np.random.normal(size=(25, self.latent_dim))
    #     gen_imgs = self.predict_generator(z)

    #     gen_imgs = 0.5 * gen_imgs + 0.5

    #     fig, axs = plt.subplots(r, c)
    #     cnt = 0
    #     for i in range(r):
    #         for j in range(c):
    #             axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
    #             axs[i,j].axis('off')
    #             cnt += 1
    #     fig.savefig("bigan/images/mnist_%d.png" % epoch)
    #     plt.close()

    def save_imgs(self, epoch,imgs):
        r, c = 5, 5
        z = np.random.normal(size=(25, self.latent_dim))
        gen_imgs = self.predict_generator(z)
        gen_imgs = 0.5 * gen_imgs + 0.5

        z_imgs = self.predict_encoder(imgs)
        gen_enc_imgs = self.predict_generator(z_imgs)
        gen_enc_imgs = 0.5 * gen_enc_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("bigan/images/mnist_%d.png" % epoch)
        plt.close()


        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_enc_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("bigan/images/mnist_%d_enc.png" % epoch)
        plt.close()

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("bigan/images/mnist_%d_real.png" % epoch)
        plt.close()

if __name__ == '__main__':
    bigan = BIGAN()
    bigan.train(epochs=1, batch_size=32, save_interval=400)






