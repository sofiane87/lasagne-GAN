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
import platform
import matplotlib
from skimage.io import imsave
# matplotlib.use('TkAgg')

if not (('alison' in platform.node().lower()) or ('desktop' in platform.node().lower()) or ('sofiane' in platform.node().lower())):
    print('backend changed')
    matplotlib.use('Agg')

print(matplotlib.get_backend())

from matplotlib import pyplot as plt
from keras.models import load_model

import numpy as np
from time import time
import os

import sys
if sys.version_info >= (3, 0):
    raw_input = input

backend_name = K.backend()

is_tf = False
if 'tensorflow' in backend_name.lower():
    is_tf = True


class BIGAN_ROOT(object):
    def __init__(self,example_bool = False, img_rows=28,img_cols=28,channels=1, optimizer = Adam,
                optimizer_dis = Adam,  optimizer_dis_params={'beta_1' : 0.5},
                learningRate=0.00005,optimizer_params = {'beta_1' : 0.5}, test_model = False,
                save_folder='bigan/',interpolate_bool=True,
                interpolate_params = {'n_intp':10,'idx':None,'save_intp_input' : True ,'reload_idx':True,'n_steps' : 10},
                learningRate_dis=0.00005, clip_dis_weight = False,dis_clip_value = 0.2,
                latent_dim = 100,preload=False,train_bool=True):
        self.img_rows =  img_rows 
        self.img_cols =  img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim
        self.test_model = test_model
        self.train_bool = train_bool 
        self.preload = preload or test_model
        self.optimizer_dis_params = optimizer_dis_params
        self.optimizer_params = optimizer_params
        self.optimizer_params['lr'] = learningRate
        self.optimizer_dis_params['lr'] = learningRate_dis

        self.optimizer_dis = optimizer_dis(**self.optimizer_dis_params)
        self.optimizer = optimizer(**self.optimizer_params)
        self.save_model_folder = save_folder + 'saved_model/'
        self.save_img_folder = save_folder + 'images/'
        self.save_intp_input_folder = save_folder + 'intp_input/'
        self.save_intp_folder = save_folder + 'intp/'
        self.interpolate_bool = interpolate_bool
        self.interpolate_params = interpolate_params
        self.batch_index = 0
        self.train_data = None
        self.example = example_bool
        self.save_img_example_folder = save_folder + 'example_images/'

        self.clip_dis_weight = clip_dis_weight
        self.dis_clip_value = dis_clip_value
        if self.channels == 1:
            self.cmap = 'gray'


        # Build and compile the generator
        if self.preload :
            self.model_load()
        else:
            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()            
            self.generator = self.build_generator()
            # Build and compile the encoder
            self.encoder = self.build_encoder()

        print ('disciminator model')
        self.discriminator.summary()
        print ('generator model')
        self.generator.summary()
        print ('encoder model')
        self.encoder.summary()

        self.discriminator.compile(loss=['binary_crossentropy'], 
        optimizer=self.optimizer_dis,
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

    def load_data(self):
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



    def run_interpolation(self,n_intp=10,idx=None,save_intp_input = True,reload_imgs=True,n_steps = 10):
        

        input_images = np.zeros(shape=[2*n_intp,self.img_rows,self.img_cols,self.channels])
        index = 0

        if reload_imgs : 
            images_loaded = self.reload_idx()
            index = min(images_loaded.shape[0],input_images.shape[0])
            input_images[:index] = images_loaded


        while index<2*n_intp:
            sample = self.get_batch(1)
            decoded_sample = self.encode_decode(sample)
            self.display_pair(sample,decoded_sample)
            decision = raw_input("should we keep this image ?\n").lower()
            if 'y' in decision:
                input_images[index] = sample.squeeze()
                index += 1
                print('index {} added to the list'.format(index))
                print('remaining : {}'.format(n_intp*2 - index))
                plt.close()

        if save_intp_input:
            self.save_intp_input(input_images)

        for i in range(index):
            self.interpolate(initial = input_images[2*i:2*i+1],final = input_images[2*i+1:2*i+2],index = i,n_steps = n_steps)


    def interpolate(self,initial,final,index,n_steps=10):
        
        if not(os.path.exists(self.save_intp_folder)):
            os.makedirs(self.save_intp_folder)

        initial_encoded = self.encoder.predict(initial)
        final_encoded = self.encoder.predict(final)
        alphas = np.arange(0,1,1/float(n_steps)).tolist()
        alphas.append(1)

        fig, axs = plt.subplots(1, n_steps+1)
        for i in range(n_steps+1):
            alpha = alphas[i]
            interpolated_encoding = (1-alpha)*initial_encoded + alpha * final_encoded
            interpolated_image = 0.5 * self.generator.predict(interpolated_encoding) + 0.5
            axs[i].imshow(interpolated_image.squeeze(), cmap=self.cmap)
            axs[i].axis('off')

        fig.savefig(self.save_intp_folder + "{}_intp.png".format(index))
        plt.close()



    def generate_examples(self,n_imgs=7):
        

        input_images = np.zeros(shape=[n_imgs,self.img_rows,self.img_cols,self.channels])
        index = 0


        while index<n_imgs:
            sample = self.get_batch(1)
            decoded_sample = self.encode_decode(sample)
            self.display_pair(sample,decoded_sample)
            decision = raw_input("should we keep this image ?\n").lower()
            if 'y' in decision:
                input_images[index] = sample[0]
                index += 1
                print('index {} added to the list'.format(index))
                print('remaining : {}'.format(n_imgs - index))
                plt.close()

        self.save_examples(input_images)


    def save_examples(self,imgs):
        
        if not(os.path.exists(self.save_img_example_folder)):
            os.makedirs(self.save_img_example_folder)

        fig, axs = plt.subplots(2, len(imgs))


        for i in range(len(imgs)):
            real_image = imgs[i]
            encoded_img = self.generator.predict(self.encoder.predict(np.array(imgs[i:i+1]))).squeeze()
            
            real_image = 0.5 + 0.5 * real_image
            encoded_img = 0.5 + 0.5 * encoded_img
            if self.channels == 1:
                axs[0,i].imshow(real_image.squeeze(), cmap=self.cmap)
                axs[1,i].imshow(encoded_img, cmap=self.cmap)
            else:
                axs[0,i].imshow(real_image.squeeze())
                axs[1,i].imshow(encoded_img)


            axs[0,i].axis('off')
            axs[1,i].axis('off')

        fig.savefig(self.save_img_example_folder + "example.png")
        plt.close()



    def reload_intp_images(self):
        if os.path.exists(self.save_intp_input_folder + 'intp_images.npy'):
            idx = np.load(self.save_intp_input_folder + 'intp_images.npy').tolist()
        else:
            idx = []
        return idx

    def save_intp_input(self,intp_input):
        if not(os.path.exists(self.save_intp_input_folder)):
            os.makedirs(self.save_intp_input_folder)

        np.save(self.save_intp_input_folder + 'intp_images',intp_input)


    def encode_decode(self,img):
        encoded_img = self.encoder.predict(img)
        return self.generator.predict(encoded_img)

    def display_pair(self,img,decoded_img):
        img = 0.5 + 0.5*(img)
        decoded_img = 0.5 + 0.5*(decoded_img)
        fig, axs = plt.subplots(1, 2)
        self.plot(axs[0],img.squeeze())
        self.plot(axs[1],decoded_img.squeeze())

        # axs[0].imshow(img.squeeze(), cmap=self.cmap)
        # axs[0].axis('off')
        # axs[1].imshow(decoded_img.squeeze(), cmap=self.cmap)
        # axs[1].axis('off')

        windowmanager = plt.get_current_fig_manager()
        windowmanager.window.wm_geometry("+0+0")
        fig.show()

    def get_batch(self,batch_size=128):
        if self.batch_index == 0 or self.train_data.all() == None:
            self.train_data = self.load_data()

        self.batch_index += 1

        idx = np.random.randint(0, self.train_data.shape[0], batch_size)
        batch_imgs = self.train_data[idx]
        return batch_imgs
 
    def train(self, epochs, batch_size=128, save_interval=50,start_iteration=0):



        half_batch = int(batch_size / 2)

        for epoch in range(start_iteration,start_iteration+epochs):
            start_time = time()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Sample noise and generate img
            z = np.random.normal(size=(half_batch, self.latent_dim)).astype('float32')
            imgs_ = self.generator.predict(z)

            # Select a random half batch of images and encode
            # idx = np.random.randint(0, X_train.shape[0], half_batch)
            # imgs = X_train[idx]
            imgs = self.get_batch(batch_size = half_batch)
            z_ = self.encoder.predict(imgs)

            valid = np.ones((half_batch, 1)).astype('float32')
            fake = np.zeros((half_batch, 1)).astype('float32')

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # if self.clip_dis_weight :
            #     for l in self.discriminator.layers:
            #         weights = l.get_weights()
            #         weights = [np.clip(w, -self.dis_clip_value, self.dis_clip_value) for w in weights]
            #         l.set_weights(weights)



            # ---------------------
            #  Train Generator
            # ---------------------

            # Sample gaussian noise
            z = np.random.normal(size=(batch_size, self.latent_dim)).astype('float32')

            # Select a random half batch of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # imgs = X_train[idx]

            imgs = self.get_batch(batch_size=batch_size)

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
        imgs = self.get_batch(batch_size)
        self.save_imgs('test',imgs)
        print('done...')

    def run(self,epochs=30001, batch_size=32, save_interval=100,start_iteration=0,n_imgs=7):
        
        if  self.train_bool:
            print('training ...')
            self.train(epochs=epochs, batch_size=batch_size, save_interval=save_interval,start_iteration=start_iteration)
            print('done training ...')


        if self.test_model:
            print('test...')
            self.test(batch_size=batch_size)
            print('done test...')
        
        if self.interpolate_bool:
            print('interpolating ...')
            self.run_interpolation(**self.interpolate_params)
            print('interpolation done !')

        if self.example : 
            print('getting examples')
            self.generate_examples(n_imgs=n_imgs)
            print('generation done')

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

        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         self.plot(axs[i,j],gen_imgs[cnt, :,:,:].squeeze())
        #         cnt += 1
        
        # print('----- Saving generated -----')
        # if isinstance(epoch, str):
        #     fig.savefig(self.save_img_folder + "{}_gen.png".format(epoch))
        # else:
        #     fig.savefig(self.save_img_folder + "%d_gen.png" % epoch)
        # plt.close()


        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         self.plot(axs[i,j],gen_enc_imgs[cnt, :,:,:].squeeze())
        #         cnt += 1
        # print('----- Saving encoded -----')
        # if isinstance(epoch, str):
        #     fig.savefig(self.save_img_folder + "{}_enc.png".format(epoch))
        # else : 
        #     fig.savefig(self.save_img_folder + "%d_enc.png" % epoch)
        # plt.close()

        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         self.plot(axs[i,j],imgs[cnt, :,:,:].squeeze())
        #         cnt += 1
        
        # print('----- Saving real -----')
        # if isinstance(epoch, str):
        #     fig.savefig(self.save_img_folder + "{}_real.png".format(epoch))
        # else : 
        #     fig.savefig(self.save_img_folder + "%d_real.png" % epoch)
        # plt.close()


    def plot(self, fig, img):
        # raise NotImplementedError
        if self.channels == 1:
            fig.imshow(img,cmap=self.cmap)
            fig.axis('off')
        else:
            fig.imshow(img)
            fig.axis('off')






if __name__ == '__main__':
    reload_bool = False
    interpolate_bool = False
    preload=False
    bigan = BIGAN_ROOT(test_model = reload_bool,interpolate=interpolate_bool,preload=preload)    
    bigan.run(epochs=30001, batch_size=32, save_interval=100)






