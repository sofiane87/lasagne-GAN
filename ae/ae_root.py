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

if not ('alison' in platform.node().lower()) or ('desktop' in platform.node().lower()):
    print('backend changed')
    matplotlib.use('Agg')

print(matplotlib.get_backend())

from matplotlib import pyplot as plt
from keras.models import load_model

import numpy as np
from time import time
import os


backend_name = K.backend()

is_tf = False
if 'tensorflow' in backend_name.lower():
    is_tf = True


class AE_ROOT(object):
    def __init__(self,img_rows=28,img_cols=28,channels=1, optimizer = Adam,
                learningRate=0.00005,optimizer_params = {'beta_1' : 0.5}, test_model = False,
                save_folder='ae/',interpolate_bool=True,
                interpolate_params = {'n_intp':10,'idx':None,'save_intp_input' : True ,'reload_idx':True,'n_steps' : 10},
                latent_dim = 100,preload=False,train_bool=True, train_gen = False):
        self.img_rows =  img_rows 
        self.img_cols =  img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim
        self.test_model = test_model
        self.train_bool = train_bool 
        self.preload = preload or test_model
        self.optimizer_params = optimizer_params
        self.optimizer_params['lr'] = learningRate

        self.optimizer = optimizer(**self.optimizer_params)
        self.save_model_folder = save_folder + 'saved_model/'
        self.save_img_folder = save_folder + 'images/'
        self.save_intp_input_folder = save_folder + 'intp_input/'
        self.save_intp_folder = save_folder + 'intp/'
        self.interpolate_bool = interpolate_bool
        self.interpolate_params = interpolate_params
        self.batch_index = 0
        self.train_data = None
        self.train_gen = train_gen

        if self.channels == 1:
            self.cmap = 'gray'


        # Build and compile the generator
        if self.preload :
            self.model_load()
        else:
            # Build and compile the discriminator
            if self.train_gen:
                self.decoder = self.build_decoder()
            else:
                self.model_load(enc_or_dec=0)
            # Build and compile the encoder
            self.encoder = self.build_encoder()

        print ('decoder model')
        self.decoder.summary()
        print ('encoder model')
        self.encoder.summary()


        self.decoder.compile(loss=['binary_crossentropy'], 
            optimizer=self.optimizer)
        self.encoder.compile(loss=['binary_crossentropy'], 
        optimizer=self.optimizer)


        # The part of the bigan that trains the discriminator and encoder


        self.decoder.trainable = self.train_gen


        # Encode image
        img = Input(shape=self.img_shape)
        enc_img = self.encoder(img)
        dec_z = self.decoder(enc_img)


        # Set up and compile the combined model
        self.generator = Model([img], [dec_z])
        self.generator.compile(loss=['binary_crossentropy'],
            optimizer=self.optimizer,metrics=['accuracy'])


    def build_encoder(self):
        raise NotImplementedError

    def build_decoder(self):
        raise NotImplementedError


    def load_data(self):
        raise NotImplementedError

    def model_save(self):
        if not(os.path.exists(self.save_model_folder)):
            os.makedirs(self.save_model_folder)

        print('--------------- saving model ----------------')
        self.encoder.save(self.save_model_folder + 'encoder.h5')
        self.decoder.save(self.save_model_folder + 'decoder.h5')
        print('--------------- saving done ----------------')


    def model_load(self,enc_or_dec = 1):
        print('--------------- loading model ----------------')

        if enc_or_dec>= 1 :
            self.encoder = load_model(self.save_model_folder + 'encoder.h5')
        if enc_or_dec <= 1 :
            self.decoder = load_model(self.save_model_folder + 'decoder.h5') 

        print('--------------- loading done ----------------')



    def run_interpolation(self,n_intp=10,idx=None,save_intp_input = True,reload_imgs=True,n_steps = 10):
        

        input_images = np.zeros(shape=[2*n_intp,self.img_rows,self.img_cols,self.channels])
        index = 0


        if os.path.exists(self.save_intp_input_folder + 'intp_images.npy') and reload_imgs:
            images_loaded = np.load(self.save_intp_input_folder + 'intp_images.npy').tolist()
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

        imgs_montage = []
        for i in range(n_steps+1):
            alpha = alphas[i]
            interpolated_encoding = (1-alpha)*initial_encoded + alpha * final_encoded
            interpolated_image = 0.5 * self.decoder.predict(interpolated_encoding) + 0.5
            imgs_montage.append(interpolated_image)

        self.save_montage(imgs_montage,rows=1,cols=n_steps+1)


    # def reload_intp_images(self):
    #     if os.path.exists(self.save_intp_input_folder + 'intp_images.npy'):
    #         idx = np.load(self.save_intp_input_folder + 'intp_images.npy').tolist()
    #     else:
    #         idx = []
    #     return idx

    def save_intp_input(self,intp_input):
        if not(os.path.exists(self.save_intp_input_folder)):
            os.makedirs(self.save_intp_input_folder)

        np.save(self.save_intp_input_folder + 'intp_images',intp_input)


    def encode_decode(self,img):
        return self.generator.predict(img)

    def display_pair(self,img,decoded_img):
        img = self.normalize(images)
        decoded_img = self.normalize(decoded_img)
        fig, axs = plt.subplots(1, 2)
        self.plot(axs[0],img.squeeze())
        self.plot(axs[1],decoded_img.squeeze())

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


            # Select a random half batch of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # imgs = X_train[idx]

            imgs = self.get_batch(batch_size=batch_size)

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.generator.train_on_batch([imgs], [imgs])

            end_time = time()
            # Plot the progress
            print ("[acc: %.2f%%] [loss: %f] [time : %.2fs]" % (epoch, 100*g_loss[1], g_loss[0],end_time-start_time))

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

    def run(self,epochs=30001, batch_size=32, save_interval=100,start_iteration=0):
        
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

    def montage(self, image,rows = 5,cols=5):
        if image.shape[0] < rows * cols : 
            raise 'not Enough Images provided for montage'

        montage=[]
        image = image.squeeze()
        for i in range(rows):
            col = image[i*cols:(i+1)*cols]
            col = np.hstack(col)
            montage.append(col)
        montage = np.vstack(montage)
        return montage

    def save_montage(self,imgs,name,rows=5,cols=5):
        real_montage = self.normalize(self.montage(imgs,rows=rows,cols=cols))
        imsave(self.save_img_folder + '{}.png'.format(name),real_montage)

    def normalize(self,arr):
        return 0.5 * arr + 0.5

    def save_imgs(self, epoch,imgs):
        
        if not(os.path.exists(self.save_img_folder)):
            os.makedirs(self.save_img_folder)

        gen_enc_imgs = self.generator.predict(imgs)

        self.save_montage(imgs, '{}_real.png'.format(epoch))
        print('-- Saving real --')

        self.save_montage(gen_imgs, '{}_gen.png'.format(epoch))
        print('-- Saving generated --')



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
        raise NotImplementedError
        # if self.channels == 1:
        #     fig.imshow(img,cmap=self.cmap)
        #     fig.axis('off')
        # else:
        #     fig.imshow(img)
        #     fig.axis('off')






if __name__ == '__main__':
    reload_bool = False
    interpolate_bool = False
    preload=False
    bigan = AE_ROOT(test_model = reload_bool,interpolate=interpolate_bool,preload=preload)    
    bigan.run(epochs=30001, batch_size=32, save_interval=100)






