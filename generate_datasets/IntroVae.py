# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:13:25 2020

@author: MrVicente
"""

import data_access

import tensorflow as tf
from tensorflow.keras.layers import Dense,Reshape,Conv2D,Flatten,UpSampling2D,LeakyReLU,Cropping2D,AveragePooling2D,BatchNormalization,ReLU,InputLayer,Activation
from tensorflow.keras.optimizers import Adam

import numpy as np
import time
import os


class ResidualBlock(tf.keras.Model):
    def __init__(self, input_channels, out_channels, stride=(1,1), activation=None):
        super().__init__(name = "Residual_Block")
        if(input_channels != out_channels):
            self.conv_expand = Conv2D(out_channels, (1,1), padding='same', use_bias=False)
            #self.norm_expand = BatchNormalization() 
            #self.act = ReLU()
        else:
            self.conv_expand = None
        self.activation = activation
        self.conv_1 = Conv2D(out_channels,(3,3), padding='same',use_bias=False)
        self.batch_norm_1 = BatchNormalization()
        self.leaky_1 = ReLU()
        self.conv_2 = Conv2D(out_channels,(3,3), padding='same',use_bias=False)
        self.batch_norm_2 = BatchNormalization()
        self.leaky_2 = ReLU()
        
    def call(self, input_tensor, training = True):
        if(self.conv_expand is not None):
            residual = self.conv_expand(input_tensor)
            #residual = self.norm_expand(residual)
            #residual = self.act(residual)
        else:
            residual = input_tensor
        out = self.conv_1(input_tensor)
        out = self.batch_norm_1(out, training = training)
        out = self.leaky_1(out)
        out = self.conv_2(out)
        out = self.batch_norm_2(out, training = training)
        out += residual
        
        if self.activation is None:
            out = self.leaky_2(out)
        else:
            out = Activation(self.activation)(out)
        
        return out

class Generator(tf.keras.Model):
    def __init__(self, latent_dim = 256, batch_size = 64, channels=[32,64,64,128,128]):
        super().__init__(name = "Generator")
        cc = channels[-1]
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.inp = InputLayer(input_shape=(self.latent_dim,))
        self.dense_1 = Dense( 8* batch_size * 5 * 5, name = 'Generator_Dense_1')
        self.relu = ReLU()
        self.reshape_1 = Reshape((5,5, 8 * batch_size))
        self.reses = list()
        for ch in reversed(channels[:-1]):
            self.reses.append([ResidualBlock(cc,ch),UpSampling2D()])
            cc = ch
        self.res_block_n = ResidualBlock(cc,cc)
        self.toRGB = Conv2D(3, (3,3),activation = 'tanh',padding='same', use_bias=False,name = 'Generator_To_RGB')
        self.optimizer = Adam(learning_rate=0.0002,beta_1=0.5,beta_2=0.9)
        
    def call(self, input_tensor, training = True):
        x = self.inp(input_tensor)
        x = self.dense_1(x)
        x = self.relu(x)
        x = self.reshape_1(x)
        for i in range(len(self.reses)):
            x = self.reses[i][0](x, training = training)
            x = self.reses[i][1](x, training = training)
            if(i == 1):
               x = Cropping2D(cropping=((1,0),(1,0)))(x)
        x = self.res_block_n(x, training = training)
        x = self.toRGB(x)
        return x

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def set_seed(self):
        self.seed = tf.random.normal([self.batch_size, self.latent_dim])
        data_access.store_seed_in_file('seed',self.seed)

    def load_seed(self):
        self.seed = data_access.load_seed_from_file('seed')
   
    
class Encoder(tf.keras.Model):
    def __init__(self, latent_dim = 256, channels=[32,64,64,128,128]):
        super().__init__(name = "Encoder")
        self.conv_1 = Conv2D(32, (3,3), padding='same', use_bias=False, input_shape = (152,152,3))
        self.batch_norm_1 = BatchNormalization()
        self.latent_dim = latent_dim
        self.leaky_1 = LeakyReLU(alpha=0.2)
        self.downsampling = AveragePooling2D()
        self.reses = list()
        cc = channels[0]
        for ch in channels[1:]:
            self.reses.append([ResidualBlock(cc,ch),AveragePooling2D()])
            cc = ch
        self.res_block_n = ResidualBlock(cc,cc,activation = 'linear')
        self.flat = Flatten()
        self.dense = Dense(self.latent_dim * 2)
        self.optimizer = Adam(learning_rate=0.0002,beta_1=0.5,beta_2=0.9)
        
    def call(self, input_tensor, training = True):
        out = self.conv_1(input_tensor)
        out = self.batch_norm_1(out, training = training)
        out = self.leaky_1(out)
        out = self.downsampling(out)
        for i in range(len(self.reses)):
            out = self.reses[i][0](out, training = training)
            out = self.reses[i][1](out, training = training)
        out = self.res_block_n(out, training = training)
        out = self.flat(out)
        out = self.dense(out)
        return out 

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
   
class IntroVae(tf.keras.Model):
    def __init__(self, params = None, channels=[16,32,64,128,256,512]):
        super().__init__(name = "Intro_Vae")

        if params is None: 
            params = {
              'batchsz': 64,
              'z_dim': 256,
              'epochs': 20,
              'm': 110,
              'alpha': 0.25,
              'beta': 0.5,
              'gamma': 1.0,
              'lr': 0.0002
            }
        self.params = params
   
        self.latent_dim = params['z_dim']
        self.set_learning_constants(m = params['m'], alpha = params['alpha'], beta = params['beta'])
        self.inference_net = Encoder(latent_dim = self.latent_dim, channels = channels)
        self.generative_net = Generator(latent_dim = self.latent_dim, channels = channels)
        
        self.prepare_seed()
        #self.load_weights()

        self.train_dataset = None
    
    def set_learning_constants(self, m=110, alpha=0.25, beta=0.5):
        self.m = m
        self.alpha = alpha
        self.beta = beta
        
    def prepare_seed(self):
        if('seed.npz' not in os.listdir('.')):
            self.generative_net.set_seed()
        else :
            self.generative_net.load_seed()
            
    def load_weights(self):
        if ('weights' in os.listdir('.')):
            self.inference_net.load_weights('weights/enc_weights/enc_weights')
            self.generative_net.load_weights('weights/g_weights/g_weights')
        
    def load_dataset(self,dataset):
        """
        Load images as numpy vectors and store dataset number of classes
        """
        self.train_dataset,_,_,_ = dataset
        print('Dataset loaded')
        
    
    def sample(self, eps=None):
        if eps is None:
          eps = tf.random.normal(shape=(self.params['batchsz'], self.latent_dim))
        return self.decode(eps, train = False)

    @tf.function   
    def encode(self, x, train = True):
        mean, logvar = tf.split(self.inference_net(x, training = train), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def decode(self, z, train = True):
        logits = self.generative_net(z, training = train)
        return logits

    @tf.function
    def kl_loss(self,mean, logvar):
        kl_div = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + logvar - tf.square(mean)- tf.exp(logvar), axis=-1))
        return kl_div
    @tf.function
    def L_AE(self,gen,real):
        mse = tf.reduce_mean(tf.square(tf.subtract(real,gen)))
        return mse

    @tf.function    
    def compute_loss_inf(self, z, mean, logv , z_r, mean_r, logv_r, z_pp, mean_pp, logv_pp, gen, real):
        #regr_ng = tf.math.maximum(0.0, self.m - self.kl_loss(mean_r, logv_r)) -> same as bellow, paper equation
        regr_ng = tf.keras.activations.relu(self.m - self.kl_loss(mean_r, logv_r))
        #regpp_ng = tf.math.maximum(0.0, self.m - self.kl_loss(mean_pp, logv_pp)) -> same as bellow, paper equation
        regpp_ng = tf.keras.activations.relu(self.m - self.kl_loss(mean_pp, logv_pp))
        regs_ng = self.alpha * (regr_ng + regpp_ng)
        reg_ae = self.kl_loss(mean, logv)
        #tf.print('regr_ng: ', regr_ng)
        #tf.print('regpp_ng: ', regpp_ng)
        #tf.print('reg_ae: ',reg_ae)
        return reg_ae + regs_ng + (self.beta * self.L_AE(gen,real))

    @tf.function
    def compute_loss_gen(self, z_r, mean_r, logv_r, z_pp, mean_pp, logv_pp, gen, real):
        regr = self.kl_loss(mean_r, logv_r)
        regpp = self.kl_loss(mean_pp, logv_pp)
        regs_adv = self.alpha * (regr + regpp)
        #tf.print('regs_adv: ',regs_adv)
        #tf.print('MSE: ',self.beta * self.L_AE(gen,real))
        return regs_adv + (self.beta * self.L_AE(gen,real))

    @tf.function
    def compute_apply_gradients(self,x):
        with tf.GradientTape(persistent=True) as tape:
            #tf.print('--------start-------')
            mean, logvar = self.encode(x, train = True)
            z = self.reparameterize(mean, logvar)
            x_r = self.decode(z, train = True)
            z_prior = tf.random.normal(shape=mean.shape)
            x_p = self.decode(z_prior, train = True)
            #tf.print('mean: ',tf.reduce_mean(mean))
            #tf.print('logvar: ',tf.reduce_mean(logvar))
            #tf.print('x_logit: ',tf.reduce_mean(x_r))
            #tf.print('z_prior: ',tf.reduce_mean(z_prior))
            #tf.print('x_p: ',tf.reduce_mean(x_p))

            #tf.print('-------inf--------')
            mean_r, logvar_r = self.encode(x_r, train = False) # ng(.) no training
            z_r =  self.reparameterize(mean_r, logvar_r)

            #tf.print('mean_r: ',tf.reduce_mean(mean_r))
            #tf.print('logvar_r: ',tf.reduce_mean(logvar_r))

            mean_pp, logvar_pp = self.encode(x_p, train = False) # ng(.) no training
            z_pp =  self.reparameterize(mean_pp, logvar_pp)

            #tf.print('mean_pp: ',tf.reduce_mean(mean_pp))
            #tf.print('logvar_pp: ',tf.reduce_mean(logvar_pp))

            inf_loss = self.compute_loss_inf(z, mean, logvar, z_r, mean_r , logvar_r , z_pp, mean_pp, logvar_pp, x_r, x)

            #tf.print('--------gen-------')
            mean_r, logvar_r = self.encode(x_r, train = True)
            z_r =  self.reparameterize(mean_r, logvar_r) 
            mean_pp, logvar_pp = self.encode(x_p, train = True)
            z_pp =  self.reparameterize(mean_pp, logvar_pp)
            
            gen_loss = self.compute_loss_gen(z_r, mean_r,logvar_r, z_pp, mean_pp, logvar_pp, x_r, x)

        gradients_of_inf = tape.gradient(inf_loss, self.inference_net.trainable_variables)
        gradients_of_gen = tape.gradient(gen_loss, self.generative_net.trainable_variables)
        self.inference_net.backPropagate(gradients_of_inf, self.inference_net.trainable_variables)
        self.generative_net.backPropagate(gradients_of_gen, self.generative_net.trainable_variables)
        return inf_loss, gen_loss

    def generate_real_samples(self, n_samples):
        # choose random instances
        ix = np.random.randint(0, self.train_dataset.shape[0], n_samples)
        # select images
        X = self.train_dataset[ix]
        #convert X to tensor
        return tf.convert_to_tensor(X.astype(np.float32))
       
    def train_model(self,epochs = None ,batch_size = None, images_per_epoch=4,directory='imgs'):
        if epochs is None: epochs = self.params['epochs']
        if batch_size is None: batch_size = self.params['batchsz']

        batch_per_epoch = int(self.train_dataset.shape[0] / batch_size)
        # calculate the number of training iterations
        n_steps = batch_per_epoch * epochs

        start_time = time.time()
        try:
            epoch = int(open('current_epoch.txt').read())
        except:
            epoch = 0


        for step_i in range(n_steps):
            train_x = self.generate_real_samples(batch_size)
            inf_loss, gen_loss = self.compute_apply_gradients(train_x)
            data_access.print_training_output_vae(step_i,n_steps,inf_loss,gen_loss)
            if((step_i % (n_steps / epochs)) == 0):
                epoch += 1
                gen_images = self.sample(self.generative_net.seed)
                data_access.store_images_seed(directory,gen_images[:images_per_epoch],epoch)
                
                #self.generative_net.save_weights('weights/g_weights/g_weights',save_format='tf')
                #self.inference_net.save_weights('weights/enc_weights/enc_weights',save_format='tf')
                data_access.write_current_epoch(filename='current_epoch',epoch=epoch)
        end_time = time.time()
        data_access.print_training_time(start_time,end_time,self.params)

    def generate_images(self,number_of_samples=5,directory="imgs"):
        random_vector_for_generation = tf.random.normal(shape=[number_of_samples, self.latent_dim])
        predictions = self.sample(random_vector_for_generation)
        data_access.prepare_directory(directory)
        for i in range(predictions.shape[0]):
            data_access.store_image(directory,None,predictions[i],i,'vae')
       
    def generate_image(self, z):
        return self.sample(z)

    def get_latent_code(self, image):
        image = data_access.preprocess_image(image)
        mean, logvar = self.encode(image,training=False)
        l_code = self.reparameterize(mean, logvar)
        return l_code
