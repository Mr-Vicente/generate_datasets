# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:13:25 2020

@author: MrVicente
"""

import data_access

import tensorflow as tf
from tensorflow.keras.layers import Dense,Reshape,Conv2D,Flatten,UpSampling2D,LeakyReLU,Cropping2D,AveragePooling2D,BatchNormalization,ZeroPadding2D,ReLU,InputLayer
from tensorflow.keras.optimizers import Adam

import numpy as np
import time

class ResidualBlock(tf.keras.Model):
    def __init__(self, input_channels, out_channels, layer_n = -1,  stride=(1,1), downsample=None):
        super().__init__(name = "Residual_Block")
        if(input_channels != out_channels):
            self.conv_expand = Conv2D(out_channels, (1,1), padding='same', kernel_initializer='he_normal')
        else:
            self.conv_expand = None
        
        self.conv_1 = Conv2D(out_channels,(4,4), padding='same', kernel_initializer='he_normal')
        self.batch_norm_1 = BatchNormalization()
        self.leaky_1 = LeakyReLU(alpha=0.2)
        self.conv_2 = Conv2D(out_channels,(4,4), padding='same', kernel_initializer='he_normal')
        self.batch_norm_2 = BatchNormalization()
        self.leaky_2 = LeakyReLU(alpha=0.2)
        
    def call(self, input_tensor):
        if(self.conv_expand is not None):
            residual = self.conv_expand(input_tensor)
        else:
            residual = input_tensor
        out = input_tensor
        out = self.conv_1(input_tensor)
        out = self.batch_norm_1(out)
        out = self.leaky_1(out)
        out = self.conv_2(out)
        out = self.batch_norm_2(out)
        out += residual
        out = self.leaky_2(out)
        return out
    
class Generator(tf.keras.Model):
    def __init__(self, latent_dim = 128, channels=[128,128,128,128,128]):
        super().__init__(name = "Generator")
        cc = channels[-1]
        self.inp = InputLayer(input_shape=(latent_dim,))
        self.dense_1 = Dense(latent_dim * 5 * 5, name = 'Generator_Dense_1')
        self.relu = ReLU()
        self.reshape_1 = Reshape((5,5,latent_dim))
        self.reses = list()
        count = 0
        for ch in channels[::-1]:
            self.reses.append([ResidualBlock(cc,ch,count),UpSampling2D()])
            cc = ch
            count += 1
        self.res_block_n = ResidualBlock(cc,cc)
        self.toRGB = Conv2D(3, (5,5), padding='same', kernel_initializer='he_normal', name = 'Generator_To_RGB')
        self.optimizer = Adam(learning_rate=0.0002,beta_1=0.9,beta_2=0.999)
        
    def call(self, input_tensor):
        x = self.inp(input_tensor)
        x = self.dense_1(x)
        x = self.relu(x)
        x = self.reshape_1(x)
        for i in range(len(self.reses)):
            x = self.reses[i][0](x)
            x = self.reses[i][1](x)
            if(i == 1):
               x = Cropping2D(cropping=((1,0),(1,0)))(x)
        x = self.toRGB(x)
        return x

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    
class Encoder(tf.keras.Model):
    def __init__(self, channels=[128,128,128,128,128]):
        super().__init__(name = "Encoder")
        self.conv_1 = Conv2D(128, (4,4), padding='same', kernel_initializer='he_normal', input_shape = (152,152,3))
        self.batch_norm_1 = BatchNormalization()
        self.leaky_1 = LeakyReLU(alpha=0.2)
        self.downsampling = AveragePooling2D()
        self.reses = list()
        cc = channels[0]
        for ch in channels[1:]:
            self.reses.append([ResidualBlock(cc,ch),AveragePooling2D()])
            cc = ch
        self.res_block_n = ResidualBlock(cc,cc)
        self.flat = Flatten()
        self.optimizer = Adam(learning_rate=0.0002,beta_1=0.9,beta_2=0.999)
        
    def call(self, input_tensor):
        out = self.conv_1(input_tensor)
        out = self.batch_norm_1(out)
        out = self.leaky_1(out)
        out = self.downsampling(out)
        for i in range(len(self.reses)):
            out = self.reses[i][0](out)
            out = self.reses[i][1](out)
        out = self.res_block_n(out)
        out = self.flat(out)
        return out

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
       
class IntroVae(tf.keras.Model):
    def __init__(self, latent_dim = 512, channels=[128,128,128,128,128]):
        super().__init__(name = "Intro_Vae")
        
        self.latent_dim = latent_dim
        self.inference_net = Encoder(channels)
        self.generative_net = Generator(self.latent_dim,channels)
        
        self.train_dataset = None
        
    def load_dataset(self,dataset):
        """
        Load images as numpy vectors and store dataset number of classes
        """
        self.train_dataset,_,_,_ = dataset
        print('Dataset loaded')
        
    
    @tf.function
    def sample(self, eps=None):
        if eps is None:
          eps = tf.random.normal(shape=(8, self.latent_dim))
        return self.decode(eps, train = False, apply_sigmoid=True)

    @tf.function   
    def encode(self, x, train = True):
        mean, logvar = tf.split(self.inference_net(x, training = train), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        print(mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def decode(self, z, train = True, apply_sigmoid=False):
        logits = self.generative_net(z, training = train)
        if apply_sigmoid:
          probs = tf.sigmoid(logits)
          return probs
        return logits

    #@tf.function
    def kl_loss(self,mean, logvar):
        #a = tf.clip_by_value(tf.exp(logvar), 1e-8, 1-1e-8)
        return tf.reduce_mean(-0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1))
    @tf.function
    def L_AE(self,gen,real):
        return tf.reduce_mean(tf.keras.losses.mean_squared_error(real,gen))

    def log_normal_pdf(self,sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)
        
    def compute_loss_inf(self, z, logv_z, z_r, logv_z_r, z_pp, logv_z_pp, gen, real):
        beta = 0.05
        alpha = 0.25
        m = 120
        eq = alpha * ((m - self.kl_loss(z_r, logv_z_r)) + (m - self.kl_loss(z_pp, logv_z_pp)))
        return self.kl_loss(z,logv_z) + eq + (beta * self.L_AE(gen,real))

    @tf.function
    def compute_loss_gen(self, mean_r, logv_r, mean_pp, logv_pp, gen, real):
        beta = 0.05
        alpha = 0.25
        x = alpha * (self.kl_loss(mean_r,logv_r) + self.kl_loss(mean_pp, logv_pp))
        return x + (beta * self.L_AE(gen,real))

    #@tf.function
    def compute_apply_gradients(self,x):
        with tf.GradientTape(persistent=True) as tape:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            x_logit = self.decode(z)
            z_prior = tf.random.normal(shape=mean.shape)
            x_p = self.decode(z_prior, train = True)

            mean_r, logvar_r = self.encode(x_logit, train = False)
            z_r =  self.reparameterize(mean_r, logvar_r)
            mean_pp, logvar_pp = self.encode(x_p, train = False)
            z_pp =  self.reparameterize(mean_pp, logvar_pp)

            inf_loss = self.compute_loss_inf(mean, logvar, mean_r , logvar_r , mean_pp, logvar_pp, x_logit, x)

            mean_r, logvar_r = self.encode(x_logit, train = True)
            z_r =  self.reparameterize(mean_r, logvar_r) 
            mean_pp, logvar_pp = self.encode(x_p, train = True)
            z_pp =  self.reparameterize(mean_pp, logvar_pp)
            
            gen_loss = self.compute_loss_gen(mean_r,logvar_r, mean_pp, logvar_pp, x_logit, x)
            #gen_loss = self.compute_loss_gen(z_r, z_pp, x_logit, x)

        gradients_of_inf = tape.gradient(inf_loss, self.inference_net.trainable_variables)
        gradients_of_gen = tape.gradient(gen_loss, self.generative_net.trainable_variables)
        self.inference_net.backPropagate(gradients_of_inf, self.inference_net.trainable_variables)
        self.generative_net.backPropagate(gradients_of_gen, self.generative_net.trainable_variables)
        return inf_loss, gen_loss
        
    def train_model(self,epochs):
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            steps = 5000 / 8
            counter = 0
            for train_x in self.train_dataset:
                inf_loss, gen_loss = self.compute_apply_gradients(train_x)
                print('Step {} of {}'.format(counter,steps))
                print('Inference Loss: {} ------ Generator Loss: {}'.format(inf_loss,gen_loss))
                print('-------------------------------------------------')
                counter += 1
            end_time = time.time()
            print('Epoch: {}, '
                    'time elapse for current epoch {}'.format(epoch,
                                                              end_time - start_time))

    def generate_images(self,number_of_samples=5,directory="imgs"):
        random_vector_for_generation = tf.random.normal(shape=[number_of_samples, 1024])
        predictions = self.sample(random_vector_for_generation)
        data_access.prepare_directory(directory)
        for i in range(predictions.shape[0]):
            data_access.store_image(directory,None,predictions[i],i,'vae')

