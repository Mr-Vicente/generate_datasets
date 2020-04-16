# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:13:25 2020

@author: MrVicente
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense,Reshape,Conv2D,Flatten,UpSampling2D,LeakyReLU,Cropping2D,AveragePooling2D,BatchNormalization,ZeroPadding2D,ReLU
from tensorflow.keras.optimizers import Adam

import numpy as np
import time

class ResidualBlock(tf.keras.Model):
    def __init__(self, input_channels, out_channels, stride=(1,1), downsample=None):
        super().__init__(name = "Residual_Block")
        if(input_channels != out_channels):
            self.conv_expand = Conv2D(out_channels,(1,1), strides=stride)
        else:
            self.conv_expand = None
        
        self.conv_1 = Conv2D(out_channels,(4,4), strides=stride)
        self.batch_norm_1 = BatchNormalization()
        self.leaky_1 = LeakyReLU(alpha=0.2)
        self.conv_2 = Conv2D(out_channels,(4,4), strides=stride)
        self.batch_norm_2 = BatchNormalization()
        self.leaky_2 = LeakyReLU(alpha=0.2)
        
    def call(self, input_tensor):
        if(self.conv_expand is not None):
            residual = self.conv_expand(input_tensor)
        else:
            residual = input_tensor
        out = self.conv_1(input_tensor)
        out = self.batch_norm_1(out)
        out = self.leaky_1(out)
        out = self.conv_2(out)
        out = self.batch_norm_2(out)
        out += residual
        out = self.leaky_2(out)
        return out
    
class Generator(tf.keras.Model):
    def __init__(self, latent_dim = 128, channels=[64,128,256,512,512,512]):
        super().__init__(name = "Generator")
        cc = channels[0]
        self.dense_1 = Dense(latent_dim * 5 * 5, name = 'Generator_Dense_1')
        self.relu = ReLU()
        self.reshape_1 = Reshape((5,5,latent_dim))
        self.res_block_1 = ResidualBlock(cc,cc)
        self.reses = list()
        for ch in channels[1:]:
            self.reses.append([UpSampling2D(),ResidualBlock(cc,ch)])
            cc = ch
        self.toRGB = Conv2D(3, (5,5), padding='same', kernel_initializer='he_normal', name = 'Generator_To_RGB')
        
    def call(self, input_tensor):
        x = self.dense_1(input_tensor)
        x = self.relu(x)
        x = self.reshape_1(x)
        x = self.res_block_1(x)
        for i in range(len(self.reses)):
            x = self.reses[i][0](x)
            x = self.reses[i][1](x)
        x = self.toRGB(x)
        x = tf.keras.layers.Activation(activation='sigmoid')(x)
        return x
    
class Encoder(tf.keras.Model):
    def __init__(self, channels=[64,128,256,512,512,512]):
        super().__init__(name = "Encoder")
        self.conv_1 = Conv2D(128, (4,4),strides=(1,1), input_shape = (152,152,3))
        self.batch_norm_1 = BatchNormalization()
        self.leaky_1 = LeakyReLU(alpha=0.2)
        self.downsampling = AveragePooling2D()
        self.reses = list()
        cc = channels[0]
        for ch in channels[1:]:
            self.reses.append([ResidualBlock(cc,ch),AveragePooling2D()])
            cc = ch
        self.res_block_n = ResidualBlock(cc,ch)
        self.flat = Flatten()
        
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
        
class IntroVae(tf.keras.Model):
    def __init__(self, latent_dim = 128, channels=[64,128,256,512,512,512]):
        super().__init__(name = "Intro_Vae")
        
        self.latent_dim = latent_dim
        self.inference_net = Encoder(channels)
        self.generative_net = Generator(self.latent_dim,channels)
        
        self.optimizer = Adam(learning_rate=0.0001,beta_1=0,beta_2=0.9)
        
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
          eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
        
    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
          probs = tf.sigmoid(logits)
          return probs

        return logits

    def log_normal_pdf(self,sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)
        
    @tf.function
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def compute_apply_gradients(self,x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
    def train_model(self,epochs):
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            for train_x in self.train_dataset:
                self.compute_apply_gradients(train_x)
            end_time = time.time()
