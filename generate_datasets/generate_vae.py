#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:46:32 2020
"""
"""
    VAE
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""
import data_access

import tensorflow as tf
from tensorflow.keras.layers import InputLayer,Dense,Conv2DTranspose,Reshape,Conv2D,Flatten
from tensorflow.keras.optimizers import Adam

import numpy as np
#import matplotlib.pyplot as plt
import time
from IPython import display

class Inference_net(tf.keras.Model):
    def __init__(self,latent_dim):
        super().__init__(name='inference')
         #layers
        self.input_layer = InputLayer(input_shape=(28, 28, 1))
        self.conv_1 = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')
        self.conv_2 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')
        self.flat = Flatten()
        self.last_inf = Dense(latent_dim + latent_dim)
        
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.conv_1(self.input_layer(input_tensor))
        x = self.conv_2(x)
        return  self.last_inf(self.flat(x))

class Generator_net(tf.keras.Model):
    def __init__(self,latent_dim):
        super().__init__(name='inference')
         #layers
        self.input_layer = InputLayer(input_shape=(latent_dim,))
        self.dense_1 = Dense(units=7*7*32, activation="relu")
        self.reshape = Reshape(target_shape=(7, 7, 32))
        self.conv_trans_1 = Conv2DTranspose(
              filters=64,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu')
        self.conv_trans_2 = Conv2DTranspose(
              filters=32,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu')
        self.conv_trans_3 = Conv2DTranspose(
              filters=1,
              kernel_size=3,
               strides=(1, 1),
                padding="SAME")
        
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.dense_1(self.input_layer(input_tensor))
        x = self.reshape(x)
        x = self.conv_trans_1(x)
        x = self.conv_trans_2(x)
        return  self.conv_trans_3(x)

class VAE(tf.keras.Model):
    
    def __init__(self,latent_dim):
        super().__init__(name='vae')
        self.latent_dim = latent_dim
        self.inference_net = Inference_net(self.latent_dim)
        self.generative_net = Generator_net(self.latent_dim)
        self.optimizer = Adam(1e-4)
        
        self.train_dataset = None
        self.test_dataset = None
        self.train_labels = None
        self.test_labels = None
        
    def load_dataset(self,dataset):
        self.train_dataset,self.train_labels,self.test_dataset,self.test_labels = dataset

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

            if epoch % 1 == 0:
                loss = tf.keras.metrics.Mean()
                for test_x in self.test_dataset:
                    loss(self.compute_loss(test_x))
                elbo = -loss.result()
                display.clear_output(wait=False)
                print('Epoch: {}, Test set ELBO: {}, '
                    'time elapse for current epoch {}'.format(epoch,
                                                              elbo,
                                                              end_time - start_time))
    def generate_images(self,number_of_samples=5,directory="imgs"):
        random_vector_for_generation = tf.random.normal(shape=[number_of_samples, self.latent_dim])
        predictions = self.sample(random_vector_for_generation)
        data_access.prepare_directory(directory)
        for i in range(predictions.shape[0]):
            data_access.store_image(directory,i,predictions[i, :, :, 0])
