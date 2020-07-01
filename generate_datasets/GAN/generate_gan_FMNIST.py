#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:46:48 2020

"""

"""
    GAN
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""

import sys
sys.path.append("..")

from classifier import Classifier
import data_access

import tensorflow as tf
from tensorflow.keras.layers import Dense,BatchNormalization,Reshape,Conv2D,Dropout,Flatten,UpSampling2D,LeakyReLU
from tensorflow.keras.optimizers import Adam

import numpy as np
#import matplotlib.pyplot as plt
import time
from IPython import display

class Generator(tf.keras.Model):
    
    def __init__(self, random_noise_size = 100):
        super().__init__(name='generator')
        #layers
        
        self.dense_1 = Dense(7*7*256, use_bias = False, input_shape = (random_noise_size,))
        self.batchNorm1 = BatchNormalization()
        self.leaky_1 = LeakyReLU()
        self.reshape_1 = Reshape((7,7,256))
        
        #self.conv2 = Conv2DTranspose(128, (5, 5), strides = (1,1), padding = "same", use_bias = False)
        self.up_2 = UpSampling2D((1,1), interpolation='nearest')
        self.conv2 = Conv2D(128, (3, 3), strides = (1,1), padding = "same", use_bias = False)
        self.batchNorm2 = BatchNormalization()
        self.leaky_2 = LeakyReLU()
        
        #self.conv3 = Conv2DTranspose(64, (5, 5), strides = (2,2), padding = "same", use_bias = False)
        self.up_3 = UpSampling2D((2,2), interpolation='nearest')
        self.conv3 = Conv2D(64, (3, 3), strides = (1,1), padding = "same", use_bias = False)
        self.batchNorm3 = BatchNormalization()
        self.leaky_3 = LeakyReLU()
        
        #self.conv4 = Conv2DTranspose(1, (5, 5), strides = (2,2), padding = "same", use_bias = False, activation = "tanh")
        self.up_4 = UpSampling2D((2,2), interpolation='nearest')
        self.conv4 = Conv2D(1, (3, 3), strides = (1,1), padding = "same", use_bias = False)

        self.optimizer = Adam(1e-4)
        
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.reshape_1(self.leaky_1(self.batchNorm1(self.dense_1(input_tensor))))
        x = self.leaky_2(self.batchNorm2(self.conv2(self.up_2(x))))
        x = self.leaky_3(self.batchNorm3(self.conv3(self.up_3(x))))
        return  self.conv4(self.up_4(x))
    
    def generate_noise(self,batch_size, random_noise_size):
        return tf.random.normal([batch_size, random_noise_size])

    def objective(self,dx_of_gx):
        # Labels are true here because generator thinks he produces real images. 
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        return cross_entropy(tf.ones_like(dx_of_gx), dx_of_gx) 
    
    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__(name = "discriminator")
        
        #Layers
        self.conv_1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1])
        self.leaky_1 = LeakyReLU(alpha = 0.01)
        self.dropout_1 = Dropout(0.3)
        
        self.conv_2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.leaky_2 = LeakyReLU(alpha = 0.01)
        self.dropout_2 = Dropout(0.3)

        self.flat = Flatten()
        self.logits = Dense(1)  # This neuron tells us if the input is fake or real
        
        self.optimizer = Adam(1e-4)
        
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.dropout_1(self.leaky_1(self.conv_1(input_tensor)))
        x = self.dropout_2(self.leaky_2(self.conv_2(x)))
        x = self.flat(x)
        #x = tf.expand_dims(x, axis=-1)
        return self.logits(x)
    
    def objective(self,d_x, g_z, smoothing_factor = 0.9):
        """
        d_x = real output
        g_z = fake output
        """
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        real_loss = cross_entropy(tf.ones_like(d_x) * smoothing_factor, d_x) 
        fake_loss = cross_entropy(tf.zeros_like(g_z), g_z)
        total_loss = real_loss + fake_loss

        return total_loss
    
    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        
        
class GAN(tf.keras.Model):
    def __init__(self):
        super().__init__(name = "GAN")
        self.generator = Generator(100)
        self.discriminator = Discriminator()
        self.classifier = Classifier()
               
        self.train_dataset = None
        self.test_dataset = None
        self.train_labels = None
        self.test_labels = None
        
    def load_dataset(self,dataset):
        self.train_dataset,self.train_labels,self.test_dataset,self.test_labels = dataset

    @tf.function()
    def training_step(self,images:np.ndarray , batch_size):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            noise = self.generator.generate_noise(batch_size, 100)
            g_z = self.generator(noise, training=True)
            d_x_true = self.discriminator(images, training=True) # Trainable?
            d_x_fake = self.discriminator(g_z, training=True) # dx_of_gx

            discriminator_loss = self.discriminator.objective(d_x_true, d_x_fake)
            generator_loss = self.generator.objective(d_x_fake)
            
        # Adjusting Gradient of Discriminator
        gradients_of_discriminator = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.discriminator.backPropagate(gradients_of_discriminator, self.discriminator.trainable_variables)

        # Adjusting Gradient of Generator
        gradients_of_generator = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator.backPropagate(gradients_of_generator, self.generator.trainable_variables)
        return g_z
    
    def train_model(self,epoches,batch_size):
        s_time = time.time()
        for epoch in range(epoches):
            start_time = time.time()
            for batch in self.train_dataset: 
                images_gen = self.training_step(batch ,batch_size)
                self.predict_generated_image(images_gen)
            end_time = time.time()
            ## After ith epoch plot image 
            if (epoch % 1) == 0: 
                display.clear_output(wait=False)
                print('Epoch: {}, time elapse for current epoch {}'.format(epoch,
                                                          end_time - start_time))
        print('Time elapse {}'.format(time.time() - s_time))


    def generate_images(self,number_of_samples,directory):
        seed = tf.random.normal([number_of_samples, 100])
        predictions = self.generator(seed)
        data_access.prepare_directory(directory)
        for i in range(predictions.shape[0]):
            data_access.store_image(directory,i,predictions[i, :, :, 0])
            #image = tf.reshape(predictions[i, :, :, 0],shape=(1,28,28,1))
            #print(self.classifier.predict(image))

    def predict_generated_image(self,images_gen):
         #image = tf.reshape(images_gen[0, :, :, 0],shape=(1,28,28,1))
         #print(self.classifier.predict(image))
         pass