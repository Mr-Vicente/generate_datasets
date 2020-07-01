# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 19:46:21 2020

@author: MrVicente
"""

from classifier import Classifier
import data_access

import tensorflow as tf
from tensorflow.keras.layers import Dense,BatchNormalization,Reshape,Conv2D,Dropout,Flatten,UpSampling2D,LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend

import numpy as np
#import matplotlib.pyplot as plt
import time
from IPython import display

class Generator(tf.keras.Model):
    
    def __init__(self, random_noise_size = 100):
        super().__init__(name='generator')
        #layers
        init = RandomNormal(stddev=0.2)
        self.dense_1 = Dense(7*7*256, use_bias = False, input_shape = (random_noise_size,))
        self.batchNorm1 = BatchNormalization()
        self.leaky_1 = LeakyReLU(alpha=0.2)
        self.reshape_1 = Reshape((7,7,256))
        
        self.up_2 = UpSampling2D((1,1), interpolation='nearest')
        self.conv2 = Conv2D(128, (3, 3), strides = (1,1), padding = "same", use_bias = False, kernel_initializer=init)
        self.batchNorm2 = BatchNormalization()
        self.leaky_2 = LeakyReLU(alpha=0.2)
        
        self.up_3 = UpSampling2D((2,2), interpolation='nearest')
        self.conv3 = Conv2D(64, (3, 3), strides = (1,1), padding = "same", use_bias = False, kernel_initializer=init)
        self.batchNorm3 = BatchNormalization()
        self.leaky_3 = LeakyReLU(alpha=0.2)
        
        self.up_4 = UpSampling2D((2,2), interpolation='nearest')
        self.conv4 = Conv2D(1, (3, 3), activation='tanh', strides = (1,1), padding = "same", use_bias = False, kernel_initializer=init)

        self.optimizer = RMSprop(lr=0.00005)
        
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.reshape_1(self.leaky_1(self.batchNorm1(self.dense_1(input_tensor))))
        x = self.leaky_2(self.batchNorm2(self.conv2(self.up_2(x))))
        x = self.leaky_3(self.batchNorm3(self.conv3(self.up_3(x))))
        return  self.conv4(self.up_4(x))
    
    def generate_noise(self,batch_size, random_noise_size):
        return tf.random.normal([batch_size, random_noise_size])

    def compute_loss(self,y_true,y_pred,class_wanted,class_y):
        """ Wasserstein loss - prob of classfier get it right
        """
        #return tf.math.subtract(backend.mean(y_true * y_pred),categorical_crossentropy(class_wanted,class_y))
        return backend.mean(y_true * y_pred) - categorical_crossentropy(class_wanted,class_y)

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        
class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__(name = "critic")
        
        init = RandomNormal(stddev=0.2)
        #Layers
        self.conv_1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init, input_shape=[28, 28, 1])
        self.leaky_1 = LeakyReLU(alpha=0.2)
        self.dropout_1 = Dropout(0.3)
        
        self.conv_2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)
        self.leaky_2 = LeakyReLU(alpha=0.2)
        self.dropout_2 = Dropout(0.3)

        self.flat = Flatten()
        self.logits = Dense(1)  # This neuron tells us if the input is fake or real
        
        self.optimizer = RMSprop(lr=0.00005)
        
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.dropout_1(self.leaky_1(self.conv_1(input_tensor)))
        x = self.dropout_2(self.leaky_2(self.conv_2(x)))
        x = self.flat(x)
        return self.logits(x)

    def compute_loss(self,y_true,y_pred,grad_p):
        """ Wasserstein loss
        """
        lambda_ = 10.0
        return backend.mean(y_true * y_pred) + (lambda_ * grad_p)

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        
        
class WGAN(tf.keras.Model):
    def __init__(self):
        super().__init__(name = "WGAN")
        self.generator = Generator(100)
        self.critic = Critic()
        self.classifier_m = Classifier()
               
        self.train_dataset = None
        self.test_dataset = None
        self.train_labels = None
        self.test_labels = None
        
    def load_dataset(self,dataset):
        self.train_dataset,self.train_labels,self.test_dataset,self.test_labels = dataset


    def predict_batch(self,images,type_class):
        images_predictions = list()
        ys = list()
        for gen_image in images:
            img = tf.expand_dims(gen_image,axis=0)
            c_type = self.classifier_m.predict_image(img)
            w_list = [0] * 10
            w_list[c_type] = 1
            images_predictions.append(w_list)
            y_list = [0] * 10
            y_list[type_class] = 1
            ys.append(y_list)

        return np.float32(images_predictions), np.float32(ys)

    def gradient_penalty(self,generated_samples,ys,batch_size):
        alpha = backend.random_uniform(shape=[batch_size,1,1,1],minval=0.0,maxval=1.0)
        differences = generated_samples - ys
        interpolates = ys + (alpha * differences)
        gradients = tf.gradients(self.critic(interpolates),[interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        gradient_p = tf.reduce_mean((slopes-1.)**2)
        return gradient_p

    @tf.function()
    def training_step_critic(self,images:np.ndarray,ys,real,batch_size):
        with tf.GradientTape() as tape:
            d_x = self.critic(images, training=True) # Trainable?
            critic_loss = self.critic.compute_loss(ys, d_x,self.gradient_penalty(images,real,batch_size))
        
        gradients_of_critic = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.backPropagate(gradients_of_critic, self.critic.trainable_variables)
    
    #@tf.function()
    def training_step_generator(self,noise_size, batch_size,class_type):
        # prepare points in latent space as input for the generator
        X_g = self.generator.generate_noise(batch_size,noise_size)
        # create inverted labels for the fake samples
        y_g = -np.ones((batch_size, 1)).astype(np.float32)
        with tf.GradientTape() as tape:
            d_x = self.generator(X_g, training=True) # Trainable?
            d_z = self.critic(d_x, training=True)
            with tape.stop_recording():
                images_predictions, ys = self.predict_batch(d_x,class_type)
            generator_loss = self.generator.compute_loss(d_z, y_g, images_predictions, ys)
        
        gradients_of_generator = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator.backPropagate(gradients_of_generator, self.generator.trainable_variables)

    def generate_real_samples(self, n_samples):
        # choose random instances
        ix = np.random.randint(0, self.train_dataset.shape[0], n_samples)
        # select images
        X = self.train_dataset[ix]
        # associate with class labels of -1 for 'real'
        y = -np.ones((n_samples, 1)).astype(np.float32)
        return X, y

    
    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, noise_size, n_samples):
        # generate points in latent space
        x_input = self.generator.generate_noise(n_samples,noise_size)
        # get images generated
        X = self.generator(x_input,training=True)
        # associate with class labels of 1.0 for 'fake'
        y = np.ones((n_samples, 1)).astype(np.float32)
        return X, y
    
    def train_model(self,epoches,batch_size,n_critic=5,noise_size=100,class_type=0):

        batch_per_epoch = int(self.train_dataset.shape[0] / batch_size)

        # calculate the number of training iterations
        n_steps = batch_per_epoch * epoches
        # calculate the size of half a batch of samples
        half_batch = int(batch_size / 2)

        self.classifier_m.load_local_model()
        for i in range(n_steps):
            for _ in range(n_critic):
                # get randomly selected 'real' samples
                X_real, y_real = self.generate_real_samples(half_batch)
                # generate 'fake' examples
                X_fake, y_fake = self.generate_fake_samples(noise_size, half_batch)
                
                # update critic model weights
                c_loss1 = self.training_step_critic(X_real, y_real,X_real,half_batch)
                # update critic model weights
                c_loss2 = self.training_step_critic(X_fake, y_fake,X_real,half_batch)

            print("{} OUT OF {}".format(i,n_steps))

            self.training_step_generator(noise_size,batch_size,class_type)


    def generate_images(self,number_of_samples,directory):
        seed = tf.random.normal([number_of_samples, 100])
        predictions = self.generator(seed)
        data_access.prepare_directory(directory)
        for i in range(predictions.shape[0]):
            data_access.store_image(directory,i,predictions[i, :, :, 0])


