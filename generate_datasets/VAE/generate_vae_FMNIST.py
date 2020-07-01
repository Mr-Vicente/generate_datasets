#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    VAE
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""
import sys
sys.path.append("..")

import data_access

import tensorflow as tf
from tensorflow.keras.layers import InputLayer,Dense,Reshape,Conv2D,Flatten,UpSampling2D,ReLU
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

class Inference_net(tf.keras.Model):
    def __init__(self,latent_dim=128):
        super().__init__(name='inference')
         #layers
        self.conv_1 = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu',input_shape=(28,28,1))
        self.conv_2 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')
        self.flat = Flatten()
        self.last_inf = Dense(2 * latent_dim)
        
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.conv_1(input_tensor)
        x = self.conv_2(x)
        x = self.last_inf(self.flat(x))
        return x


class Generator_net(tf.keras.Model):
    def __init__(self,batch_size = 64, latent_dim=128):
        super().__init__(name='Generator')
         #layers
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.input_layer = InputLayer(input_shape=(latent_dim,))
        self.dense_1 = Dense(units=7*7*batch_size, activation="relu")
        self.reshape = Reshape(target_shape=(7, 7, batch_size))

        self.up_1 = UpSampling2D((2,2), interpolation='nearest')
        self.conv_1 = Conv2D(filters=64, kernel_size=(3, 3), strides = (1,1), padding = "same")
        self.leaky_1 = ReLU()

        self.up_2 = UpSampling2D((2,2), interpolation='nearest')
        self.conv_2 = Conv2D(filters=32, kernel_size=(3, 3), strides = (1,1), padding = "same")
        self.leaky_2 = ReLU()

        self.conv_3 = Conv2D(filters=1, kernel_size=(3, 3), strides = (1,1), padding = "same")

    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.dense_1(self.input_layer(input_tensor))
        x = self.reshape(x)
        x = self.leaky_1(self.conv_1(self.up_1(x)))
        x = self.leaky_2(self.conv_2(self.up_2(x)))
        x = self.conv_3(x)
        return x

    def set_seed(self):
        self.seed = tf.random.normal([self.batch_size, self.latent_dim])
        data_access.store_seed_in_file('seed',self.seed)

    def load_seed(self):
        self.seed = data_access.load_seed_from_file('seed')

class VAE(tf.keras.Model):
    
    def __init__(self,batch_size = 64, latent_dim = 128):
        super().__init__(name='vae')
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.inference_net = Inference_net(self.latent_dim)
        self.generative_net = Generator_net(batch_size, self.latent_dim)
        self.optimizer = Adam(1e-4)

        if('seed.npz' not in os.listdir('.')):
            self.generative_net.set_seed()
        else :
            self.generative_net.load_seed()
        
        self.train_dataset = None
        self.test_dataset = None
        self.train_labels = None
        self.test_labels = None
        
    def load_dataset(self,dataset):
        self.train_dataset,self.train_labels,self.test_dataset,self.test_labels = dataset

    @tf.function
    def sample(self, eps=None,training = True):
        if eps is None:
          eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True,training=training)
        
    @tf.function
    def encode(self, x, training=True):
        mean, logvar = tf.split(self.inference_net(x,training=training), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def decode(self, z, apply_sigmoid=False,training = True):
        logits = self.generative_net(z,training=training)
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
        return loss

    def define_loss_tensorboard(self):
        logdir="logs/train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        return tf.summary.create_file_writer(logdir=logdir)

    def generate_real_samples(self, n_samples):
        # choose random instances
        ix = np.random.randint(0, self.train_dataset.shape[0], n_samples)
        # select images
        X = self.train_dataset[ix]
        #convert X to tensor
        return tf.convert_to_tensor(X.astype(np.float32))


    def train_model(self,epochs,batch_size=None,directory='imgs',images_per_epoch=4):
        if batch_size is None: batch_size = self.batch_size
        sum_writer_loss = self.define_loss_tensorboard()
        avg_vae_loss = tf.keras.metrics.Mean()
        start_time = time.time()

        batch_per_epoch = int(self.train_dataset.shape[0] / batch_size)
        # calculate the number of training iterations
        n_steps = batch_per_epoch * epochs

        try:
            epoch = int(open('current_epoch.txt').read())
        except:
            epoch = 0

        for step_i in range(n_steps):
            train_x = self.generate_real_samples(batch_size)
            loss = self.compute_apply_gradients(train_x)
            avg_vae_loss(loss)
            if((step_i % (n_steps / epochs)) == 0):
                epoch += 1
                gen_images = self.sample(self.generative_net.seed,False)
                data_access.store_images_seed(directory,gen_images[:images_per_epoch],epoch)
                data_access.write_current_epoch('current_epoch',epoch=epoch)
            data_access.print_training_output_simple_loss(step_i,n_steps,loss)
            with sum_writer_loss.as_default():
                tf.summary.scalar('loss_vae', avg_vae_loss.result(),step=self.optimizer.iterations)
        end_time = time.time()
        data_access.print_training_time(start_time,end_time,None)

    def generate_images(self,number_of_samples=5,directory="imgs"):
        random_vector_for_generation = tf.random.normal(shape=[number_of_samples, self.latent_dim])
        images = self.sample(random_vector_for_generation,False)
        data_access.store_images_seed(directory,images,'None','vae')

    def get_latent_code(self, image):
        mean, logvar = self.encode(image,training=False)
        l_code = self.reparameterize(mean, logvar)
        return l_code.numpy()
    
    def interpolate_c1_to_c2(self, images=None, labels=None , c1=0, c2=1 , alphas=np.linspace(-1, 1, 20)):
        if images == None: images = self.test_dataset
        if labels == None: labels = self.test_labels
        n_samples = alphas.shape[0]
        figsize=(14 * int(n_samples / 10),20)
        #Find centroids of class c1 and c2
        z_c1_avg = self.get_latent_code(images[labels == c1]).mean(axis=0)
        z_c2_avg = self.get_latent_code(images[labels == c2]).mean(axis=0)

        #Find medoid of class c1
        z_c1_med = np.median(self.get_latent_code(images[labels == c1]), axis=0)

        #Interpolation vector c1->c2
        z_c1_c2 = z_c2_avg - z_c1_avg

        x_gens = []
        for alpha in alphas:
            z_interp = z_c1_med + alpha * z_c1_c2
            z_interp = z_interp.reshape(1,-1)
            image = self.sample(z_interp, training=False)
            x_gens.append(image)
        self.display_images(x_gens,figsize)
        return x_gens

    def interpolate_many_c1_to_c2(self, images=None, labels=None , c1=0, c2=1 , alphas=np.linspace(-1, 1, 20), class_names=None):
        if images == None: images = self.test_dataset
        if labels == None: labels = self.test_labels
        n_samples = alphas.shape[0]
        figsize=(14 * int(n_samples / 10),20)
        #Find latent code of classes
        z_c1 = self.get_latent_code(images[labels == c1])
        z_c2 = self.get_latent_code(images[labels == c2])

        N_SPLITS = 10
        c1_splits = np.array(np.split(z_c1, N_SPLITS)).astype(np.float32)
        c2_splits = np.array(np.split(z_c2, N_SPLITS)).astype(np.float32)

        classifier = tf.keras.models.load_model('classifier.h5')
        for i_c in range(N_SPLITS):
            z_c1_i = c1_splits[i_c]
            z_c2_i = c2_splits[i_c]
            z_c1_i_avg = z_c1_i.mean(axis=0)
            z_c2_i_avg = z_c2_i.mean(axis=0)

            c_diff = z_c2_i_avg - z_c1_i_avg

            #Find medoid of class c1
            z_c1_i_med = np.median(z_c1_i, axis=0)

            x_gens = []
            for alpha in alphas:
                z_interp = z_c1_i_med + alpha * c_diff
                z_interp = z_interp.reshape(1,-1)
                image = self.sample(z_interp, training=False)
                x_gens.append(image)
            self.display_images(x_gens,figsize)
            self.plot_alpha_probs(x_gens, classifier, classes = [c1,c2], class_names=class_names)
        return x_gens
          
    def interpolate_c1_to_c2_though_c3(self, images=None, labels=None , c1=0, c2=1, c3=2, alphas=np.linspace(-1, 1, 20)):
        if images == None: images = self.test_dataset
        if labels == None: labels = self.test_labels
        n_samples = alphas.shape[0]
        figsize=(14 * int(n_samples / 10),20)
        #Find centroids of class c1 and c2
        z_c1_avg = self.get_latent_code(images[labels == c1]).mean(axis=0)
        z_c2_avg = self.get_latent_code(images[labels == c2]).mean(axis=0)

        #Find medoid of class c1
        z_c3_med = np.median(self.get_latent_code(images[labels == c3]), axis=0)

        #Interpolation vector c1->c2
        z_c1_c2 = z_c2_avg - z_c1_avg

        x_gens = []
        for alpha in alphas:
            z_interp = z_c3_med + alpha * z_c1_c2
            z_interp = z_interp.reshape(1,-1)
            image = self.sample(z_interp, training=False)
            x_gens.append(image)
        self.display_images(x_gens,figsize)
        return x_gens  
    
    def display_images(self, images, figsize=None):
        n_images = len(images)
        plt.subplots(figsize=figsize, squeeze=False)
        for i, image in enumerate(images):
            plt.subplot(1, n_images, i+1)
            if (image.shape[-1] == 1):
                plt.imshow(image[0,:,:,0], cmap='gray')
            else:
                plt.imshow(image[0,:,:,:])
            plt.axis('off')

    def plot_alpha_probs(self,images, classifier, alphas=np.linspace(-1, 1, 20), classes=[0,1],class_names=None):

        fig, ax = plt.subplots()
        for c in classes:
            cs = []
            for image in images:
                prediction = classifier(image).numpy()[0]
                t_class = prediction[c]
                cs.append(t_class)
            ax.plot(alphas,cs, label=class_names[c])

        ax.legend(loc='best')
        ax.set_ylabel('Probability')
        ax.set_xlabel('delta distance')