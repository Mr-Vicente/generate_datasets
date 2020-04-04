# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:00:34 2020

@author: MrVicente
"""

import data_access
from spectral_normalization import SpectralNormalization

import tensorflow as tf
from tensorflow.keras.layers import Dense,BatchNormalization,Reshape,Conv2D,Dropout,Flatten,UpSampling2D,LeakyReLU,Cropping2D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend

import numpy as np
#import matplotlib.pyplot as plt
import time
#from IPython import display
from datetime import datetime
import os

class WGAN(tf.keras.Model):
    def __init__(generator,critic, classifier_filename='TypeC_0.99_ConvExp19.hdf5'):
        super().__init__(name = "WGAN")
        self.random_noise_size = random_noise_size
        self.generator = generator
        self.classifier = tf.keras.models.load_model(classifier_filename)
        self.critic =critic
        
        self.prepare_seed()
        self.load_weights()
               
        self.train_dataset = None
        self.batch_size = batch_size
        
    def load_dataset(self,dataset,n_classes):
        self.train_dataset,self.train_labels,self.test_dataset,self.test_labels = dataset
        self.num_classes = n_classes
        
    def load_weights(self):
        if ('weights' in os.listdir('.')):
            self.critic.load_weights('weights/c_weights/c_weights')
            self.generator.load_weights('weights/g_weights/g_weights')
    def prepare_seed():
        if('seed.npz' not in os.listdir('.')):
            self.generator.set_seed()
        else :
            self.generator.load_seed()

    @tf.function
    def predict_batch(self,images,type_class):
        """
        Classify each image received and prepare for loss function

        :param images: - images tensors
        :param type_class: - class chosen to influence generator. Its must be a number
        """
        images_predictions = tf.TensorArray(tf.float32,size=0,dynamic_size=True)
        ys = tf.TensorArray(tf.float32,size=0,dynamic_size=True)
        matched_images = tf.TensorArray(tf.float32,size=0,dynamic_size=True)
        index = 0
        basis = tf.convert_to_tensor([0,1],dtype=tf.float32)
        for i in tf.range(len(images)):
            gen_image = data_access.normalize(data_access.de_standardize(images[i]))
            img = tf.expand_dims(gen_image,axis=0)
            c = self.classifier(img)
            if(self.num_classes == 2):
                x = tf.subtract(c,basis)
                w_list = tf.abs(x)
            else:
                w_list = c
            w_list = tf.reshape(w_list,(w_list.shape[1],))
            
            images_predictions = images_predictions.write(i,w_list)
            y_list = tf.one_hot(type_class,self.num_classes)
            ys = ys.write(i,y_list)
            if(tf.reduce_all(tf.equal(w_list,y_list))):
                matched_images = matched_images.write(index,images[i])
                index +=1
                
        return images_predictions.stack(), ys.stack(),matched_images.stack()

    @tf.function
    def gradient_penalty(self,generated_samples,real_images,half_batch):
        alpha = backend.random_uniform(shape=[half_batch,1,1,1],minval=0.0,maxval=1.0)
        differences = generated_samples - real_images
        interpolates = real_images + (alpha * differences)
        gradients = tf.gradients(self.critic(interpolates),[interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),axis=[1,2,3]))
        gradient_p = tf.reduce_mean((slopes-1.)**2)
        return gradient_p

    @tf.function
    def training_step_critic(self,real_imgs,gen_imgs,real_labels,gen_labels,half_batch):
        lambda_ = 10.0
        with tf.GradientTape() as tape:
            d_x_real = self.critic(real_imgs, training=True) 
            d_x_gen = self.critic(gen_imgs, training=True) 
            critic_r_loss = self.critic.compute_loss(real_labels, d_x_real)
            critic_g_loss = self.critic.compute_loss(gen_labels, d_x_gen)
            total_loss = critic_r_loss + critic_g_loss + (lambda_ * self.gradient_penalty(gen_imgs,real_imgs,half_batch))
        
        gradients_of_critic = tape.gradient(total_loss, self.critic.trainable_variables)
        self.critic.backPropagate(gradients_of_critic, self.critic.trainable_variables)
        return total_loss


    @tf.function
    def training_step_generator(self,noise_size,class_type):
        # prepare points in latent space as input for the generator
        X_g = self.generator.generate_noise(self.batch_size,noise_size)
        # create inverted labels for the fake samples
        y_g = -np.ones((self.batch_size, 1)).astype(np.float32)
        with tf.GradientTape() as tape:
            d_x = self.generator(X_g, training=True) # Trainable?
            d_z = self.critic(d_x, training=True)
            images_predictions, ys, matched_images = self.predict_batch(d_x,class_type)
            generator_loss = self.generator.compute_loss(d_z, y_g, ys, images_predictions)
        
        gradients_of_generator = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator.backPropagate(gradients_of_generator, self.generator.trainable_variables)
        return generator_loss,matched_images, self.generator(self.generator.seed, training=False)

    def generate_real_samples(self, n_samples):
        # choose random instances
        ix = np.random.randint(0, self.train_dataset.shape[0], n_samples)
        # select images
        X = self.train_dataset[ix] #tf.gather(self.train_dataset,ix) 
        # associate with class labels of -1 for 'real'
        y = -np.ones((n_samples, 1)).astype(np.float32)
        #convert X to tensor
        return tf.convert_to_tensor(X), y

    @tf.function
    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, noise_size, n_samples):
        # generate points in latent space
        x_input = self.generator.generate_noise(n_samples,noise_size)
        # get images generated
        X = self.generator(x_input,training=True)
        # associate with class labels of 1.0 for 'fake'
        y = np.ones((n_samples, 1)).astype(np.float32)
        return X, y

    def define_loss_tensorboard(self):
        """
        Tensorboard Integration - loss scallars
        """
        logdir="logs/train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        return tf.summary.create_file_writer(logdir=logdir)

    def define_graph_tensorboard(self):
        """
        Tensorboard Integration - grath
        """
        logdir="logs/graph/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        return tf.summary.create_file_writer(logdir=logdir)
    
    def train_model(self,epoches,n_critic=5,class_type=0,directory = 'imgs',n_img_per_epoch = 4):
        """
        Train model for an amount of epochs

        :param epoches: - cycles of training over all dataset
        :param n_critic: - number of times critic trains more than generator
        :param class_type: - class number: converge generated data to this class
        :param directory: - directory where images will be placed during training
        """
        batch_per_epoch = int(self.train_dataset.shape[0] / self.batch_size)

        # calculate the number of training iterations
        n_steps = batch_per_epoch * epoches
        # calculate the size of half a batch of samples
        half_batch = int(self.batch_size / 2)

        sum_writer_loss = self.define_loss_tensorboard()
        #self.classifier_m.load_local_model()
        avg_loss_critic = tf.keras.metrics.Mean()
        avg_loss_gen = tf.keras.metrics.Mean()
        try:
            epoch = int(open('current_epoch.txt').read())
        except:
            epoch = 0
        start_time = time.time()
        for i in range(n_steps):
            for _ in range(n_critic):
                # get randomly selected 'real' samples
                X_real, y_real = self.generate_real_samples(half_batch)
                # generate 'fake' examples
                X_fake, y_fake = self.generate_fake_samples(self.random_noise_size, half_batch)
                
                # update critic model weights
                c_loss = self.training_step_critic(X_real,X_fake, y_real,y_fake,half_batch)
                avg_loss_critic(c_loss)
                
            gen_loss, matched_images, gen_images = self.training_step_generator(self.random_noise_size,class_type)
            avg_loss_gen(gen_loss)
            data_access.print_training_output(i,n_steps, avg_loss_critic.result(),avg_loss_gen.result()) 
            if((i % (n_steps / epoches)) == 0):
                data_access.store_images_seed(directory,gen_images[:n_img_per_epoch],epoch)
                with sum_writer_loss.as_default():
                    tf.summary.scalar('loss_gen', avg_loss_gen.result(),step=self.generator.optimizer.iterations)
                    tf.summary.scalar('avg_loss_critic', avg_loss_critic.result(),step=self.critic.optimizer.iterations)
                epoch += 1
                if((epoch % 1) == 0):
                    self.generator.save_weights('weights/g_weights/g_weights',save_format='tf')
                    self.critic.save_weights('weights/c_weights/c_weights',save_format='tf')
                    data_access.write_current_epoch(filename='current_epoch',epoch)
        print('Time elapse {}'.format(time.time() - start_time))

    def generate_images(self,number_of_samples,directory):
        seed = tf.random.normal([number_of_samples, self.random_noise_size])
        images = self.generator(seed)
        predictions = self.classifier(data_access.normalize(data_access.de_standardize(images)))
        data_access.produce_generate_figure('imgs',images,predictions)
