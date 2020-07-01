# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:40:42 2020

@author: MrVicente
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 09:41:56 2020
"""

"""
    152x152 WGAN GP WITH CLASSIFIER
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""
import sys
sys.path.append("..")

import data_access

import tensorflow as tf
from tensorflow.keras.layers import Dense,BatchNormalization,Reshape,Conv2D,Flatten,UpSampling2D,LeakyReLU,LayerNormalization,ReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy,KLDivergence
from tensorflow.keras import backend

import numpy as np
#import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import math

class Generator(tf.keras.Model):
    
    def __init__(self, model_parameters=None):
        super().__init__(name='generator')
        #layers
        if model_parameters is None:
            model_parameters = {
                'lr': 0.0001,
                'beta1': 0,
                'batch_size': 64,
                'latent_dim': 128,
                'image_size': 152
            }
        self.model_parameters = model_parameters
        self.batch_size = model_parameters['batch_size']
        self.noise_size = model_parameters['latent_dim']
        dim = 8 * self.batch_size
        init = RandomNormal(stddev=0.02)
        self.dense_1 = Dense(dim*4*4, use_bias = False, input_shape = (self.noise_size,))
        self.batchNorm1 = BatchNormalization()
        self.leaky_1 = ReLU()
        self.reshape_1 = Reshape((4,4,dim))
        self.layers_blocks = list()
        
        number_of_layers_needed = int(math.log(model_parameters['image_size'],2))-3
        for i in range(number_of_layers_needed):
            dim /= 2
            self.layers_blocks.append([
                UpSampling2D((2,2), interpolation='nearest'),
                Conv2D(dim, (5, 5), strides = (1,1), padding = "same", use_bias = False, kernel_initializer=init),
                BatchNormalization(),
                ReLU(),
            ])
    
        self.up_toRGB = UpSampling2D((2,2), interpolation='nearest')
        self.conv_toRGB = Conv2D(3, (5, 5), activation='tanh', strides = (1,1), padding = "same", use_bias = False, kernel_initializer=init)
        
        self.optimizer = Adam(learning_rate=model_parameters['lr'],beta_1=model_parameters['beta1'],beta_2=0.9)

    def call(self, input_tensor, training = True):
        ## Definition of Forward Pass
        x = self.leaky_1(self.batchNorm1(self.reshape_1(self.dense_1(input_tensor)),training = training))
        for i in range(len(self.layers_blocks)):
            layers_block = self.layers_blocks[i]
            for layer in layers_block:
                x = layer(x, training = training)
        x = self.conv_toRGB(self.up_toRGB(x))
        return x
    
    def generate_noise(self,batch_size, random_noise_size):
        return tf.random.normal([batch_size, random_noise_size])

    def compute_loss(self,y_true,y_pred):
        return backend.mean(y_true * y_pred)

    def compute_loss_class(self,y_true,y_pred,class_wanted,class_prediction):
        """ Wasserstein loss - prob of classifier get it right
        """
        k = 10 # hiper-parameter
        return backend.mean(y_true * y_pred) + (k * categorical_crossentropy(class_wanted,class_prediction))

    def compute_loss_divergence(self,y_true,y_pred,class_wanted,class_prediction):
        k = 10 # hiper-parameter
        kl = KLDivergence()
        return backend.mean(y_true * y_pred) + (k * kl(class_wanted,class_prediction))

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def save_optimizer(self):
        weights = self.optimizer.get_weights()
        data_access.store_weights_in_file('g_optimizer_weights',weights)

    def set_seed(self):
        self.seed = tf.random.normal([self.batch_size, self.noise_size])
        data_access.store_seed_in_file('seed',self.seed)

    def load_seed(self):
        self.seed = data_access.load_seed_from_file('seed')
      
class Critic(tf.keras.Model):
    def __init__(self,model_parameters=None):
        super().__init__(name = "critic")
        if model_parameters is None:
            model_parameters = {
                'lr': 0.0001,
                'beta1': 0,
                'batch_size': 64,
                'latent_dim': 128,
                'image_size': 152
            }
        self.layers_blocks = list()
        self.model_parameters = model_parameters
        dim = model_parameters['batch_size'] / 2
        init = RandomNormal(stddev=0.02)
        #Layers
        self.conv_1 = Conv2D(dim, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, input_shape=[model_parameters['image_size'], model_parameters['image_size'], 3])
        self.leaky_1 = LeakyReLU(alpha=0.2)
        
        number_of_layers_needed = int(math.log(model_parameters['image_size'],2))-3
        for i in range(number_of_layers_needed):
            dim *= 2
            self.layers_blocks.append([
               Conv2D(dim, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init),
               LayerNormalization(),
               LeakyReLU(alpha=0.2)
            ])

        self.flat = Flatten()
        self.logits = Dense(1)  # This neuron tells us how real or fake the input is
        
        self.optimizer = Adam(learning_rate=model_parameters['lr'],beta_1=model_parameters['beta1'],beta_2=0.9)

    def call(self, input_tensor, training = True):
        ## Definition of Forward Pass
        x = self.leaky_1(self.conv_1(input_tensor))
        for i in range(len(self.layers_blocks)):
            layers_block = self.layers_blocks[i]
            for layer in layers_block:
                x = layer(x, training = training)
        x = self.flat(x)
        return self.logits(x)

    def compute_loss(self,y_true,y_pred):
        """ Wasserstein loss
        """
        return backend.mean(y_true * y_pred) 

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def save_optimizer(self):
        weights = self.optimizer.get_weights()
        data_access.store_weights_in_file('c_optimizer_weights',weights)
        
        
        
class WGAN_GP(tf.keras.Model):
    def __init__(self, model_parameters = None, classifier_filename=None,classifier_study_type = "class_convergence"):
        super().__init__(name = "BIG_WGAN")
        # 'TypeC_0.99_ConvExp19.hdf5'
        
        if model_parameters is None:
            model_parameters = {
                'lr': 0.0001,
                'beta1': 0,
                'batch_size': 64,
                'latent_dim': 128,
                'image_size': 152
            }
        # only accept power of 2 sizes
        model_parameters['image_size'] = 2**int(math.log(model_parameters['image_size'],2))
        self.model_parameters = model_parameters
        self.random_noise_size = model_parameters['latent_dim']
        self.generator = Generator(model_parameters)
        if('seed.npz' not in os.listdir('.')):
            self.generator.set_seed()
        else :
            self.generator.load_seed()
        self.critic = Critic(model_parameters)
        if ('weights' in os.listdir('.')):
            self.critic.load_weights('/content/weights/c_weights/c_weights')
            self.generator.load_weights('/content/weights/g_weights/g_weights')
        self.classifier = None
        if classifier_filename is not None:
            self.classifier = tf.keras.models.load_model(classifier_filename)
        self.classifier_study_type = classifier_study_type
            
        self.train_dataset = None
        self.test_dataset = None
        self.train_labels = None
        self.test_labels = None
        self.batch_size = model_parameters['batch_size']
        
    def load_dataset(self,dataset,n_classes):
        self.train_dataset,self.train_labels,self.test_dataset,self.test_labels = dataset
        self.num_classes = n_classes

    @tf.function
    def predict_batch(self,images,type_class):
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
            
            if self.classifier_study_type == 'class_convergence':
                y_list = tf.one_hot(type_class,self.num_classes)
            elif self.classifier_study_type == 'class_divergence':
                y_list = tf.convert_to_tensor(type_class,dtype=tf.float32)
                
                
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
            if self.classifier is not None:
                images_predictions, ys, matched_images = self.predict_batch(d_x,class_type)
                if self.classifier_study_type == 'class_convergence':
                    generator_loss = self.generator.compute_loss_class(d_z, y_g, ys, images_predictions)
                elif self.classifier_study_type == 'class_divergence':
                    generator_loss = self.generator.compute_loss_divergence(d_z, y_g, ys, images_predictions)
            else:
                images_predictions, ys, matched_images = [],[],[]
                generator_loss = self.generator.compute_loss(d_z, y_g)
        
        gradients_of_generator = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator.backPropagate(gradients_of_generator, self.generator.trainable_variables)
        return generator_loss,matched_images, self.generator(self.generator.seed, training=False)

    def generate_real_samples(self, n_samples):
        # choose random instances
        ix = np.random.randint(0, self.train_dataset.shape[0], n_samples)
        # select images
        X = self.train_dataset[ix]
        # associate with class labels of -1 for 'real'
        y = -np.ones((n_samples, 1)).astype(np.float32)
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
    
    def train_model(self,epoches,n_critic=5,class_type=0,directory = 'imgs',n_img_per_epoch=4):
        """
        Train model for an amount of epochs

        :param epoches: - cycles of training over all dataset
        :param n_critic: - number of times critic trains more than generator
        :param class_type: - it can be a number if class convergence is wanted
            or a probability distribution (array) discribing the probabilities for each class
            Example: class convergence(when there are 5 classes): 0
                     class divergence(when there are 5 classes): [0.25, 0.25, 0.25, 0.25, 0.0]
        :param directory: - directory where images will be placed during training
        :n_img_per_epoch: - number of images stored per epoch, while training
        """
        batch_per_epoch = int(self.train_dataset.shape[0] / self.batch_size)

        # calculate the number of training iterations
        n_steps = batch_per_epoch * epoches
        # calculate the size of half a batch of samples
        half_batch = int(self.batch_size / 2)

        sum_writer_loss = self.define_loss_tensorboard()
        avg_loss_critic = tf.keras.metrics.Mean()
        avg_loss_gen = tf.keras.metrics.Mean()
        try:
            epoch = int(open('current_epoch.txt').read())
        except:
            epoch = 0
        start_time = time.time()
        print(self.model_parameters)
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
                    data_access.write_current_epoch(filename='current_epoch',epoch=epoch)
        data_access.create_collection(epoches,n_img_per_epoch,directory)
        data_access.print_training_time(start_time,time.time(),self.model_parameters)

    def generate_images(self,number_of_samples,directory,class_names=None):
        seed = tf.random.normal([number_of_samples, self.random_noise_size])
        images = self.generator(seed,training=False)
        if self.classifier is not None: 
            predictions = self.classifier(data_access.normalize(data_access.de_standardize(images)))
            data_access.produce_generate_figure(directory,images,predictions,class_names)
        else:
            data_access.store_images_seed(directory,images,'gen_images','gan')
   