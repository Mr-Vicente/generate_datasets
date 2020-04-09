# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:03:58 2020

    152x152 PGGAN WITH CLASSIFIER
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense,Reshape,Conv2D,Flatten,UpSampling2D,LeakyReLU,Cropping2D,AveragePooling2D,InputLayer,ZeroPadding2D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend
from tensorflow.keras.constraints import max_norm

import data_access
import os
from PG_extra_classes import WeightedSum, MinibatchStdev,PixelNormalization
from PG_wgan_class import WGAN

import numpy as np
from skimage.transform import resize
from enum import Enum


class ModelType(Enum):
    FIRST = 1
    NORMAL = 2
    SPECIAL = 3
    LAST = 4
    
class PGGAN(tf.keras.Model):
    def __init__(self,batches_size = None,random_noise_size = 128, isTrains=True, classifier_filename='classifiers/TypeC_0.99_ConvExp19.hdf5'):
        super().__init__(name = "PGGAN")
        self.batches_per_phase = batches_size if batches_size else [16,16,16,16,16,16]
        self.shape_size = [5,10,19,38,76,152]
        self.noise_size = random_noise_size
        self.train_dataset = None
        if isTrains:
            self.wgan = self.create_Wgan()
        
    #######################################
    '''             Generators          '''
    #######################################
    
    def create_first_generator(self):
        return Generator(m_type = ModelType.FIRST, shape=(self.noise_size,))
    
    def create_generators(self):
        gen_list = list()
        gen_list.append(self.create_first_generator())
        for i in range(1,len(self.batches_per_phase)):
            gen = self.create_gen_block(i)
            gen_list.append(gen)
        return gen_list
    
    def create_gen_block(self,i):
        gen = None
        input_shape = (self.shape_size[i],self.shape_size[i],3)
        if(i == 2):
            gen = Generator(m_type = ModelType.SPECIAL,shape=input_shape)
        else:
            gen = Generator(m_type = ModelType.NORMAL,shape=input_shape)
        return gen
    
    #######################################
    '''             Critic              '''
    #######################################
            
    def create_last_critic(self):
        return Critic(m_type = ModelType.LAST, shape = (self.shape_size[0],self.shape_size[0],3))
    
    def create_critics(self):
        critic_list = list()
        critic_list.append(self.create_last_critic())
        for i in range(1,len(self.batches_per_phase)):
            critic = self.create_critic_block(i)
            critic_list.insert(-len(critic_list),critic)
        
        return critic_list
    
    def create_critic_block(self,i):
        critic = None
        input_shape = (self.shape_size[i],self.shape_size[i],3)
        if(i == 2):
            critic = Critic(m_type = ModelType.SPECIAL,shape=input_shape)
        else:
            critic = Critic(m_type = ModelType.NORMAL,shape=input_shape)
        return critic
    #######################################
    '''             Combos              '''
    #######################################
    '''  
    def create_Wgans(self):
        gen_list = self.create_generators()
        critic_list = self.create_critics()
        wgan_list = list()
        for i in range(len(self.batches_per_phase)):
            wgan = WGAN(gen_list[i], critic_list[i])
            wgan_list.append(wgan)
        return wgan_list
    '''
    def create_Wgan(self):
        gen_list = self.create_generators()
        critic_list = self.create_critics()
        generators = Collection_Generator(gen_list)
        critics = Collection_Critic(critic_list)
        wgan = WGAN(generators, critics)
        return wgan
    #######################################
    '''             Dataset             '''
    #######################################
    
    def load_dataset(self,dataset,n_classes):
        """
        Load images as numpy vectors and store dataset number of classes
        """
        self.train_dataset,_,_,_ = dataset
        self.num_classes = n_classes
        print('Dataset loaded')

    def scale_dataset(self, images, new_shape):
        """
        Scale images to given shape in form (x,y,color_channels)
        """
        image_filename = 'dataset_size_{}x{}.npz'.format(new_shape[0],new_shape[1])
        if image_filename in os.listdir('.'):
            print('Images Already Scaled -> Fetching..')
            n = np.load(image_filename,allow_pickle=True)['images_resized']
        else:
            n = np.array([resize(image,new_shape) for image in images])
            print('Images Scaled')
            np.savez(image_filename,images_resized=n)
        print('Fetched {}x{} images'.format(new_shape[0],new_shape[1]))
        return n
    
    #######################################
    '''              Train              '''
    #######################################
    def train(self,epoches=None,n_critic=1,class_type=0,directory = 'imgs',n_img_per_epoch = 4):
        
        epoches = epoches if epoches else [20,40,80,160,320,640]
        image_shape = (self.shape_size[0],self.shape_size[0],3)
        scaled_images = self.scale_dataset(self.train_dataset, new_shape=image_shape)
        self.wgan.load_dataset(scaled_images,2)
        self.wgan.train_model(epoches=epoches[0],b_size=0,n_critic=1,class_type=0,directory = 'imgs',n_img_per_epoch = 4)
        for i in range(1,5):
            self.wgan.start_fadein(i)
            image_shape = (self.shape_size[i],self.shape_size[i],3)
            scaled_images = self.scale_dataset(self.train_dataset, new_shape=image_shape)
            self.wgan.load_dataset(scaled_images,2)
            self.wgan.train_model(epoches=epoches[i],b_size=i,n_critic=1,class_type=0,directory = 'imgs',n_img_per_epoch = 4)
            self.wgan.end_fadein(i)
    
#######################################
'''           GENERATOR             '''
#######################################    
class Generator(tf.keras.Model):
    def __init__(self, m_type, shape):
        super().__init__(name = "Generator")
        
        self.init = RandomNormal(stddev=0.2)
        self.const = max_norm(1.0) # maybe byebye
        self.model_type = m_type
        self.input_layer = InputLayer(input_shape=(None,None,3))
        if m_type.value == ModelType.FIRST.value:
            # define new input processing layer
            self.input_layer = InputLayer(input_shape=shape)
            self.dense_1 = Dense(128 * 5 * 5, kernel_initializer=self.init, kernel_constraint=self.const, use_bias = False)
            self.reshape_1 = Reshape((5,5,128))
        else:
            self.upsampling_1 = UpSampling2D()
            self.weigth_sum = WeightedSum()
            
        if m_type.value == ModelType.SPECIAL.value:
            self.crop_3 = Cropping2D(cropping=((1,0),(1,0)))
    
        # define new block
        self.conv_2 = Conv2D(128, (3,3), padding='same', kernel_initializer=self.init, kernel_constraint=self.const)
        self.pixel_2 = PixelNormalization()
        self.leaky_2 = LeakyReLU(alpha=0.2)
        self.conv_3 = Conv2D(128, (3,3), padding='same', kernel_initializer=self.init, kernel_constraint=self.const)
        self.pixel_3 = PixelNormalization()
        self.leaky_3 = LeakyReLU(alpha=0.2)
        self.out = Conv2D(3, (3,3), padding='same', kernel_initializer=self.init, kernel_constraint=self.const)
        self.upsampling_2 =None

    def add_upsampling(self):
        self.upsampling_2 = UpSampling2D()
    def activate_fade_in(self):
        self.add_upsampling()
        
    def remove_upsampling(self):
        self.upsampling_2 = None

    def disactivate_fade_in(self):
        self.remove_upsampling()
    
    def call(self, input_tensor,fadein = False):
        ## Definition of Forward Pass
        x = self.input_layer(input_tensor)
        if self.model_type.value == ModelType.FIRST.value:
            x = self.reshape_1(self.dense_1(x))
        else:
            x = self.upsampling_1(x)
        if self.model_type.value == ModelType.SPECIAL.value:
            x = self.crop_3(x)

        x = self.leaky_2(self.pixel_2(self.conv_2(x)))
        x = self.leaky_3(self.pixel_3(self.conv_3(x)))
        x = self.out(x)
        
        if self.upsampling_2 is not None:
            y = self.upsampling_2(input_tensor)
            x = self.weigth_sum([x,y])
        return x
        
#######################################
'''              Critic             '''
#######################################    
class Critic(tf.keras.Model):
    def __init__(self, m_type, shape):
        super().__init__(name = "Critic")
        
        self.init = RandomNormal(stddev=0.2)
        self.const = max_norm(1.0)
        self.model_type = m_type
        
        self.input_layer = InputLayer(input_shape=(None,None,3))
        kernel_last = (3,3)
        if m_type.value == ModelType.LAST.value:
            kernel_last = (4,4)
            self.mini_batch_stdv = MinibatchStdev()
            self.flatten = Flatten()
            self.logit = Dense(1)
        else:
            self.weigth_sum = WeightedSum()
            self.avg_pool = AveragePooling2D()
            
        if m_type.value == ModelType.SPECIAL.value:
            self.zero_pad = ZeroPadding2D(padding=((1,0),(1,0)))
        
        # define new block
        self.conv_1 = Conv2D(128, (1,1), padding='same', kernel_initializer=self.init, kernel_constraint=self.const,input_shape=(None,None,3))
        self.leaky_1 = LeakyReLU(alpha=0.2)
        self.conv_2 = Conv2D(128, (3,3), padding='same', kernel_initializer=self.init, kernel_constraint=self.const)
        self.leaky_2 = LeakyReLU(alpha=0.2)
        self.conv_3 = Conv2D(128, kernel_last, padding='same', kernel_initializer=self.init, kernel_constraint=self.const)
        self.leaky_3 = LeakyReLU(alpha=0.2)    
        self.downsampling =None
    def add_downsampling(self):
        self.downsampling = AveragePooling2D()
        
    def add_toRGB(self):
        self.out = Conv2D(3, (3,3), padding='same', kernel_initializer=self.init, kernel_constraint=self.const)
        self.leakOut = LeakyReLU(alpha=0.2)  
        
    def activate_fade_in(self):
        self.add_downsampling()
        self.add_toRGB()
        
    def remove_downsampling(self):
        self.downsampling = None
    def remove_toRGB(self):
        self.out = None
        self.leakOut = None
    def disactivate_fade_in(self):
        self.remove_downsampling()
        self.remove_toRGB()
    
    
    def call(self, input_tensor):
        ## Definition of Forward Pass
        #x = self.input_layer(input_tensor)
        x = self.leaky_1(self.conv_1(input_tensor))
        if self.model_type.value == ModelType.LAST.value:
            x = self.mini_batch_stdv(x)
        x = self.leaky_2(self.conv_2(x))
        x = self.leaky_3(self.conv_3(x))
        if self.model_type.value == ModelType.LAST.value:
            x = self.logit(self.flatten(x))
        else: 
            x = self.avg_pool(x)
            if self.model_type.value == ModelType.SPECIAL.value : x = self.zero_pad(x)
        
        if self.downsampling is not None:
            y = self.downsampling(input_tensor)
            y = self.out(y)
            y = self.leakOut(y)
            x = self.weigth_sum([x,y])
        return x

#######################################
'''           GENERATORS             '''
#######################################    
class Collection_Generator(tf.keras.Model):
    def __init__(self, generators):
        super().__init__(name = "Coll_Generator")
        self.gens = generators
        self.optimizer = Adam(learning_rate=0.0001,beta_1=0,beta_2=0.9)
        self.n = 1
    def update_current_n_layer(self):
        self.n += 1
    def start_fading(self,n):
        self.gens[n-1].activate_fade_in()
        self.update_current_n_layer()
    def stop_fading(self,n):
        self.gens[n-1].disactivate_fade_in()
    def call(self, input_tensor):
        x = input_tensor
        for i in range(self.n):
            x = self.gens[i](x)
        return x
    def set_seed(self):
        self.seed = tf.random.normal([16, 100])
        data_access.store_seed_in_file('seed',self.seed)
    def load_seed(self):
        self.seed = data_access.load_seed_from_file('seed')
        
    def generate_noise(self,batch_size, random_noise_size):
        return tf.random.normal([batch_size, random_noise_size])
    
    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def compute_loss(self,y_true,y_pred,class_wanted,class_prediction):
        """ Wasserstein loss - prob of classifier get it right
        """
        k = 10 # hiper-parameter
        return backend.mean(y_true * y_pred)# + (k * categorical_crossentropy(class_wanted,class_prediction))


            
            
#######################################
'''            Critics              '''
#######################################    
class Collection_Critic(tf.keras.Model):
    def __init__(self, critics):
        super().__init__(name = "Coll_Critic")
        self.crit = critics
        self.optimizer = Adam(learning_rate=0.0001,beta_1=0,beta_2=0.9)
        self.n = 1
    def update_current_n_layer(self):
        self.n += 1
    def start_fading(self,n):
        self.crit[len(self.crit)-n].activate_fade_in()
        self.update_current_n_layer()
    def stop_fading(self,n):
        self.crit[len(self.crit)-n].disactivate_fade_in()
    def call(self, input_tensor):
        x = input_tensor
        for i in range(len(self.crit)-self.n,len(self.crit)):
            x = self.crit[i](x)
        return x
    def compute_loss(self,y_true,y_pred):
        """ Wasserstein loss
        """
        return backend.mean(y_true * y_pred) 

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))