# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:39:18 2020

    152x152 PGGAN WITH CLASSIFIER
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense,BatchNormalization,Reshape,Conv2D,Dropout,Flatten,UpSampling2D,LeakyReLU,Cropping2D,AveragePooling2D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend
from tensorflow.keras.constraints import max_norm

import data_access
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
    def __init__(self,batches_size = None,random_noise_size = 128, isTrains=True, classifier_filename='TypeC_0.99_ConvExp19.hdf5'):
        super().__init__(name = "PGGAN")
        self.batches_per_phase = batches_size if batches_size else [16,16,16,16,16,16]
        self.noise_size = random_noise_size
        self.train_dataset = None
        if isTrains:
            self.generators = self.create_generators()
            self.critics = self.create_critics()
            self.wgans = self.create_WGANS(self.generators,self.critics)
        self.classifier = tf.keras.models.load_model(classifier_filename)
        
    #######################################
    '''             Generators          '''
    #######################################
    
    def create_first_generator(self):
        return Generator(m_type = ModelType.FIRST, noise_size= self.noise_size)
    
    def create_generators(self):
        models_list = list()
        first_gen = self.create_first_generator()
        models_list.append([first_gen,first_gen])
        
        for i in range(1,len(self.batches_per_phase)):
            last_model = models_list[i-1][0]
            gen = self.create_gen_block(last_model,i)
            gen_other = Generator(m_type = ModelType.LAST, old_model = last_model, curr_model = gen.layers[-1].output)
            models_list.append([gen,gen_other])
        
        return models_list
    
    def create_gen_block(self, old_model,i):
        gen = None
        old_layer =  old_model.layers[-2].output
        if(i == 2):
            gen = Generator(m_type = ModelType.SPECIAL, old_end_layer = old_layer)
        else:
            gen = Generator(m_type = ModelType.NORMAL, old_end_layer = old_layer)
        return gen
    
    #######################################
    '''             Critic              '''
    #######################################
            
    def create_last_critic(self):
        return Critic(m_type = ModelType.Last, input_shape = (5,5,3))
    
    def create_critics(self,n_input_layers):
        models_list = list()
        last_critic = self.create_last_critic()
        models_list.append([last_critic,last_critic])
        
        for i in range(1,len(self.batches_per_phase)):
            last_model = models_list[i-1][0]
            in_shape = list(last_model.input.shape)
            input_s = (in_shape[-2].value*2, in_shape[-2].value*2, in_shape[-1].value)
            critic = Critic(m_type = ModelType.NORMAL, input_shape = input_s)
            Avgp2d = critic.layers.get_layer('avgP2d')
            other_critic = Generator(m_type = ModelType.SPECIAL, old_model = last_model, block_new = Avgp2d.output)
            models_list.append([critic,other_critic])
        
        return models_list
    
    #######################################
    '''             Combos              '''
    #######################################
    
    def create_WGANS(generators, critics):
        model_list = list()
        for i in range(len(generators)):
            g_models, c_models = generators[i], critics[i]
            c_models[0].trainable = False
            gan_normal = WGAN(g_models[0],c_models[0])
            c_models[1].trainable = False
            gan_fadein = WGAN(g_models[1],c_models[1])
            model_list.append(gan_normal,gan_fadein)
        return model_list
    
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
        n = np.array([resize(image,new_shape) for image in images])
        print('Images Scaled')
        return n
    
    #######################################
    '''           Operations            '''
    #######################################
    def update_fadein(self, models, step, n_steps):
        
        alpha = step / float(n_steps -1)
        for mod in models:
            for layer in mod.layers:
                if isinstance(layer, WeightedSum):
                    backend.set_value(layer.alpha,alpha)
    
    
    #######################################
    '''              Train              '''
    #######################################
    def train(self,epoches=None,n_critic=1,class_type=0,directory = 'imgs',n_img_per_epoch = 4):
        
        epoches = epoches if epoches else [20,40,80,160,320,640]
        g_normal = self.generators[0][0]
        c_normal = self.critics[0][0]
        wgan_normal = self.wgans[0][0]
        gen_shape = g_normal.output_shape
        scaled_images = self.scale_dataset(self.train_dataset, gen_shape[1:])
        wgan_normal.train(scaled_images)
        
        for i in range(1, len(self.wgans)):
            
            [g_normal, g_fadein] = self.generators[i]
            [c_normal, c_fadein] = self.critics[i]
            [wgan_normal, wgan_fadein] = self.wgans[i]
            
            gen_shape = g_normal.output_shape
            scaled_images = self.scale_dataset(self.train_dataset, gen_shape[1:])
            
            #fadein train
            wgan_fadein.train(scaled_images)
            #normal train
            wgan_normal.train(scaled_images)

    # ??????????
    def train_epoches(self):
       raise NotImplementedError('Yet to be developed :)')


#######################################
'''              GENERATOR          '''
#######################################    
class Generator(tf.keras.Model):
    def __init__(self, m_type, **kwargs):
        super().__init__(name = "Generator")
        
        init = RandomNormal(stddev=0.2)
        const = max_norm(1.0)
        self.model_type = m_type
        
        if m_type == ModelType.FIRST:
            # define new input processing layer
            self.dense_1 = Dense(128 * 5 * 5, kernel_initializer=init, kernel_constraint=const, use_bias = False, input_shape = (kwargs.get('noise_size'),))
            self.reshape_1 = Reshape((5,5,128))
        elif m_type == ModelType.NORMAL:
            self.input = kwargs.get('old_end_layer')
            self.upsampling = UpSampling2D()
        elif m_type == ModelType.SPECIAL:
            self.crop_3 = Cropping2D(cropping=((1,0),(1,0)))
        
        if m_type != ModelType.LAST:
            # define new block
            self.conv_2 = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)
            self.pixel_2 = PixelNormalization()
            self.leaky_2 = LeakyReLU(alpha=0.2)
            self.conv_3 = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)
            self.pixel_3 = PixelNormalization()
            self.leaky_3 = LeakyReLU(alpha=0.2)
            self.out = Conv2D(3, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)      
        else:
            self.old_model = kwargs.get('old_model')
            self.curr_model_out = kwargs.get('curr_model_out')
            self.weigth_sum = WeightedSum()
            
        self.optimizer = Adam(learning_rate=0.0001,beta_1=0,beta_2=0.9)
            
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = input_tensor
        if self.model_type == ModelType.FIRST:
            x = self.reshape_1(self.dense_1(x))
        elif self.model_type != ModelType.FIRST:
            x = self.input(x)
            x = self.upsampling(x)
        if self.model_type == ModelType.SPECIAL:
            x = self.crop_3(x)
        if self.model_type != ModelType.LAST:
            x = self.leaky_2(self.pixel_2(self.conv_2(x)))
            x = self.leaky_3(self.pixel_3(self.conv_3(x)))
            x = self.out(x)
        else:
            x = self.old_model(x)
            x = self.weigth_sum([x,self.curr_model_out])
        return x
    
    def generate_noise(self,batch_size, random_noise_size):
        return tf.random.normal([batch_size, random_noise_size])

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def compute_loss(self,y_true,y_pred,class_wanted,class_prediction):
        """ Wasserstein loss - prob of classifier get it right
        """
        k = 10 # hiper-parameter
        return backend.mean(y_true * y_pred) + (k * categorical_crossentropy(class_wanted,class_prediction))

    def set_seed(self):
        self.seed = tf.random.normal([self.batch_s, self.noise_size])
        data_access.store_seed_in_file('seed',self.seed)

    def load_seed(self):
        self.seed = data_access.load_seed_from_file('seed')
        
        
#######################################
'''              Critic             '''
#######################################    
class Critic(tf.keras.Model):
    def __init__(self, m_type, **kwargs):
        super().__init__(name = "Critic")
        
        init = RandomNormal(stddev=0.2)
        const = max_norm(1.0)
        self.model_type = m_type
        kernel_last = (3,3)
        if m_type == ModelType.Last:
            kernel_last = (4,4)
            self.mini_batch_stdv = MinibatchStdev()
            self.flatten = Flatten()
            self.logit = Dense(1)
        elif m_type == ModelType.Normal or m_type == ModelType.SPECIAL:
            self.avg_pool = AveragePooling2D(name='avgP2d')
            self.old_model = kwargs.get('old_model')
        if m_type == ModelType.SPECIAL:
            self.block_new = kwargs.get('block_new')
            self.weigth_sum = WeightedSum()
        # define new block
        self.conv_1 = Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape = kwargs.get('input_shape'))
        self.leaky_1 = LeakyReLU(alpha=0.2)
        self.conv_2 = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)
        self.leaky_2 = LeakyReLU(alpha=0.2)
        self.conv_3 = Conv2D(128, kernel_last, padding='same', kernel_initializer=init, kernel_constraint=const)
        self.leaky_3 = LeakyReLU(alpha=0.2)    
        
        self.optimizer = Adam(learning_rate=0.0001,beta_1=0,beta_2=0.9)
        
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = input_tensor
        if self.model_type == ModelType.SPECIAL:
            #x = self.block_new(x)
            x = self.avg_pool(x)
            x = self.old_model.layers[1](x)
            x = self.old_model.layers[2](x)
            #y = input(x)
            x = self.weigth_sum([x,self.block_new])
            for i in range(1,len(self.old_model.layers)):
                x = self.old_model.layers[i](x)
        else:
            x = self.leaky_1(self.conv_1(x))
            if self.model_type == ModelType.Last:
                x = MinibatchStdev(x)
            x = self.leaky_2(self.conv_2(x))
            x = self.leaky_3(self.conv_3(x))
            if self.model_type == ModelType.Last:
                x = self.logit(self.flatten(x))
            elif self.model_type == ModelType.NORMAL:
                x = self.avg_pool(x)
                for i in range(1, len(self.old_model.layers)):
                    x = self.old_model.layers[i](x)
        return x

    def compute_loss(self,y_true,y_pred):
        """ Wasserstein loss
        """
        return backend.mean(y_true * y_pred) 

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
