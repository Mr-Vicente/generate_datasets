from classifier import Classifier
import data_access
from spectral_normalization import SpectralNormalization

import tensorflow as tf
from tensorflow.keras.layers import Dense,BatchNormalization,Reshape,Conv2D,Dropout,Flatten,UpSampling2D,LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend

import numpy as np
#import matplotlib.pyplot as plt
import time
from IPython import display
from datetime import datetime

class Generator(tf.keras.Model):
    
    def __init__(self, random_noise_size = 100,batch_s = 64):
        super().__init__(name='generator')
        #layers
        init = RandomNormal(stddev=0.2)
        self.dense_1 = Dense(7*7*batch_s, use_bias = False, input_shape = (random_noise_size,))
        self.batchNorm1 = BatchNormalization()
        self.leaky_1 = LeakyReLU(alpha=0.2)
        self.reshape_1 = Reshape((7,7,batch_s))
        
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

        self.optimizer = Adam(learning_rate=0.0001,beta_1=0,beta_2=0.9) #RMSprop(lr=0.00005)
        self.seed = tf.random.normal([batch_s, random_noise_size])
               
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.reshape_1(self.leaky_1(self.batchNorm1(self.dense_1(input_tensor))))
        x = self.leaky_2(self.batchNorm2(self.conv2(self.up_2(x))))
        x = self.leaky_3(self.batchNorm3(self.conv3(self.up_3(x))))
        return  self.conv4(self.up_4(x))
    
    def generate_noise(self,batch_size, random_noise_size):
        return tf.random.normal([batch_size, random_noise_size])

    def compute_loss(self,y_true,y_pred,class_wanted,class_prediction):
        """ Wasserstein loss - prob of classifier get it right
        """
        k = 10 # hiper-parameter
        return backend.mean(y_true * y_pred) + (k * categorical_crossentropy(class_wanted,class_prediction))

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        
class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__(name = "critic")
        
        init = RandomNormal(stddev=0.2)
        #Layers
        self.conv_1 = SpectralNormalization(Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init, input_shape=[28, 28, 1]))
        self.leaky_1 = LeakyReLU(alpha=0.2)
        self.dropout_1 = Dropout(0.3)
        
        self.conv_2 = SpectralNormalization(Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init))
        self.leaky_2 = LeakyReLU(alpha=0.2)
        self.dropout_2 = Dropout(0.3)

        self.flat = Flatten()
        self.logits = Dense(1)  # This neuron tells us if the input is fake or real
        
        self.optimizer = Adam(learning_rate=0.0001,beta_1=0,beta_2=0.9)
        
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.dropout_1(self.leaky_1(self.conv_1(input_tensor)))
        x = self.dropout_2(self.leaky_2(self.conv_2(x)))
        x = self.flat(x)
        return self.logits(x)

    def compute_loss(self,y_true,y_pred):
        """ Wasserstein loss
        """
        return backend.mean(y_true * y_pred) 

    def backPropagate(self,gradients,trainable_variables):
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        
        
class WGAN(tf.keras.Model):
    def __init__(self,batch_size = 64):
        super().__init__(name = "WGAN")
        self.generator = Generator(100,batch_size)
        self.critic = Critic()
        self.classifier_m = Classifier()
               
        self.train_dataset = None
        self.test_dataset = None
        self.train_labels = None
        self.test_labels = None
        self.batch_size = batch_size
        
    def load_dataset(self,dataset,n_classes):
        self.train_dataset,self.train_labels,self.test_dataset,self.test_labels = dataset
        self.num_classes = n_classes

    @tf.function
    def predict_batch(self,images,type_class):
        images_predictions = tf.TensorArray(tf.float32,size=10,dynamic_size=True)
        ys = tf.TensorArray(tf.float32,size=10,dynamic_size=True)
        matched_images = tf.TensorArray(tf.float32,size=0,dynamic_size=True)
        index = 0
        for i in tf.range(len(images)):
            gen_image = data_access.normalize(data_access.de_standardize(images[i]))
            img = tf.expand_dims(gen_image,axis=0)
            c_type = self.classifier_m.predict_image(img)
            w_list = tf.one_hot(c_type,self.num_classes)
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
        X = self.train_dataset[ix]
        # associate with class labels of -1 for 'real'
        y = -np.ones((n_samples, 1)).astype(np.float32)
        return X, y

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
        logdir="logs/train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        return tf.summary.create_file_writer(logdir=logdir)

    def define_graph_tensorboard(self):
        logdir="logs/graph/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        return tf.summary.create_file_writer(logdir=logdir)
    
    def train_model(self,epoches,n_critic=5,noise_size=100,class_type=5):

        batch_per_epoch = int(self.train_dataset.shape[0] / self.batch_size)

        # calculate the number of training iterations
        n_steps = batch_per_epoch * epoches
        # calculate the size of half a batch of samples
        half_batch = int(self.batch_size / 2)

        sum_writer_loss = self.define_loss_tensorboard()
        self.classifier_m.load_local_model()
        avg_loss_critic = tf.keras.metrics.Mean()
        avg_loss_gen = tf.keras.metrics.Mean()
        epoch = 0
        n_dif_images = 4
        directory = 'imgs'
        start_time = time.time()
        for i in range(n_steps):
            for _ in range(n_critic):
                # get randomly selected 'real' samples
                X_real, y_real = self.generate_real_samples(half_batch)
                # generate 'fake' examples
                X_fake, y_fake = self.generate_fake_samples(noise_size, half_batch)
                
                # update critic model weights
                c_loss = self.training_step_critic(X_real,X_fake, y_real,y_fake,half_batch)
                avg_loss_critic(c_loss)
                
            gen_loss, matched_images, gen_images = self.training_step_generator(noise_size,class_type)
            avg_loss_gen(gen_loss)
            data_access.print_training_output(i,n_steps, avg_loss_critic.result(),avg_loss_gen.result()) 
            
            if((i % (n_steps / epoches)) == 0):
                data_access.store_images_seed(directory,gen_images[:n_dif_images],epoch)
                with sum_writer_loss.as_default():
                    tf.summary.scalar('loss_gen', avg_loss_gen.result(),step=self.generator.optimizer.iterations)
                    tf.summary.scalar('avg_loss_critic', avg_loss_critic.result(),step=self.critic.optimizer.iterations)
                epoch += 1
        data_access.create_collection(epoches,n_dif_images,directory)
        print('Time elapse {}'.format(time.time() - start_time))

    def generate_images(self,number_of_samples,directory):
        seed = tf.random.normal([number_of_samples, 100])
        images = self.generator(seed)
        predictions = self.classifier_m.predict_image_vector(data_access.normalize(data_access.de_standardize(images)))
        data_access.produce_generate_figure('imgs',images,predictions)
      
