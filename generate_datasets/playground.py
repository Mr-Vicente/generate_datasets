# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 08:45:21 2020
"""

"""
    Playground - play with models    
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""

import tensorflow as tf
import data_access
from generate_gan import GAN

fmnist = tf.keras.datasets.fashion_mnist
BATCH_SIZE = 256
EPOCHS = 20

gan = GAN()
gan.load_dataset(data_access.prepare_data(fmnist.load_data(),"gan",BATCH_SIZE))
gan.train_model(EPOCHS,BATCH_SIZE)
gan.generate_images(5,"imgs")


