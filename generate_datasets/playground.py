# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 08:45:21 2020
"""

"""
    Playground - play with the models    
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""

import data_access
#import tensorflow as tf
from PGGAN_class import PGGAN

'''
BATCH_SIZE = 64
EPOCHS = 1000
NOISE_SIZE = 128
'''

pggan = PGGAN()
EPOCHS = [2,2,2,2,2,2]
pggan.load_dataset(data_access.prepare_data('gan'),2)
pggan.train(EPOCHS)


'''
gan = Big_WGAN(BATCH_SIZE,NOISE_SIZE)
gan.load_dataset(data_access.prepare_data('gan',BATCH_SIZE),2)

gan.train_model(EPOCHS)
gan.generate_images(10,"imgs")
'''

