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
from wgan_gp_class_big import Big_WGAN

BATCH_SIZE = 64
EPOCHS = 1000

gan = Big_WGAN(BATCH_SIZE)
gan.load_dataset(data_access.prepare_data('gan',BATCH_SIZE),2)

gan.train_model(EPOCHS)
gan.save_weights('big_wgan.h5')
gan.generate_images(5,"imgs")


