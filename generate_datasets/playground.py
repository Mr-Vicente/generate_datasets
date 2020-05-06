# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 08:45:21 2020
"""

"""
    Playground - play with the models    
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""
import argparse
import data_access
#import tensorflow as tf
from PGGAN_class import PGGAN

'''
BATCH_SIZE = 64
EPOCHS = 1000
NOISE_SIZE = 128
'''

'''
gan = Big_WGAN(BATCH_SIZE,NOISE_SIZE)
gan.load_dataset(data_access.prepare_data('gan',BATCH_SIZE),2)

gan.train_model(EPOCHS)
gan.generate_images(10,"imgs")
'''
def main(args):
    '''
    introVae = IntroVae()
    introVae.load_dataset(data_access.prepare_data('vae'))
    introVae.train(args.epochs)
    '''
    pggan = PGGAN()
    e = args.epochs
    EPOCHS = [e,e,e,e,e,e]
    pggan.load_dataset(data_access.prepare_data('gan'),2)
    pggan.train(EPOCHS)



if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add('--batchsz', type=int, default=8, help='batch size')
    arg_parser.add('--z_dim', type=int, default=128, help='Latent dimension / Noise size')
    arg_parser.add('--epochs', type=int, default=500, help='Learning iterations through the dataset')
    arg_parser.add('--alpha', type=float, default=0.25, help='Control weight of adversarial loss (alpha * adv_loss)')
    arg_parser.add('--beta', type=float, default=0.5, help='Control weight of encoder loss (beta * ae_loss)')
    arg_parser.add('--gamma', type=float, default=1.0, help='gamma')     
    arg_parser.add('--lr', type=float, default=0.0002, help='learning rate value') 
    
    args = arg_parser.parse_args()
    main(args)

