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
from IntroVae import IntroVae


def main(args):
    
    vae = IntroVae()
    vae.load_dataset(data_access.prepare_data('vae'))
    
    vae.train_model()
    n_images = 10
    vae.generate_images(n_images,"imgs")



if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--batchsz', type=int, default=8, help='batch size')
    arg_parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension / Noise size')
    arg_parser.add_argument('--epochs', type=int, default=500, help='Learning iterations through the dataset')
    arg_parser.add_argument('--m', type=float, default=120, help='Margin')
    arg_parser.add_argument('--alpha', type=float, default=0.25, help='Control weight of adversarial loss (alpha * adv_loss)')
    arg_parser.add_argument('--beta', type=float, default=0.5, help='Control weight of encoder loss (beta * ae_loss)')
    arg_parser.add_argument('--gamma', type=float, default=1.0, help='gamma')     
    arg_parser.add_argument('--lr', type=float, default=0.0002, help='learning rate value') 
    
    args = arg_parser.parse_args()
    main(args)

