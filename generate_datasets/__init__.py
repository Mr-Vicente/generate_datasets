__version__ = '0.1'
__author__ = 'Frederico Vicente & Ludwig Krippahl'
__description__ = 'Library to generate datasets using GAN/VAE'

__all__ = ['classifier','data_access','generate_gan' \
           'generate_vae','generate_wgan_gp_class','generate_wgan_gp' \
           'IntroVae','lsun_preprocess','playground' \
           'spectral_normalization','train','wgan_gp_class_big' ]

from generate_datasets import   generate_gan, \
                                generate_vae, \
                                classifier, \
                                data_access
                                
