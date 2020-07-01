# -*- coding: utf-8 -*-
"""
    Intro VAE parameters adjustment

    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""

import sys
sys.path.append("..")

import os
import data_access
import tensorflow as tf
from IntroVae import IntroVae
from tensorboard.plugins.hparams import api as hp
from itertools import product

class model_trainer():
    def __init__(self, params=None, epochs=200, batch_size=64, params_combination_file=''):
        if params == None:
            params = {
                'beta': [0.05,0.5],
                'lr': [0.0002,0.0001],
            }
        
        hparams, hparams_values = self.construct_hparams(params)
        if params_combination_file not in os.listdir('.'):
            information = {
                'keys': list(params.keys()),
                'combinations': hparams_values,
                'epochs': epochs,
                'batch_size': batch_size 
            }
            print(information)
            self.create_combination_file(information)
            
        self.set_tensorboard(hparams)
            
    def set_tensorboard(self,hparams):
        with tf.summary.create_file_writer('logs/hparam_tunning').as_default():
            hp.hparams_config(
                hparams=hparams,
                metrics=[hp.Metric('accuracy', display_name='Accuracy')]
            )
        
    def construct_hparams(self,params):
        hparams = list()
        hparams_values = list()
        for key,value in params.items():
            hparam = hp.HParam(key, hp.Discrete(value))
            hparams.append(hparam)
            hparams_values.append(value)
        return hparams,hparams_values
    
    def create_combination_file(self, information):
        all_params = information['combinations']
        information['combinations'] = [p for p in product(*all_params)]
        data_access.write_combinations('combinations',information) 
        data_access.write_current_phase_number('phase_number',0)

    def train_all(self,combinations_filename = 'combinations', phase_number = 'phase_number'):
        information = data_access.read_combinations(combinations_filename)
        combinations = information['combinations']
        keys = information['keys']
        current_params_index = data_access.read_phase_number(phase_number)
        for i in range(current_params_index,len(combinations)):
            current_params = combinations[i]
            print('Current params being tested: ', current_params)
            hparams = {}
            for j,value in enumerate(keys):
                hparams[value] = current_params[j]
            run_name = "run-%d" % i
            print('--- Starting trial: %s' % i)
            print({h: hparams[h] for h in hparams})
            self.train_introVae('logs/hparam_tunning/' + run_name, hparams,i)
            data_access.write_current_phase_number('phase_number',i+1)

    def train_introVae(self,run_dir, hparams,i,information):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            introVae = IntroVae()
            introVae.load_dataset(data_access.prepare_data('vae'))
            introVae.train_model(information['epochs'],information['batch_size'])
            random_vector_for_generation = tf.random.normal(shape=[4, introVae.latent_dim])
            images = introVae.sample(random_vector_for_generation)
            tf.summary.image(name='image',data=images,step=i)

        