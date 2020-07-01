# -*- coding: utf-8 -*-
"""
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""

import argparse
import tensorflow as tf

from IntroVae.IntroVae import IntroVae

import scipy.optimize as optimize

def main(args):

    introVae = IntroVae()   
    classifier = tf.keras.models.load_model(args.classifier)
    
    def equal_class_probs(z):
        z_t = tf.convert_to_tensor(z,dtype=tf.float32)
        z_t = tf.reshape(z_t, shape=(1,128))
        img = introVae.sample(z_t).numpy()
        prediction = classifier(img)
        prediction = prediction[0][0].numpy()
        return [(prediction - 0.5) for i in range(128)]
    
    z = tf.random.normal([1, 128])
    z_optimal = optimize.fsolve(equal_class_probs, [z])
    print(z_optimal)
    
    img = introVae.sample(tf.convert_to_tensor([z_optimal],dtype=tf.float32)).numpy()
    prediction = classifier(img)
    prediction = prediction[0][0].numpy()
    print(prediction)


if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--classifier', type=str, default='classifiers/TypeC_0.99_ConvExp19.hdf5', help='classifier file')
    arg_parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension / Noise size')

    args = arg_parser.parse_args()
    main(args)

