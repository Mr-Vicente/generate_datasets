# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 08:37:38 2020
"""

"""
    Data_Access
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt


def standardize(x):
    return (x - 127.5) / 127.5
def normalize(x):
    return x / 255.

def prepare_data(dataset,generator, batch_size = 1):
    (train_x, train_y),(test_x,test_y) = dataset

    train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1).astype('float32')

    if(generator == 'gan'):
        train_x = standardize(train_x)
        test_x = standardize(test_x)
    elif(generator == 'vae'):
        train_x = normalize(train_x)
        test_x = normalize(test_x)

        train_x[train_x >= .5] = 1
        train_x[train_x < .5] = 0
        test_x[test_x >= .5] = 1
        test_x[test_x < .5] = 0
        
    buffer_size_train = train_x.shape[0]
    buffer_size_test = test_x.shape[0]
    print(buffer_size_train)
    print(buffer_size_test)

    train_x = tf.data.Dataset.from_tensor_slices(train_x).shuffle(buffer_size_train).batch(batch_size)
    test_x = tf.data.Dataset.from_tensor_slices(test_x).shuffle(buffer_size_test).batch(batch_size)

    return train_x,train_y,test_x,test_y

def store_image(directory,image_name,image):
    plt.axis('off')
    plt.imsave('{}/{}.png'.format(directory,image_name),image, cmap="gray")

def prepare_directory(directory = "imgs"):
    if not os.path.exists(directory):
        os.makedirs(directory)