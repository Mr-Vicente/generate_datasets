# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 08:37:38 2020
"""

"""
    Data_Access
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""

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
import imageio 
import glob


def standardize(x):
    return (x - 127.5) / 127.5

def de_standardize(x):
    return (x * 127.5) + 127.5

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
        
        train_x = tf.data.Dataset.from_tensor_slices(train_x).shuffle(buffer_size_train).batch(batch_size)
        test_x = tf.data.Dataset.from_tensor_slices(test_x).shuffle(buffer_size_test).batch(batch_size)
        

    return train_x,train_y,test_x,test_y

def store_image(directory,image_name,image):
    plt.axis('off')
    plt.imsave('{}/{}.png'.format(directory,image_name),image, cmap="gray")

def tf_store_image(image,epoch,i):
    img = tf.image.convert_image_dtype(image,tf.uint8)
    img = tf.image.encode_jpeg(img,quality=100)
    folder = 'step{}'.format(epoch)
    prepare_directory(folder)
    dirt= '{}/test{}.jpg'.format(folder,i)
    tf.io.write_file(dirt,img)

def store_images(images,epoch):
    fig = plt.figure(figsize=(2,2))
    for i in range(images.shape[0]):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

def create_gif(filename):
    anim_file = '{}.gif'.format(filename)
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    try:
        from google.colab import files
    except ImportError:
        pass
    else:
        files.download(anim_file)
        
def print_training_output(step, n_steps, critic_loss, gen_loss):
    print('-----------------------------')
    print('{}th OUT OF {} steps'.format(step,n_steps))
    print('Critic Loss: {}'.format(critic_loss))
    print('Generator Loss: {}'.format(gen_loss))

def prepare_directory(directory = "imgs"):
    if not os.path.exists(directory):
        os.makedirs(directory)
        