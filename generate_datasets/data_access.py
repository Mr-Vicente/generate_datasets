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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio 
import glob

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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
    
def store_image_simple(directory,image_name,image,prediction):

    plt.axis('off')
    plt.imsave('{}/{}.png'.format(directory,image_name),image, cmap="gray")

def plot_image(i, predictions_array, images):
  predictions_array, img = predictions_array, images[i, :, :, 0]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)

  plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                100*np.max(predictions_array)),
                                color='red')

def produce_generate_figure(directory,gen_images,predictions):
    prepare_directory(directory)
    num_images = gen_images.shape[0]
    num_cols = 2
    num_rows = num_images/num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], gen_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i])
        _ = plt.xticks(range(10),class_names,rotation=80)
    plt.tight_layout()
    plt.savefig('{}/classifications.png'.format(directory))

def plot_value_array(i, predictions_array):
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')

def store_image(directory,epoch,image,i):
    prepare_directory(directory)
    prepare_directory('{}/epoch_{}'.format(directory,epoch))
    plt.axis('off')
    plt.xlabel(epoch)
    plt.imsave('{}/epoch_{}/id_{}.png'.format(directory,epoch,i),de_standardize(image[:,:,0]), cmap="gray")

def store_images_seed(directory,images,epoch):
    for i in range(len(images)):
        store_image(directory,epoch,images[i],i)


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

def create_collection(epoches,n_dif_images,directory):
    fig = plt.figure(figsize=(20,200))
    k=0
    for epoch in range(epoches):
        for im in range(n_dif_images):
            k+=1
            img = mpimg.imread('{}/epoch_{}/id_{}.png'.format(directory,epoch,im))
            fig.add_subplot(epoches, n_dif_images, k)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('training.png')
    
    
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
        