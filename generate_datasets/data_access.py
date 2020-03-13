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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import imageio 
import glob

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names = ['TypeC','TypeO']

def standardize(x):
    return (x - 127.5) / 127.5

def de_standardize(x):
    return abs((x * 127.5) + 127.5)

def de_standardize_norm(x):
    return np.interp(x,[-1,1],[0,1])

def normalize(x):
    return x / 255.

def de_normalize(x):
    return x * 255.

def load_data(data_type = '.png', data_dir='images', resize_shape=(152,152)):
    if (data_type == '.png'):
        return np.array([cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(data_dir, img)),dsize=resize_shape),cv2.COLOR_BGR2RGB) for img in os.listdir(data_dir) if img.endswith(data_type)])
    elif (data_type == '.npz'):
        images = list()
        for img in os.listdir(data_dir):
            currentSetOfImages = np.load(os.path.join(data_dir, img),'r')
            print(currentSetOfImages.files)
            imgs = currentSetOfImages['images']
            images.append(imgs)
        return images, len(os.listdir(data_dir))
      
'''
def prepare_data(dataset,generator, batch_size = 1, resize=None):
    (train_x, train_y),(test_x,test_y) = dataset
    
    if(resize is not None):
        train_x = tf.image.resize(train_x, resize)
        test_x = tf.image.resize(test_x, resize)
        
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
 
def prepare_data(dataset,generator, batch_size = 1):
    train_x = dataset
    
    train_x = tf.convert_to_tensor(train_x,dtype=tf.float32)
  
    if(generator == 'gan'):
        train_x = standardize(train_x)
    elif(generator == 'vae'):
        train_x = normalize(train_x)

        train_x[train_x >= .5] = 1
        train_x[train_x < .5] = 0
        test_x[test_x >= .5] = 1
        test_x[test_x < .5] = 0
        
        buffer_size_train = train_x.shape[0]
        buffer_size_test = test_x.shape[0]
        
        train_x = tf.data.Dataset.from_tensor_slices(train_x).shuffle(buffer_size_train).batch(batch_size)
        test_x = tf.data.Dataset.from_tensor_slices(test_x).shuffle(buffer_size_test).batch(batch_size)
        

    return train_x, None, None, None
'''
def prepare_data(generator, batch_size = 1,data_dir='imgs'):

    train_x, npzs = load_data(data_type = '.npz', data_dir=data_dir)
    images_size = 5000 * npzs
    train_x = tf.convert_to_tensor(train_x,dtype=tf.float32)
    print(train_x.shape)
    train_x = tf.reshape(train_x,shape=(images_size,152,152,3))
    print(train_x.shape)

    if(generator == 'gan'):
        train_x = standardize(train_x)

    elif(generator == 'vae'):
        train_x = normalize(train_x)
        buffer_size_train = train_x.shape[0]
        train_x = tf.data.Dataset.from_tensor_slices(train_x).shuffle(buffer_size_train).batch(batch_size,drop_remainder=True)

    return train_x,None, None, None

def get_images_of_certain_class(csv_filename,data_dir,class_type):
    images,npzs = load_data(data_type = '.npz', data_dir=data_dir)
    images_size = 5000 * npzs
    images = tf.convert_to_tensor(images,dtype=tf.float32)
    images = tf.reshape(images,shape=(images_size,152,152,3))
    df = pd.read_csv(csv_filename,sep=',')
    #class_type = TypeA (for example)
    df = df.head(images_size)
    class_images = df.loc[df[class_type] == 1]
    ix = class_images['name']
    print(ix)
    images = tf.gather(images,ix) #images[ix]
    store_images_seed("images",images,"none")


def prepare_img(img,type_de = 'gan'):
    img = img.numpy()
    if (type_de == 'gan'):
        img = de_standardize_norm(img)
    else:
        print(img)
    return img

def store_image_simple(directory,image_name,image,prediction):

    plt.axis('off')
    plt.imsave('{}/{}.png'.format(directory,image_name),image)

def plot_image(i, predictions_array, images):
  img = prepare_img(images[i])
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img)

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
        _ = plt.xticks(range(len(class_names)),class_names,rotation=80)
    plt.tight_layout()
    plt.savefig('{}/classifications.png'.format(directory))

def plot_value_array(i, predictions_array):
  plt.grid(False)
  plt.xticks(range(len(class_names)))
  plt.yticks([])
  thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')

def store_image(directory,epoch,image,i,type_de):
    prepare_directory(directory)
    prepare_directory('{}/epoch_{}'.format(directory,epoch))
    plt.axis('off')
    plt.xlabel(epoch)
    image = prepare_img(image,type_de)

    plt.imsave('{}/epoch_{}/id_{}.png'.format(directory,epoch,i),image)

def store_images_seed(directory,images,epoch,type_de='gan'):
    for i in range(len(images)):
        store_image(directory,epoch,images[i],i,type_de)


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
        image = prepare_img(images[i])
        plt.imshow(image)
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
        